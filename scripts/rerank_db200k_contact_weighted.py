#!/usr/bin/env python3
"""Rerank DB200K hits using contact-weighted structure terms.

This differs from mean-RSA/mean-pLDDT reranking by weighting only the target
residues that actually carry favorable DB200K contributions. Offset rescues are
mapped onto the donor residue that provides the favorable contact.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contact_alignment import db200k_scan  # noqa: E402
from scripts.scan_db200k_accessibility import compute_sequence_bonus  # noqa: E402


def load_db200k_cli():
    spec = importlib.util.spec_from_file_location(
        "_db200k_cli",
        REPO_ROOT / "scripts" / "_db200k_cli.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load _db200k_cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_surface_walk():
    spec = importlib.util.spec_from_file_location(
        "surface_walk_module",
        REPO_ROOT / "src" / "surface_walk" / "surface_walk.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load surface_walk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    cli = load_db200k_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    cli.add_profile_args(parser)
    cli.add_alignment_args(parser)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--seq-start", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--contact-threshold", type=float, default=0.0)
    parser.add_argument("--rsa-weight", type=float, default=1.0)
    parser.add_argument("--disorder-weight", type=float, default=1.0)
    parser.add_argument("--protrusion-weight", type=float, default=1.0)
    parser.add_argument("--neighbor-radius", type=float, default=12.0)
    parser.add_argument("--exclude-seq-neighbors", type=int, default=2)
    parser.add_argument("--protrusion-assembly-cif", type=str, default=None)
    parser.add_argument("--protrusion-assembly-id", type=str, default="1")
    parser.add_argument("--protrusion-target-chain", type=str, default=None)
    parser.add_argument("--protrusion-template-pdb", type=str, default=None)
    parser.add_argument("--protrusion-template-chain", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def percentile_from_rank(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - ((rank - 1) / (total - 1))


def build_protrusion_lookup(
    table: pd.DataFrame,
    *,
    neighbor_radius: float,
    exclude_seq_neighbors: int,
) -> dict[int, float]:
    coords = table[["ca_x", "ca_y", "ca_z"]].to_numpy(dtype=float)
    resseqs = table["resseq"].to_numpy(dtype=int)
    protrusion: dict[int, float] = {}
    radius2 = neighbor_radius * neighbor_radius
    for i, resseq in enumerate(resseqs):
        deltas = coords - coords[i]
        dist2 = (deltas * deltas).sum(axis=1)
        seq_sep = abs(resseqs - resseq)
        mask = (dist2 <= radius2) & (seq_sep > exclude_seq_neighbors)
        # Higher score means fewer nonlocal neighbors, i.e. more protruding.
        neighbor_count = int(mask.sum())
        protrusion[resseq] = 1.0 / (1.0 + neighbor_count)
    return protrusion


def _parse_oper_expression(expr: str) -> list[str]:
    parts: list[str] = []
    for token in expr.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            parts.extend(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            parts.append(token)
    return parts


def build_assembly_protrusion_lookup(
    cif_path: str,
    *,
    assembly_id: str,
    target_chain: str,
    neighbor_radius: float,
    exclude_seq_neighbors: int,
    template_pdb: str | None = None,
    template_chain: str | None = None,
) -> dict[int, float]:
    mmcif = MMCIF2Dict(cif_path)
    assembly_ids = mmcif.get("_pdbx_struct_assembly_gen.assembly_id", [])
    oper_exprs = mmcif.get("_pdbx_struct_assembly_gen.oper_expression", [])
    asym_lists = mmcif.get("_pdbx_struct_assembly_gen.asym_id_list", [])
    op_ids = mmcif.get("_pdbx_struct_oper_list.id", [])
    m11 = mmcif.get("_pdbx_struct_oper_list.matrix[1][1]", [])
    m12 = mmcif.get("_pdbx_struct_oper_list.matrix[1][2]", [])
    m13 = mmcif.get("_pdbx_struct_oper_list.matrix[1][3]", [])
    v1 = mmcif.get("_pdbx_struct_oper_list.vector[1]", [])
    m21 = mmcif.get("_pdbx_struct_oper_list.matrix[2][1]", [])
    m22 = mmcif.get("_pdbx_struct_oper_list.matrix[2][2]", [])
    m23 = mmcif.get("_pdbx_struct_oper_list.matrix[2][3]", [])
    v2 = mmcif.get("_pdbx_struct_oper_list.vector[2]", [])
    m31 = mmcif.get("_pdbx_struct_oper_list.matrix[3][1]", [])
    m32 = mmcif.get("_pdbx_struct_oper_list.matrix[3][2]", [])
    m33 = mmcif.get("_pdbx_struct_oper_list.matrix[3][3]", [])
    v3 = mmcif.get("_pdbx_struct_oper_list.vector[3]", [])

    ops: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, op_id in enumerate(op_ids):
        mat = np.array(
            [
                [float(m11[i]), float(m12[i]), float(m13[i])],
                [float(m21[i]), float(m22[i]), float(m23[i])],
                [float(m31[i]), float(m32[i]), float(m33[i])],
            ],
            dtype=float,
        )
        vec = np.array([float(v1[i]), float(v2[i]), float(v3[i])], dtype=float)
        ops[op_id] = (mat, vec)

    selected_ops: list[str] = []
    selected_chains: set[str] = set()
    for aid, expr, asym in zip(assembly_ids, oper_exprs, asym_lists):
        if str(aid) != str(assembly_id):
            continue
        selected_ops.extend(_parse_oper_expression(expr))
        selected_chains.update(ch.strip() for ch in asym.split(",") if ch.strip())
    selected_ops = list(dict.fromkeys(selected_ops))

    base_chain_coords: dict[str, list[tuple[int, np.ndarray]]] = {}
    target_base: list[tuple[int, np.ndarray]] = []

    if template_pdb:
        parser = MMCIFParser(QUIET=True) if template_pdb.endswith(".cif") else None
        if parser is not None:
            structure = parser.get_structure("template", template_pdb)
            model = next(structure.get_models())
        else:
            from Bio.PDB import PDBParser

            structure = PDBParser(QUIET=True).get_structure("template", template_pdb)
            model = next(structure.get_models())
        source_chain_id = template_chain or target_chain
        source_chain = model[source_chain_id]
        residues: list[tuple[int, np.ndarray]] = []
        for residue in source_chain.get_residues():
            if residue.id[0] != " " or "CA" not in residue:
                continue
            resseq = int(residue.id[1])
            coord = residue["CA"].coord.astype(float)
            residues.append((resseq, coord))
        for chain_id in selected_chains:
            base_chain_coords[chain_id] = residues
        target_base = residues
    else:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("assembly", cif_path)
        model = next(structure.get_models())
        for chain in model.get_chains():
            if chain.id not in selected_chains:
                continue
            residues = []
            for residue in chain.get_residues():
                if residue.id[0] != " " or "CA" not in residue:
                    continue
                resseq = int(residue.id[1])
                coord = residue["CA"].coord.astype(float)
                residues.append((resseq, coord))
            if residues:
                base_chain_coords[chain.id] = residues
            if chain.id == target_chain:
                target_base = residues
    if not target_base:
        raise ValueError(f"Target chain {target_chain!r} not found in assembly cif")

    expanded_coords: list[tuple[str, str, int, np.ndarray]] = []
    for op_id in selected_ops:
        mat, vec = ops[op_id]
        for chain_id, residues in base_chain_coords.items():
            for resseq, coord in residues:
                expanded_coords.append((op_id, chain_id, resseq, mat @ coord + vec))

    radius2 = neighbor_radius * neighbor_radius
    lookup: dict[int, float] = {}
    for resseq, coord in target_base:
        count = 0
        for op_id, chain_id, other_resseq, other_coord in expanded_coords:
            if op_id == "1" and chain_id == target_chain and abs(other_resseq - resseq) <= exclude_seq_neighbors:
                continue
            delta = other_coord - coord
            if float(delta @ delta) <= radius2:
                count += 1
        lookup[resseq] = 1.0 / (1.0 + count)
    return lookup


def main() -> None:
    args = parse_args()
    cli = load_db200k_cli()
    surface_walk = load_surface_walk()
    profiles = cli.build_profiles_from_args(args)
    window_len = len(profiles)
    table = surface_walk.build_residue_table(args.pdb, args.chain)
    if args.protrusion_assembly_cif:
        protrusion_lookup = build_assembly_protrusion_lookup(
            args.protrusion_assembly_cif,
            assembly_id=args.protrusion_assembly_id,
            target_chain=args.protrusion_target_chain or args.chain,
            neighbor_radius=args.neighbor_radius,
            exclude_seq_neighbors=args.exclude_seq_neighbors,
            template_pdb=args.protrusion_template_pdb,
            template_chain=args.protrusion_template_chain,
        )
    else:
        protrusion_lookup = build_protrusion_lookup(
            table,
            neighbor_radius=args.neighbor_radius,
            exclude_seq_neighbors=args.exclude_seq_neighbors,
        )

    rows: list[dict[str, object]] = []
    for header, sequence in db200k_scan.iter_fasta_records(args.fasta):
        if len(sequence) < window_len:
            continue
        for start0 in range(len(sequence) - window_len + 1):
            window = sequence[start0 : start0 + window_len]
            if any(res not in db200k_scan.CENTER_ALPHABET_SET for res in window):
                continue

            db200k_score, breakdown, donor_indices = db200k_scan.score_window_with_donor_trace(
                window,
                profiles,
                score_mode=args.score_mode,
                uncertainty_floor=args.uncertainty_floor,
            )
            seq_bonus, _ = compute_sequence_bonus(
                window,
                surface_weight=0.25,
                flexibility_weight=0.20,
                polar_weight=0.10,
                gp_weight=0.25,
                hydrophobe_penalty=0.15,
                complexity_weight=0.30,
                transition_weight=0.20,
                repeat_penalty=0.40,
                acidic_run_penalty=0.60,
                basic_run_penalty=0.55,
                charged_run_penalty=0.90,
                acidic_excess_penalty=0.80,
            )
            sequence_rank_score = db200k_score - seq_bonus

            residue_weights: dict[int, float] = {}
            for idx, (_, _label, energy) in enumerate(breakdown):
                if energy < args.contact_threshold:
                    donor = donor_indices[idx]
                    residue_weights[donor] = residue_weights.get(donor, 0.0) + (-energy)

            if residue_weights:
                weight_total = sum(residue_weights.values())
                weighted_rsa = 0.0
                weighted_disorder = 0.0
                weighted_protrusion = 0.0
                for donor_idx, weight in residue_weights.items():
                    resseq = args.seq_start + start0 + donor_idx
                    hit = table[table["resseq"] == resseq]
                    if hit.empty:
                        continue
                    rsa = float(hit["rsa"].iloc[0])
                    disorder = 1.0 - (float(hit["pLDDT"].iloc[0]) / 100.0)
                    protrusion = protrusion_lookup.get(resseq, 0.0)
                    weighted_rsa += weight * rsa
                    weighted_disorder += weight * disorder
                    weighted_protrusion += weight * protrusion
                weighted_rsa /= weight_total
                weighted_disorder /= weight_total
                weighted_protrusion /= weight_total
            else:
                weighted_rsa = 0.0
                weighted_disorder = 0.0
                weighted_protrusion = 0.0

            rows.append(
                {
                    "header": header,
                    "start": start0 + 1,
                    "end": start0 + window_len,
                    "window": window,
                    "db200k_score": db200k_score,
                    "seq_bonus": seq_bonus,
                    "sequence_rank_score": sequence_rank_score,
                    "contact_weighted_rsa": weighted_rsa,
                    "contact_weighted_disorder": weighted_disorder,
                    "contact_weighted_protrusion": weighted_protrusion,
                    "contact_weight_total": sum(residue_weights.values()) if residue_weights else 0.0,
                }
            )

    df = pd.DataFrame(rows)
    df["rsa_rank"] = df["contact_weighted_rsa"].rank(method="min", ascending=False).astype(int)
    df["disorder_rank"] = df["contact_weighted_disorder"].rank(method="min", ascending=False).astype(int)
    df["protrusion_rank"] = df["contact_weighted_protrusion"].rank(method="min", ascending=False).astype(int)
    total = len(df)
    df["rsa_pct"] = df["rsa_rank"].map(lambda r: percentile_from_rank(int(r), total))
    df["disorder_pct"] = df["disorder_rank"].map(lambda r: percentile_from_rank(int(r), total))
    df["protrusion_pct"] = df["protrusion_rank"].map(lambda r: percentile_from_rank(int(r), total))
    denom = args.rsa_weight + args.disorder_weight + args.protrusion_weight
    df["contact_struct_score"] = (
        args.rsa_weight * df["rsa_pct"]
        + args.disorder_weight * df["disorder_pct"]
        + args.protrusion_weight * df["protrusion_pct"]
    ) / denom
    df["contact_rerank_score"] = df["sequence_rank_score"] - df["contact_struct_score"]
    df = df.sort_values(["contact_rerank_score", "sequence_rank_score"])

    if args.out:
        df.to_csv(args.out, sep="\t", index=False)

    writer = csv.writer(sys.stdout, delimiter="\t")
    writer.writerow(
        [
            "start",
            "end",
            "window",
            "sequence_rank_score",
            "contact_weighted_rsa",
            "contact_weighted_disorder",
            "contact_weighted_protrusion",
            "contact_struct_score",
            "contact_rerank_score",
            "contact_weight_total",
        ]
    )
    for _, row in df.head(args.top_k).iterrows():
        writer.writerow(
            [
                int(row["start"]),
                int(row["end"]),
                row["window"],
                f'{row["sequence_rank_score"]:.6f}',
                f'{row["contact_weighted_rsa"]:.6f}',
                f'{row["contact_weighted_disorder"]:.6f}',
                f'{row["contact_weighted_protrusion"]:.6f}',
                f'{row["contact_struct_score"]:.6f}',
                f'{row["contact_rerank_score"]:.6f}',
                f'{row["contact_weight_total"]:.6f}',
            ]
        )


if __name__ == "__main__":
    main()
