#!/usr/bin/env python3
"""Second-pass orientation scoring for mapped pocket hits.

Computes rigid-body orientation diagnostics from residue mappings:
- Kabsch rotation determinant (det(R))
- Post-Kabsch coordinate RMSD on SC points
- Per-position sidechain orientation angles (template vs target, in degrees)
- Orientation summary metrics and optional blended ranking with geometry

Input is a TSV from an upstream mapping/ranking step with columns that contain
mapped residue tokens (for example: "cluster1", "v247_map", "l177_map", ...).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


DEFAULT_TEMPLATE_RES = "H175,D173,R176,Y323,V247,L177,L248"
DEFAULT_MAPPING_COLS = "cluster1,v247_map,l177_map,l248_map_vilmf"
DEFAULT_AXIS_SYMM_AAS = "V,L,I,A,M"
DEFAULT_AROMATIC_AAS = "F,Y,W,H"
DEFAULT_CARBOXYLATE_AAS = "D,E"

AA3_TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

RING_ATOMS = {
    "F": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "Y": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "W": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "H": ("CG", "ND1", "CD2", "CE1", "NE2"),
}

CARBOXYLATE_ATOM_SETS = {
    # Asp typically uses CG as the terminal carboxylate carbon in PDB atom naming.
    # Accept CD as well for robustness across nonstandard tables.
    "D": (("OD1", "OD2", "CD"), ("OD1", "OD2", "CG")),
    "E": (("OE1", "OE2", "CD"),),
}


def parse_residue_token(tok: str) -> tuple[str, str]:
    tok = (tok or "").strip()
    if len(tok) < 2:
        raise ValueError(f"Invalid residue token: {tok!r}")
    aa = tok[0]
    rs = tok[1:]
    if not rs.isdigit():
        raise ValueError(f"Invalid residue token: {tok!r}")
    return aa, rs


def parse_residue_list(cell: str) -> list[tuple[str, str]]:
    vals = [x.strip() for x in (cell or "").split(",") if x.strip()]
    return [parse_residue_token(v) for v in vals]


def pairdist_rmsd(points_a: np.ndarray, points_b: np.ndarray) -> float:
    n = int(points_a.shape[0])
    if n < 2:
        return 0.0
    sse = 0.0
    m = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            da = float(np.linalg.norm(points_a[i] - points_a[j]))
            db = float(np.linalg.norm(points_b[i] - points_b[j]))
            d = da - db
            sse += d * d
            m += 1
    return math.sqrt(sse / m)


def kabsch_fit(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Fit mobile -> target
    mob_c = mobile - mobile.mean(axis=0)
    tgt_c = target - target.mean(axis=0)
    h = mob_c.T @ tgt_c
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target.mean(axis=0) - (mobile.mean(axis=0) @ r)
    return r, t


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    d = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(d))


def fold_axis_symmetric_angle(theta_deg: float) -> float:
    # Treat direction-reversal on symmetric sidechain axes as equivalent.
    return min(theta_deg, 180.0 - theta_deg)


def normalize(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def load_pdb_atom_table(entry_dir: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    meta_path = entry_dir / "meta.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    pdb_path = Path(meta.get("pdb", ""))
    if not pdb_path.exists():
        return {}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(entry_dir.name, str(pdb_path))
    model = next(structure.get_models())
    chain_id = (meta.get("chain") or "").strip()
    if chain_id and chain_id in model:
        chain = model[chain_id]
    else:
        chain = next(model.get_chains())

    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for res in chain:
        # Standard amino acid residues only.
        if res.id[0] != " ":
            continue
        aa = AA3_TO1.get(res.get_resname().upper())
        if not aa:
            continue
        rs = str(res.id[1])
        amap: dict[str, np.ndarray] = {}
        for atom in res:
            amap[atom.get_name().upper()] = np.array(atom.get_coord(), dtype=float)
        out[(aa, rs)] = amap
    return out


def aromatic_vector_from_atoms(
    atom_map: dict[str, np.ndarray] | None, aa: str, mode: str
) -> np.ndarray | None:
    if atom_map is None:
        return None
    names = RING_ATOMS.get(aa)
    if not names:
        return None
    pts = [atom_map[n] for n in names if n in atom_map]
    if len(pts) < 3:
        return None
    arr = np.asarray(pts, dtype=float)
    ctr = arr.mean(axis=0)
    cen = arr - ctr
    _u, _s, vh = np.linalg.svd(cen, full_matrices=False)
    if mode == "ring_normal":
        vec = vh[-1]
    elif mode == "ring_axis":
        vec = vh[0]
    else:
        return None
    return normalize(vec)


def carboxylate_plane_normal_from_atoms(
    atom_map: dict[str, np.ndarray] | None, aa: str
) -> np.ndarray | None:
    if atom_map is None:
        return None
    atom_sets = CARBOXYLATE_ATOM_SETS.get(aa)
    if not atom_sets:
        return None
    for names in atom_sets:
        if not all(n in atom_map for n in names):
            continue
        p1, p2, p3 = (atom_map[n] for n in names)
        v = np.cross(p2 - p1, p3 - p1)
        vn = normalize(v)
        if vn is not None:
            return vn
    return None


def load_residue_table(path: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            key = (row["aa"], row["resseq"])
            out[key] = {
                "ca": np.array(
                    [float(row["ca_x"]), float(row["ca_y"]), float(row["ca_z"])],
                    dtype=float,
                ),
                "sc": np.array(
                    [float(row["sc_x"]), float(row["sc_y"]), float(row["sc_z"])],
                    dtype=float,
                ),
            }
    return out


def extract_mapping(row: dict[str, str], mapping_cols: list[str]) -> list[tuple[str, str]]:
    mapped: list[tuple[str, str]] = []
    for col in mapping_cols:
        if col not in row:
            raise KeyError(f"Missing mapping column: {col}")
        cell = (row.get(col) or "").strip()
        if not cell:
            raise ValueError(f"Empty mapping cell for column: {col}")
        if "," in cell:
            mapped.extend(parse_residue_list(cell))
        else:
            mapped.append(parse_residue_token(cell))
    return mapped


def minmax_norm(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orientation check for mapped pocket hits (Kabsch + CA->SC vectors)."
    )
    ap.add_argument("--in-tsv", required=True, help="Input mapped/ranked TSV.")
    ap.add_argument("--out", required=True, help="Output TSV with orientation metrics.")
    ap.add_argument("--ecoli-batch", default="src/ecoli_batch")
    ap.add_argument("--template-entry", default="DPO3B_ECOLI")
    ap.add_argument("--template-res", default=DEFAULT_TEMPLATE_RES)
    ap.add_argument("--mapping-cols", default=DEFAULT_MAPPING_COLS)
    ap.add_argument(
        "--axis-symmetry-aas",
        default=DEFAULT_AXIS_SYMM_AAS,
        help="Comma-separated AA codes that use axis-symmetry correction: theta=min(theta,180-theta).",
    )
    ap.add_argument(
        "--aromatic-vector-mode",
        choices=("ca_sc", "ring_normal", "ring_axis"),
        default="ca_sc",
        help="Vector mode for aromatic residues. Non-aromatics always use CA->SC.",
    )
    ap.add_argument(
        "--aromatic-aas",
        default=DEFAULT_AROMATIC_AAS,
        help="Comma-separated aromatic AAs eligible for aromatic-vector mode.",
    )
    ap.add_argument(
        "--carboxylate-vector-mode",
        choices=("ca_sc", "plane_normal"),
        default="ca_sc",
        help="Vector mode for carboxylate residues (D/E).",
    )
    ap.add_argument(
        "--carboxylate-aas",
        default=DEFAULT_CARBOXYLATE_AAS,
        help="Comma-separated residues eligible for carboxylate-vector mode.",
    )
    ap.add_argument(
        "--geometry-col",
        default="global7_vilmf",
        help="Geometry score column for optional blended rank; set empty to disable.",
    )
    ap.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="Blend weight for geometry normalized score (0..1).",
    )
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    if not (0.0 <= args.blend_alpha <= 1.0):
        raise SystemExit("--blend-alpha must be in [0,1]")

    in_path = Path(args.in_tsv)
    out_path = Path(args.out)
    batch = Path(args.ecoli_batch)
    mapping_cols = [c.strip() for c in args.mapping_cols.split(",") if c.strip()]
    axis_sym_aas = {x.strip().upper() for x in args.axis_symmetry_aas.split(",") if x.strip()}
    aromatic_aas = {x.strip().upper() for x in args.aromatic_aas.split(",") if x.strip()}
    carboxylate_aas = {
        x.strip().upper() for x in args.carboxylate_aas.split(",") if x.strip()
    }
    template_tokens = parse_residue_list(args.template_res)
    if len(template_tokens) < 2:
        raise SystemExit("Need at least 2 template residues.")

    use_atom_vectors = (
        args.aromatic_vector_mode != "ca_sc"
        or args.carboxylate_vector_mode == "plane_normal"
    )

    template_table = load_residue_table(batch / args.template_entry / "residues.tsv")
    template_sc = []
    template_vec = []
    template_vec_src = []
    template_atoms = (
        load_pdb_atom_table(batch / args.template_entry) if use_atom_vectors else {}
    )
    for aa, rs in template_tokens:
        key = (aa, rs)
        if key not in template_table:
            raise SystemExit(f"Template residue not found: {aa}{rs}")
        rec = template_table[key]
        template_sc.append(rec["sc"])
        vv = None
        src = "ca_sc"
        if args.aromatic_vector_mode != "ca_sc" and aa in aromatic_aas:
            vv = aromatic_vector_from_atoms(
                template_atoms.get(key), aa, args.aromatic_vector_mode
            )
            if vv is not None:
                src = args.aromatic_vector_mode
        if (
            vv is None
            and args.carboxylate_vector_mode == "plane_normal"
            and aa in carboxylate_aas
        ):
            vv = carboxylate_plane_normal_from_atoms(template_atoms.get(key), aa)
            if vv is not None:
                src = "carboxylate_plane_normal"
        if vv is None:
            vv = normalize(rec["sc"] - rec["ca"])
            src = "ca_sc"
        if vv is None:
            # For residues like Gly (SC==CA), orientation is undefined; skip at angle stage.
            src = "skip_zero_vector"
        template_vec.append(vv)
        template_vec_src.append(src)
    template_sc_arr = np.asarray(template_sc, dtype=float)

    rows: list[dict[str, str]] = []
    with in_path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(dict(row))

    cache: dict[str, dict[tuple[str, str], dict[str, np.ndarray]]] = {}
    cache_atoms: dict[str, dict[tuple[str, str], dict[str, np.ndarray]]] = {}
    ok_rows: list[dict[str, str]] = []
    n_template = len(template_tokens)
    for idx, row in enumerate(rows, start=1):
        row["status"] = "ok"
        try:
            entry = (row.get("entry") or "").strip()
            if not entry:
                raise ValueError("Missing entry column value")
            mapped = extract_mapping(row, mapping_cols)
            if len(mapped) != n_template:
                raise ValueError(
                    f"Mapped residue count {len(mapped)} != template count {n_template}"
                )

            if entry not in cache:
                cache[entry] = load_residue_table(batch / entry / "residues.tsv")
            if use_atom_vectors and entry not in cache_atoms:
                cache_atoms[entry] = load_pdb_atom_table(batch / entry)
            etab = cache[entry]
            eatoms = cache_atoms.get(entry, {})

            target_sc = []
            target_vec = []
            target_vec_src = []
            for aa, rs in mapped:
                key = (aa, rs)
                if key not in etab:
                    raise ValueError(f"Residue not found in {entry}: {aa}{rs}")
                rec = etab[key]
                target_sc.append(rec["sc"])
                vv = None
                src = "ca_sc"
                if args.aromatic_vector_mode != "ca_sc" and aa in aromatic_aas:
                    vv = aromatic_vector_from_atoms(
                        eatoms.get(key), aa, args.aromatic_vector_mode
                    )
                    if vv is not None:
                        src = args.aromatic_vector_mode
                if (
                    vv is None
                    and args.carboxylate_vector_mode == "plane_normal"
                    and aa in carboxylate_aas
                ):
                    vv = carboxylate_plane_normal_from_atoms(eatoms.get(key), aa)
                    if vv is not None:
                        src = "carboxylate_plane_normal"
                if vv is None:
                    vv = normalize(rec["sc"] - rec["ca"])
                    src = "ca_sc"
                if vv is None:
                    # Undefined orientation vector (e.g., Gly); keep row and skip this angle.
                    src = "skip_zero_vector"
                target_vec.append(vv)
                target_vec_src.append(src)

            target_sc_arr = np.asarray(target_sc, dtype=float)

            rmat, tvec = kabsch_fit(target_sc_arr, template_sc_arr)
            det_r = float(np.linalg.det(rmat))
            aligned = target_sc_arr @ rmat + tvec
            coord_rmsd = float(
                np.sqrt(np.mean(np.sum((aligned - template_sc_arr) ** 2, axis=1)))
            )
            pair_rmsd = pairdist_rmsd(template_sc_arr, target_sc_arr)

            target_vec_rot = [(qv @ rmat) if qv is not None else None for qv in target_vec]
            angs_raw = []
            angs = []
            angs_raw_pos: list[float | None] = [None] * n_template
            angs_pos: list[float | None] = [None] * n_template
            for i, ((taa, _), (qaa, _), tv, qv, tsrc, qsrc) in enumerate(
                zip(
                template_tokens,
                mapped,
                template_vec,
                target_vec_rot,
                template_vec_src,
                target_vec_src,
                ),
                start=1,
            ):
                if tv is None or qv is None:
                    continue
                a = angle_deg(tv, qv)
                angs_raw.append(a)
                fold_for_sym = (taa in axis_sym_aas) or (qaa in axis_sym_aas)
                if tsrc in (
                    "ring_normal",
                    "ring_axis",
                    "carboxylate_plane_normal",
                ) or qsrc in ("ring_normal", "ring_axis", "carboxylate_plane_normal"):
                    # Normal/axis vectors are sign-symmetric.
                    fold_for_sym = True
                if fold_for_sym:
                    a = fold_axis_symmetric_angle(a)
                angs.append(a)
                angs_pos[i - 1] = a
                angs_raw_pos[i - 1] = angs_raw[-1]
            if not angs:
                raise ValueError("No valid orientation vectors after skipping zero-length vectors")
            row["det_R"] = f"{det_r:.12f}"
            row["coord_rmsd_kabsch"] = f"{coord_rmsd:.6f}"
            row["pairdist_rmsd_recalc"] = f"{pair_rmsd:.6f}"
            row["orient_mean_deg"] = f"{float(np.mean(angs)):.6f}"
            row["orient_rms_deg"] = f"{float(np.sqrt(np.mean(np.square(angs)))):.6f}"
            row["orient_max_deg"] = f"{float(np.max(angs)):.6f}"
            row["orient_n"] = str(len(angs))
            row["orient_skipped_n"] = str(n_template - len(angs))
            for i in range(1, n_template + 1):
                a = angs_pos[i - 1]
                row[f"ang{i}_deg"] = f"{a:.4f}" if a is not None else ""
            for i in range(1, n_template + 1):
                a = angs_raw_pos[i - 1]
                row[f"ang{i}_raw_deg"] = f"{a:.4f}" if a is not None else ""
            for i, (ts, qs) in enumerate(zip(template_vec_src, target_vec_src), start=1):
                row[f"vecsrc{i}"] = f"{ts}|{qs}"

            ok_rows.append(row)
        except Exception as e:  # keep failures in output for auditability
            row["status"] = f"error:{e}"

        if args.progress_every > 0 and (idx % args.progress_every == 0):
            print(f"progress {idx}/{len(rows)} ok={len(ok_rows)}", flush=True)

    # Orientation rank
    ok_sorted_or = sorted(ok_rows, key=lambda x: float(x["orient_mean_deg"]))
    for i, row in enumerate(ok_sorted_or, start=1):
        row["rank_orient_mean"] = str(i)

    # Optional geometry rank + blend rank
    gcol = (args.geometry_col or "").strip()
    if gcol:
        geom_ok = [r for r in ok_rows if gcol in r and str(r[gcol]).strip()]
        geom_ok = [
            r for r in geom_ok if _is_float(r[gcol]) and _is_float(r["orient_mean_deg"])
        ]
        geom_sorted = sorted(geom_ok, key=lambda x: float(x[gcol]))
        for i, row in enumerate(geom_sorted, start=1):
            row["rank_geometry"] = str(i)

        geom_vals = [float(r[gcol]) for r in geom_ok]
        or_vals = [float(r["orient_mean_deg"]) for r in geom_ok]
        gnorm = minmax_norm(geom_vals)
        onorm = minmax_norm(or_vals)
        for row, gn, on in zip(geom_ok, gnorm, onorm):
            blend = args.blend_alpha * gn + (1.0 - args.blend_alpha) * on
            row["blend_score"] = f"{blend:.6f}"
        blend_sorted = sorted(geom_ok, key=lambda x: float(x["blend_score"]))
        for i, row in enumerate(blend_sorted, start=1):
            row["rank_blend"] = str(i)

    # Compose output fields: preserve input, then append new fields
    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = []
    for k in (
        "rank_geometry",
        "rank_orient_mean",
        "rank_blend",
        "status",
        "det_R",
        "pairdist_rmsd_recalc",
        "coord_rmsd_kabsch",
        "orient_mean_deg",
        "orient_rms_deg",
        "orient_max_deg",
        "orient_n",
        "orient_skipped_n",
        "blend_score",
    ):
        if any(k in r for r in rows) and k not in base_fields:
            extra_fields.append(k)
    for i in range(1, n_template + 1):
        k = f"ang{i}_deg"
        if any(k in r for r in rows) and k not in base_fields:
            extra_fields.append(k)
    for i in range(1, n_template + 1):
        k = f"ang{i}_raw_deg"
        if any(k in r for r in rows) and k not in base_fields:
            extra_fields.append(k)
    for i in range(1, n_template + 1):
        k = f"vecsrc{i}"
        if any(k in r for r in rows) and k not in base_fields:
            extra_fields.append(k)

    fields = base_fields + [k for k in extra_fields if k not in base_fields]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"wrote {out_path}")
    print(f"rows {len(rows)} ok {len(ok_rows)}")


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
