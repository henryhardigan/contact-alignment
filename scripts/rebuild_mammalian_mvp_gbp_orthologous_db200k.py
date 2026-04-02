#!/usr/bin/env python3
"""Rebuild mammalian MVP/GBP DB200K comparisons using orthologous window mapping.

This script maps the human MVP and GBP windows onto species ortholog candidates
by global protein alignment, then scores the species-specific windows with the
local DB200K scan utilities.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional

import requests
from Bio import BiopythonDeprecationWarning, pairwise2
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contact_alignment import db200k_scan
from scripts.build_query_ungapped_prefilter import load_blosum62, pair_blosum


DEFAULT_HUMAN_MVP_ACCESSION = "Q14764"
DEFAULT_HUMAN_GBP_ACCESSION = "P32455"
DEFAULT_HUMAN_MVP_WINDOW = "TSEAKGPDGMALPRPRDQAVFPQ"
DEFAULT_HUMAN_GBP_WINDOW = "TEKMENDRVQLLKEQERTLALKL"
DEFAULT_DB_ROOT = str(Path("~/Downloads/pisces-cache-scaac").expanduser())
DEFAULT_CACHE_DIR = str(Path("~/Downloads/pisces-cache-scaac/.db200k_cache").expanduser())
DEFAULT_MVP_META = str((ROOT / "tmp/uniprot_mammal_mvp.tsv").resolve())
DEFAULT_GBP_META = str((ROOT / "tmp/mammalian_gbp_windows_vs_nfo_by_taxa.tsv").resolve())
DEFAULT_OUT = str((ROOT / "tmp/mammalian_mvp_gbp_orthologous_db200k.tsv").resolve())


warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)


@dataclass
class SpeciesCandidate:
    species: str
    accession: str
    entry: str
    reviewed: str
    gene_names: str
    protein_name: str
    length: int
    lineage: str


@dataclass
class WindowMapping:
    sequence: str
    start_1based: int
    end_1based: int
    mapped_positions: int
    contiguous: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human-mvp-accession", default=DEFAULT_HUMAN_MVP_ACCESSION)
    p.add_argument("--human-gbp-accession", default=DEFAULT_HUMAN_GBP_ACCESSION)
    p.add_argument("--human-mvp-window", default=DEFAULT_HUMAN_MVP_WINDOW)
    p.add_argument("--human-gbp-window", default=DEFAULT_HUMAN_GBP_WINDOW)
    p.add_argument("--mvp-meta", default=DEFAULT_MVP_META)
    p.add_argument("--gbp-meta", default=DEFAULT_GBP_META)
    p.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--output", default=DEFAULT_OUT)
    p.add_argument(
        "--min-mapped-positions",
        type=int,
        default=23,
        help="Require at least this many mapped human-window positions in both species proteins.",
    )
    return p.parse_args()


def sci_name(organism: str) -> str:
    return organism.split(" (", 1)[0]


def fetch_uniprot_sequence(accession: str) -> str:
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{accession}.fasta", timeout=60)
    response.raise_for_status()
    lines = response.text.splitlines()
    return "".join(lines[1:]).strip().upper()


def read_tsv(path: Path) -> List[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_best_mvp_candidates(rows: Iterable[dict[str, str]], human_length: int) -> Dict[str, SpeciesCandidate]:
    best: Dict[str, tuple[tuple[int, int, int, int], SpeciesCandidate]] = {}
    for row in rows:
        protein_name = row["Protein names"]
        protein_name_lower = protein_name.lower()
        species = sci_name(row["Organism"])
        gene_tokens = set(row["Gene Names"].split())
        try:
            length = int(row["Length"])
        except ValueError:
            continue
        candidate = SpeciesCandidate(
            species=species,
            accession=row["Entry"],
            entry=row["Entry Name"],
            reviewed=row["Reviewed"],
            gene_names=row["Gene Names"],
            protein_name=protein_name,
            length=length,
            lineage=row["Taxonomic lineage"],
        )
        is_major_vault = int("major vault protein" in protein_name_lower)
        has_exact_mvp = int("MVP" in gene_tokens or "Mvp" in gene_tokens)
        is_reviewed = int(row["Reviewed"] == "reviewed")
        closeness = -abs(length - human_length)
        key = (is_major_vault, has_exact_mvp, is_reviewed, closeness)
        if species not in best or key > best[species][0]:
            best[species] = (key, candidate)
    return {species: candidate for species, (_, candidate) in best.items()}


def pick_gbp_candidates(rows: Iterable[dict[str, str]]) -> Dict[str, SpeciesCandidate]:
    out: Dict[str, SpeciesCandidate] = {}
    for row in rows:
        species = sci_name(row["organism"])
        try:
            length = len(row["window"])
        except Exception:
            length = 0
        out[species] = SpeciesCandidate(
            species=species,
            accession=row["accession"],
            entry=row["entry"],
            reviewed=row["reviewed"],
            gene_names=row["gene_names"],
            protein_name=row.get("protein_name", ""),
            length=length,
            lineage=row["lineage"],
        )
    return out


def map_human_window(
    human_seq: str,
    species_seq: str,
    human_window: str,
    *,
    matrix,
) -> Optional[WindowMapping]:
    start0 = human_seq.find(human_window)
    if start0 < 0:
        raise ValueError(f"Human window not found: {human_window}")
    start_1based = start0 + 1
    end_1based = start_1based + len(human_window) - 1

    alignment = pairwise2.align.globalds(
        human_seq,
        species_seq,
        matrix,
        -10,
        -0.5,
        one_alignment_only=True,
    )[0]
    human_aln, species_aln, _, _, _ = alignment

    human_pos = 0
    species_pos = 0
    mapping: Dict[int, Optional[int]] = {}
    for aa_h, aa_s in zip(human_aln, species_aln):
        if aa_h != "-":
            human_pos += 1
        if aa_s != "-":
            species_pos += 1
        if aa_h != "-":
            mapping[human_pos] = species_pos if aa_s != "-" else None

    mapped = [mapping.get(i) for i in range(start_1based, end_1based + 1)]
    non_gap = [m for m in mapped if m is not None]
    if not non_gap:
        return None

    residues = "".join(species_seq[m - 1] for m in mapped if m is not None)
    contiguous = len(non_gap) == len(human_window) and (max(non_gap) - min(non_gap) + 1 == len(human_window))
    return WindowMapping(
        sequence=residues,
        start_1based=min(non_gap),
        end_1based=max(non_gap),
        mapped_positions=len(non_gap),
        contiguous=contiguous,
    )


def blosum_total(query: str, target: str, blosum) -> int:
    return sum(pair_blosum(a, b, blosum) for a, b in zip(query, target))


def mammal_group(lineage: str) -> str:
    if "Monotremata (order)" in lineage or "Prototheria (clade)" in lineage:
        return "Monotreme"
    if "Metatheria (clade)" in lineage or "Marsupialia (infraclass)" in lineage:
        return "Marsupial"
    return "Eutheria"


def order_name(lineage: str) -> str:
    for token in (part.strip() for part in lineage.split(",")):
        if token.endswith("(order)"):
            return token[:-8].strip()
    return ""


def main() -> int:
    args = parse_args()
    blosum = load_blosum62("src/pocket_pipeline/blosum62.txt")
    matrix = substitution_matrices.load("BLOSUM62")

    human_mvp_seq = fetch_uniprot_sequence(args.human_mvp_accession)
    human_gbp_seq = fetch_uniprot_sequence(args.human_gbp_accession)
    human_mvp_len = len(human_mvp_seq)

    mvp_candidates = pick_best_mvp_candidates(read_tsv(Path(args.mvp_meta)), human_mvp_len)
    gbp_rows = read_tsv(Path(args.gbp_meta))
    gbp_candidates = pick_gbp_candidates(gbp_rows)

    idx3 = db200k_scan.load_or_build_sequence_index(args.db_root, "3x3", cache_dir=args.cache_dir)

    rows: List[dict[str, object]] = []
    shared_species = sorted(set(mvp_candidates) & set(gbp_candidates))
    for species in shared_species:
        mvp = mvp_candidates[species]
        gbp = gbp_candidates[species]
        mvp_seq = fetch_uniprot_sequence(mvp.accession)
        gbp_seq = fetch_uniprot_sequence(gbp.accession)

        mvp_map = map_human_window(human_mvp_seq, mvp_seq, args.human_mvp_window, matrix=matrix)
        gbp_map = map_human_window(human_gbp_seq, gbp_seq, args.human_gbp_window, matrix=matrix)
        if mvp_map is None or gbp_map is None:
            continue
        if mvp_map.mapped_positions < args.min_mapped_positions or gbp_map.mapped_positions < args.min_mapped_positions:
            continue
        if len(mvp_map.sequence) != len(args.human_mvp_window) or len(gbp_map.sequence) != len(args.human_gbp_window):
            continue

        profiles = db200k_scan.build_query_profiles(
            mvp_map.sequence,
            args.db_root,
            cache_dir=args.cache_dir,
            profile_strategy="shrinkage_5x5_3x3_1x1",
            one_by_one_mode="reciprocal_mean",
        )
        score_fwd, _ = db200k_scan.score_window(gbp_map.sequence, profiles)
        score_fwd_norm, _ = db200k_scan.score_window(
            gbp_map.sequence,
            profiles,
            score_mode="confidence_adjusted",
        )
        profiles_rev = db200k_scan.build_query_profiles(
            gbp_map.sequence,
            args.db_root,
            cache_dir=args.cache_dir,
            profile_strategy="shrinkage_5x5_3x3_1x1",
            one_by_one_mode="reciprocal_mean",
        )
        score_rev, _ = db200k_scan.score_window(mvp_map.sequence, profiles_rev)
        score_rev_norm, _ = db200k_scan.score_window(
            mvp_map.sequence,
            profiles_rev,
            score_mode="confidence_adjusted",
        )
        recip3_mean, _, used_positions = db200k_scan.score_window_reciprocal_3x3(
            mvp_map.sequence,
            gbp_map.sequence,
            idx3,
            metric="mean",
        )

        rows.append(
            {
                "species": species,
                "mammal_group": mammal_group(gbp.lineage),
                "order": order_name(gbp.lineage),
                "mvp_accession": mvp.accession,
                "mvp_entry": mvp.entry,
                "mvp_reviewed": mvp.reviewed,
                "mvp_length": mvp.length,
                "mvp_window_start_1based": mvp_map.start_1based,
                "mvp_window_end_1based": mvp_map.end_1based,
                "mvp_window_mapped_positions": mvp_map.mapped_positions,
                "mvp_window_contiguous": "yes" if mvp_map.contiguous else "no",
                "mvp_window": mvp_map.sequence,
                "gbp_accession": gbp.accession,
                "gbp_entry": gbp.entry,
                "gbp_reviewed": gbp.reviewed,
                "gbp_window_start_1based": gbp_map.start_1based,
                "gbp_window_end_1based": gbp_map.end_1based,
                "gbp_window_mapped_positions": gbp_map.mapped_positions,
                "gbp_window_contiguous": "yes" if gbp_map.contiguous else "no",
                "gbp_window_human_blosum": blosum_total(args.human_gbp_window, gbp_map.sequence, blosum),
                "gbp_window": gbp_map.sequence,
                "db200k_shrinkage_recip1x1mean_mvp_to_gbp": score_fwd,
                "db200k_shrinkage_recip1x1mean_mvp_to_gbp_norm": score_fwd_norm,
                "db200k_shrinkage_recip1x1mean_gbp_to_mvp": score_rev,
                "db200k_shrinkage_recip1x1mean_gbp_to_mvp_norm": score_rev_norm,
                "db200k_recip3_mean": recip3_mean,
                "db200k_recip3_used_positions": used_positions,
            }
        )

    if not rows:
        raise SystemExit("No species satisfied the orthologous window mapping criteria.")

    rows.sort(key=lambda r: (r["mammal_group"], r["order"], r["species"]))
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"out={out_path}")
    print(f"species_with_both={len(rows)}")
    by_group: Dict[str, List[float]] = {}
    for row in rows:
        by_group.setdefault(str(row["mammal_group"]), []).append(float(row["db200k_recip3_mean"]))
    for group, vals in by_group.items():
        print(
            f"{group}: n={len(vals)} median_recip3_mean={median(vals):.6f} "
            f"min={min(vals):.6f} max={max(vals):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
