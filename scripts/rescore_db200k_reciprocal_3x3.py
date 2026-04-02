#!/usr/bin/env python3
"""Rescore top DB200K scan hits with dynamic reciprocal 3x3 terms."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import _db200k_cli
except ModuleNotFoundError:
    from examples import _db200k_cli

from contact_alignment import db200k_scan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _db200k_cli.add_profile_args(parser)
    _db200k_cli.add_alignment_args(parser)
    parser.add_argument("--fasta", required=True, help="FASTA file to scan.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top first-layer hits to keep before reciprocal 3x3 rescoring.",
    )
    parser.add_argument(
        "--reciprocal-3x3-metric",
        choices=("sum", "mean"),
        default="sum",
        help="How to combine forward and reverse 3x3 terms at each aligned interior position.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional TSV path for the rescored hit table.",
    )
    return parser.parse_args()


def parse_header_fields(header: str) -> tuple[str, str]:
    fields = header.split("|")
    if len(fields) >= 3:
        return fields[1], fields[2].split()[0]
    return "", ""


def main() -> None:
    args = parse_args()
    profiles = _db200k_cli.build_profiles_from_args(args)
    first_layer_hits = db200k_scan.scan_fasta(
        args.fasta,
        profiles,
        top_k=args.top_k,
        **_db200k_cli.scan_kwargs_from_args(args),
    )
    index_3x3 = db200k_scan.load_or_build_sequence_index(
        args.db_root,
        "3x3",
        cache_dir=args.cache_dir,
        rebuild=args.rebuild_cache,
    )

    rescored_rows = []
    for first_rank, hit in enumerate(first_layer_hits, start=1):
        reciprocal_total, reciprocal_breakdown, used_positions = db200k_scan.score_window_reciprocal_3x3(
            args.query_seq,
            str(hit["window"]),
            index_3x3,
            metric=args.reciprocal_3x3_metric,
        )
        accession, entry = parse_header_fields(str(hit["header"]))
        reciprocal_mean = reciprocal_total / used_positions if used_positions else float("nan")
        top_terms = sorted(reciprocal_breakdown, key=lambda item: item[5])[:6]
        top_terms_str = ",".join(
            f"{pos}:{query_triplet}>{target_triplet}:{combined:.4f}"
            for pos, query_triplet, target_triplet, _, _, combined in top_terms
        )
        rescored_rows.append(
            {
                "first_layer_rank": first_rank,
                "first_layer_score": float(hit["score"]),
                "accession": accession,
                "entry": entry,
                "header": str(hit["header"]),
                "start": int(hit["start"]),
                "end": int(hit["end"]),
                "window": str(hit["window"]),
                "reciprocal_3x3_total": reciprocal_total,
                "reciprocal_3x3_mean": reciprocal_mean,
                "reciprocal_3x3_positions": used_positions,
                "top_reciprocal_terms": top_terms_str,
            }
        )

    rescored_rows.sort(key=lambda row: (row["reciprocal_3x3_total"], row["first_layer_rank"]))
    for reciprocal_rank, row in enumerate(rescored_rows, start=1):
        row["reciprocal_3x3_rank"] = reciprocal_rank

    fieldnames = [
        "reciprocal_3x3_rank",
        "first_layer_rank",
        "first_layer_score",
        "reciprocal_3x3_total",
        "reciprocal_3x3_mean",
        "reciprocal_3x3_positions",
        "accession",
        "entry",
        "start",
        "end",
        "window",
        "top_reciprocal_terms",
        "header",
    ]
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rescored_rows)

    print("reciprocal_3x3_rank\tfirst_layer_rank\tfirst_layer_score\treciprocal_3x3_total\taccession\tentry\tstart\tend\twindow")
    for row in rescored_rows:
        print(
            f"{row['reciprocal_3x3_rank']}\t{row['first_layer_rank']}\t"
            f"{row['first_layer_score']:.4f}\t{row['reciprocal_3x3_total']:.4f}\t"
            f"{row['accession']}\t{row['entry']}\t{row['start']}\t{row['end']}\t{row['window']}"
        )


if __name__ == "__main__":
    main()
