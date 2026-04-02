#!/usr/bin/env python3
"""Compute reciprocal DB200K 1x1 scores for all amino-acid pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contact_alignment import db200k_scan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-root",
        required=True,
        help="Path to the extracted DB200K root.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory for cached DB200K sequence indices. Defaults to <db-root>/.db200k_cache.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild cached DB200K sequence indices from the raw DB200K files.",
    )
    parser.add_argument(
        "--metric",
        choices=("sum", "mean"),
        default="sum",
        help="How to combine reciprocal directions: sum = q->t + t->q, mean = average of both.",
    )
    parser.add_argument(
        "--format",
        choices=("matrix", "pairs"),
        default="matrix",
        help="Output a square matrix or a long table of all directed pair combinations.",
    )
    parser.add_argument(
        "--order",
        choices=("standard", "db200k"),
        default="standard",
        help="Residue output order. standard = ACDEFGHIKLMNPQRSTVWY, db200k = native DB200K cache order.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places in the output.",
    )
    return parser.parse_args()


def load_1x1_index(args: argparse.Namespace) -> dict[str, db200k_scan.SequenceIndexEntry]:
    return db200k_scan.load_or_build_sequence_index(
        args.db_root,
        "1x1",
        cache_dir=args.cache_dir,
        rebuild=args.rebuild_cache,
    )


def reciprocal_score(
    source: str,
    target: str,
    index_1x1: dict[str, db200k_scan.SequenceIndexEntry],
) -> tuple[float, float, float, float]:
    alphabet = db200k_scan.CENTER_ALPHABET
    target_idx = alphabet.index(target)
    source_idx = alphabet.index(source)
    source_row = index_1x1[source].mean
    target_row = index_1x1[target].mean
    forward = float(source_row[target_idx])
    reverse = float(target_row[source_idx])
    total = forward + reverse
    mean = total / 2.0
    return forward, reverse, total, mean


def get_output_alphabet(order: str) -> str:
    if order == "db200k":
        return db200k_scan.CENTER_ALPHABET
    return "ACDEFGHIKLMNPQRSTVWY"


def print_matrix(
    index_1x1: dict[str, db200k_scan.SequenceIndexEntry],
    *,
    order: str,
    metric: str,
    precision: int,
) -> None:
    alphabet = get_output_alphabet(order)
    print("\t" + "\t".join(alphabet))
    for source in alphabet:
        cells = [source]
        for target in alphabet:
            _, _, total, mean = reciprocal_score(source, target, index_1x1)
            value = total if metric == "sum" else mean
            cells.append(f"{value:.{precision}f}")
        print("\t".join(cells))


def print_pairs(index_1x1: dict[str, db200k_scan.SequenceIndexEntry], *, order: str, precision: int) -> None:
    alphabet = get_output_alphabet(order)
    print("source\ttarget\tforward\treverse\treciprocal_sum\treciprocal_mean")
    for source in alphabet:
        for target in alphabet:
            forward, reverse, total, mean = reciprocal_score(source, target, index_1x1)
            print(
                f"{source}\t{target}\t"
                f"{forward:.{precision}f}\t{reverse:.{precision}f}\t"
                f"{total:.{precision}f}\t{mean:.{precision}f}"
            )


def main() -> None:
    args = parse_args()
    index_1x1 = load_1x1_index(args)
    if args.format == "matrix":
        print_matrix(
            index_1x1,
            order=args.order,
            metric=args.metric,
            precision=args.precision,
        )
        return
    print_pairs(index_1x1, order=args.order, precision=args.precision)


if __name__ == "__main__":
    main()
