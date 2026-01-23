"""Command-line interface for mj-align.

This module provides the CLI entry point for the mj-align package.
For full functionality, the original mj_score.py can still be used directly.

This stub imports the key functions from the modularized package and
provides a simplified interface. The full CLI with all options is
available in the original mj_score.py file.

Usage:
    mj-score --mj matrix.csv --seq1 ACDEF --seq2 FGHIK

For full options, see:
    python mj_score.py --help
"""

import argparse
import sys
from typing import Optional

from .clustering import clustal_similarity
from .fasta_io import read_fasta_all, read_fasta_entry
from .formatting import fmt_float, fmt_pct, fmt_prob
from .scoring import anchors_by_threshold, load_mj_csv, score_aligned, top_contributors
from .statistics import null_distribution, quantile


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the mj-score CLI.

    This is a simplified CLI. For full functionality with all options,
    use the original mj_score.py script directly.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success).
    """
    p = argparse.ArgumentParser(
        description="Score two aligned sequences using an MJ matrix.",
        epilog="For full options, use: python mj_score.py --help",
    )
    p.add_argument(
        "--mj",
        default="mj_matrix.csv",
        help="Path to MJ matrix CSV (default: mj_matrix.csv)",
    )
    p.add_argument("--name1", default="seq1", help="Name for sequence 1")
    p.add_argument("--name2", default="seq2", help="Name for sequence 2")
    p.add_argument("--seq1", help="Aligned sequence 1 (may include '-')")
    p.add_argument("--seq2", help="Aligned sequence 2 (may include '-')")
    p.add_argument("--fasta1", help="FASTA file for sequence 1")
    p.add_argument("--fasta2", help="FASTA file for sequence 2")
    p.add_argument("--fasta1-entry", help="Entry filter for fasta1")
    p.add_argument("--fasta2-entry", help="Entry filter for fasta2")
    p.add_argument(
        "--thr",
        type=float,
        default=-25.0,
        help="Anchor threshold (MJ <= thr). Default: -25",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top N contributing positions. Default: 10",
    )
    p.add_argument(
        "--null",
        type=int,
        default=0,
        help="Null samples for shuffled distribution. Default: 0",
    )
    p.add_argument(
        "--clustal",
        action="store_true",
        help="Print Clustal-style similarity",
    )
    p.add_argument(
        "--unknown",
        choices=["error", "skip", "zero"],
        default="error",
        help="How to handle unknown residues. Default: error",
    )

    args = p.parse_args(argv)

    # Load sequences
    name1 = args.name1
    name2 = args.name2
    seq1 = args.seq1
    seq2 = args.seq2

    if args.fasta1:
        if args.fasta1_entry:
            name1, seq1 = read_fasta_entry(args.fasta1, args.fasta1_entry)
        else:
            name1, seq1 = next(read_fasta_all(args.fasta1))

    if args.fasta2:
        if args.fasta2_entry:
            name2, seq2 = read_fasta_entry(args.fasta2, args.fasta2_entry)
        else:
            name2, seq2 = next(read_fasta_all(args.fasta2))

    if not seq1 or not seq2:
        print("Error: Both sequences must be provided.", file=sys.stderr)
        return 1

    # Ensure same length for aligned scoring
    if len(seq1) != len(seq2):
        print(
            f"Error: Sequences must be same length (got {len(seq1)} vs {len(seq2)}).",
            file=sys.stderr,
        )
        return 1

    # Load MJ matrix
    try:
        mj = load_mj_csv(args.mj)
    except FileNotFoundError:
        print(f"Error: MJ matrix file not found: {args.mj}", file=sys.stderr)
        return 1

    # Score alignment
    total, per_pos = score_aligned(
        seq1, seq2, mj, unknown_policy=args.unknown
    )

    # Find anchors
    anchors = anchors_by_threshold(per_pos, args.thr)

    # Output
    print("=== MJ Score Results ===\n")
    print(f"Sequence 1: {name1}")
    print(f"Sequence 2: {name2}")
    print(f"Alignment length: {len(seq1)}")
    print()
    print(f"Total MJ score: {fmt_float(total, 2)}")
    print(f"Anchors (MJ <= {args.thr}): {len(anchors)} positions")
    print(f"Anchor positions: {anchors[:20]}{'...' if len(anchors) > 20 else ''}")
    print()

    # Top contributors
    if args.top > 0:
        top = top_contributors(seq1, seq2, per_pos, args.top)
        print(f"Top {len(top)} contributing positions:")
        for score, pos, aa1, aa2 in top:
            print(f"  Position {pos}: {aa1}-{aa2} = {fmt_float(score, 2)}")
        print()

    # Clustal similarity
    if args.clustal:
        symbols, cscore, cnorm, nelig = clustal_similarity(seq1, seq2)
        print("Clustal similarity:")
        print(f"  Score: {fmt_float(cscore, 2)} / {nelig} = {fmt_pct(cnorm, 4)}")
        print(f"  {seq1}")
        print(f"  {symbols}")
        print(f"  {seq2}")
        print()

    # Null distribution
    if args.null > 0:
        print(f"Null distribution ({args.null} shuffles)...")
        null_scores = null_distribution(
            seq1, seq2, mj, args.null, unknown_policy=args.unknown
        )
        null_scores.sort()
        p_val = sum(1 for s in null_scores if s <= total) / len(null_scores)
        print(f"  Mean: {fmt_float(sum(null_scores)/len(null_scores), 2)}")
        print(f"  5th percentile: {fmt_float(quantile(null_scores, 0.05), 2)}")
        print(f"  Observed: {fmt_float(total, 2)}")
        print(f"  P-value: {fmt_prob(p_val)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
