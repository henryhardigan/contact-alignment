import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import _db200k_cli
except ModuleNotFoundError:
    from examples import _db200k_cli

from contact_alignment import db200k_scan, design


def get_parser():
    parser = argparse.ArgumentParser(
        description="Design reciprocal DB200K-favored aligned sequences by beam-sampling per-position favorites."
    )
    _db200k_cli.add_profile_args(parser)
    parser.add_argument(
        "--top-choices-per-position",
        type=int,
        default=4,
        help="How many residue choices to keep from each query position profile.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1000,
        help="Maximum number of partial candidate sequences to keep during beam search.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=25,
        help="How many reciprocal-rescored candidates to print.",
    )
    parser.add_argument(
        "--rescue-mode",
        type=str,
        default="none",
        choices=db200k_scan.RESCUE_MODES,
        help="Whether to use rescue logic during reciprocal rescoring.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    profiles = _db200k_cli.build_profiles_from_args(args)
    candidates = design.beam_sample_favored_sequences(
        profiles,
        top_n=args.top_choices_per_position,
        beam_width=args.beam_width,
    )
    rows = design.reciprocal_rescore_candidates(
        args.query_seq,
        args.db_root,
        candidates,
        profile_strategy=args.profile_strategy,
        one_by_one_mode=args.one_by_one_mode,
        one_by_one_matrix_tsv=args.one_by_one_matrix_tsv,
        strong_threshold=args.strong_threshold,
        strong_weight_3x3=args.strong_weight_3x3,
        weak_weight_3x3=args.weak_weight_3x3,
        shrinkage_prior_3x3=args.shrinkage_prior_3x3,
        shrinkage_prior_5x5=args.shrinkage_prior_5x5,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        rescue_mode=args.rescue_mode,
        top_k=args.final_top_k,
    )

    print("Top reciprocal designs")
    for row in rows:
        print(
            f"{row.total_score:.4f}\tforward={row.forward_score:.4f}\t"
            f"reciprocal={row.reciprocal_score:.4f}\t{row.sequence}"
        )


if __name__ == "__main__":
    main()
