import argparse

from contact_alignment import db200k_scan


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-seq", type=str, required=True, help="Query loop/peptide sequence.")
    parser.add_argument("--db-root", type=str, required=True, help="Path to the extracted DB200K root.")
    parser.add_argument(
        "--profile-strategy",
        type=str,
        default=db200k_scan.PROFILE_STRATEGIES[0],
        choices=db200k_scan.PROFILE_STRATEGIES,
        help="How to build per-position DB200K profiles.",
    )
    parser.add_argument("--strong-threshold", type=int, default=100)
    parser.add_argument("--strong-weight-3x3", type=float, default=0.8)
    parser.add_argument("--weak-weight-3x3", type=float, default=0.5)
    parser.add_argument("--shrinkage-prior-3x3", type=float, default=25.0)
    parser.add_argument("--shrinkage-prior-5x5", type=float, default=10.0)
    parser.add_argument(
        "--one-by-one-mode",
        type=str,
        default=db200k_scan.ONE_BY_ONE_MODES[0],
        choices=db200k_scan.ONE_BY_ONE_MODES,
        help="How to build 1x1 rows: directed cache rows, or reciprocal sum/mean rows.",
    )
    parser.add_argument(
        "--one-by-one-matrix-tsv",
        type=str,
        default=None,
        help="Optional reciprocal 1x1 matrix TSV to load instead of deriving 1x1 rows from the cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional directory for cached DB200K sequence indices. Defaults to <db-root>/.db200k_cache.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild cached DB200K sequence indices from the raw DB200K files.",
    )


def add_alignment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prefilter-score-threshold",
        type=float,
        default=None,
        help="Optional fast rigid-window cutoff applied before the main scan. Windows with prefilter score > threshold are skipped.",
    )
    parser.add_argument(
        "--prefilter-score-mode",
        type=str,
        default="raw",
        choices=db200k_scan.SCORE_MODES,
        help="Scoring mode for the fast rigid prefilter stage.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default=db200k_scan.SCORE_MODES[0],
        choices=db200k_scan.SCORE_MODES,
        help="How to normalize per-position scores before summation.",
    )
    parser.add_argument(
        "--uncertainty-floor",
        type=float,
        default=1.0,
        help="Minimum denominator used by confidence-adjusted scoring.",
    )
    parser.add_argument(
        "--alignment-mode",
        type=str,
        default=db200k_scan.ALIGNMENT_MODES[0],
        choices=db200k_scan.ALIGNMENT_MODES,
        help="How to align query profiles to candidate windows.",
    )
    parser.add_argument(
        "--max-target-gaps",
        type=int,
        default=1,
        help="Maximum number of skipped target residues for non-rigid alignment modes.",
    )
    parser.add_argument(
        "--min-aligned-positions",
        type=int,
        default=None,
        help="Minimum number of aligned query positions for non-rigid alignment modes.",
    )
    parser.add_argument(
        "--target-gap-penalty",
        type=float,
        default=0.0,
        help="Penalty added per skipped target residue for non-rigid alignment modes.",
    )
    parser.add_argument(
        "--target-flank",
        type=int,
        default=1,
        help="Extra target residues to include on each side for non-rigid alignment modes.",
    )
    parser.add_argument(
        "--peripheral-flank-weight",
        type=float,
        default=0.5,
        help="Interpolation weight for favorable +/-1 flank contacts at non-rigid alignment boundaries.",
    )
    parser.add_argument(
        "--fast-scan",
        action="store_true",
        help="Use score-only rigid scanning and skip per-window breakdown construction.",
    )
    parser.add_argument(
        "--numba",
        action="store_true",
        help="Use the optional Numba-accelerated rigid fast path when available.",
    )


def add_threshold_arg(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--score-threshold",
        type=float,
        required=required,
        default=None if not required else argparse.SUPPRESS,
        help="Optional score cutoff; return only windows with score <= threshold.",
    )


def build_profiles_from_args(args: argparse.Namespace):
    return db200k_scan.build_query_profiles(
        args.query_seq,
        args.db_root,
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
    )


def scan_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "score_threshold": getattr(args, "score_threshold", None),
        "prefilter_score_threshold": args.prefilter_score_threshold,
        "prefilter_score_mode": args.prefilter_score_mode,
        "score_mode": args.score_mode,
        "uncertainty_floor": args.uncertainty_floor,
        "alignment_mode": args.alignment_mode,
        "max_target_gaps": args.max_target_gaps,
        "min_aligned_positions": args.min_aligned_positions,
        "target_gap_penalty": args.target_gap_penalty,
        "target_flank": args.target_flank,
        "peripheral_flank_weight": args.peripheral_flank_weight,
        "fast_scan": args.fast_scan,
        "use_numba": args.numba,
    }
