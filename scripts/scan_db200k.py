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

from contact_alignment import db200k_scan


def get_parser():
    parser = argparse.ArgumentParser(description="Scan sequences with a DB200K-derived query profile.")
    _db200k_cli.add_profile_args(parser)
    parser.add_argument("--fasta", type=str, default=None, help="Optional FASTA file to scan.")
    parser.add_argument("--top-k", type=int, default=25, help="Number of top FASTA hits to print.")
    _db200k_cli.add_threshold_arg(parser, required=False)
    _db200k_cli.add_alignment_args(parser)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    profiles = _db200k_cli.build_profiles_from_args(args)

    print("Query profile")
    for profile in profiles:
        ranked = sorted(
            (
                (res, float(profile.energies[i]))
                for i, res in enumerate(db200k_scan.CENTER_ALPHABET)
            ),
            key=lambda kv: kv[1],
        )[:6]
        ranked_str = ", ".join(f"{res}:{energy:.4f}" for res, energy in ranked)
        print(
            f"{profile.position}\t{profile.query_context}\t{profile.center_residue}\t"
            f"{profile.support_mode}\t5x5={profile.count_5x5}\t3x3={profile.count_3x3}\t"
            f"1x1={profile.count_1x1}\tg5={profile.geometry_buckets_5x5}\t"
            f"g3={profile.geometry_buckets_3x3}\tg1={profile.geometry_buckets_1x1}\t"
            f"w5={profile.weight_5x5:.2f}\tw3={profile.weight_3x3:.2f}\t"
            f"eff={profile.effective_support:.1f}\t{ranked_str}"
        )

    if args.fasta is None:
        return

    hits = db200k_scan.scan_fasta(
        args.fasta,
        profiles,
        top_k=None if args.score_threshold is not None else args.top_k,
        **_db200k_cli.scan_kwargs_from_args(args),
    )
    print("\nTop hits")
    for hit in hits:
        line = f"{hit['score']:.4f}\t{hit['header']}\t{hit['start']}-{hit['end']}\t{hit['window']}"
        if "alignment" in hit:
            alignment = hit["alignment"]
            line += (
                f"\tq={alignment.query_start}-{alignment.query_end}"
                f"\tt={alignment.target_start}-{alignment.target_end}"
                f"\tskip={alignment.skipped_target_index}"
                f"\taligned={alignment.aligned_window}"
            )
            if alignment.peripheral_breakdown:
                periph = ",".join(
                    f"q{query_pos}@{target_idx}:{residue}:{energy:.4f}"
                    for query_pos, target_idx, residue, energy in alignment.peripheral_breakdown
                )
                line += f"\tperiph={periph}"
        print(line)


if __name__ == "__main__":
    main()
