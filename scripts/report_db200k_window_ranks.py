#!/usr/bin/env python3
"""Report exact proteome-wide ranks for fixed windows by combined DB200K score.

This is meant for cases where heuristic prefiltering should be bypassed and the
question is simply: where do specific windows rank against all windows in a
FASTA when using the final combined score

    sequence_rank_score = db200k_score - seq_bonus

The scoring path intentionally matches the sequence-aware heuristic weighting
used by scripts/scan_db200k_accessibility.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contact_alignment import db200k_scan  # noqa: E402

try:
    import _db200k_cli  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - repo layout fallback
    from examples import _db200k_cli  # type: ignore  # noqa: E402

from scripts.scan_db200k_accessibility import compute_sequence_bonus  # noqa: E402


@dataclass
class TargetWindow:
    label: str
    sequence: str
    db200k_score: float
    seq_bonus: float
    final_score: float
    better_window_count: int = 0
    equal_window_count: int = 0
    better_entry_count: int = 0
    equal_entry_count: int = 0
    exact_match_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _db200k_cli.add_profile_args(parser)
    _db200k_cli.add_alignment_args(parser)
    parser.add_argument("--fasta", required=True, help="FASTA to scan.")
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Target in label=SEQUENCE form. Repeat for multiple windows.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional TSV output path.",
    )
    parser.add_argument(
        "--progress-every-windows",
        type=int,
        default=500000,
        help="Print progress every N scanned windows.",
    )
    parser.add_argument(
        "--surface-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--flexibility-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--polar-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--gp-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--hydrophobe-penalty",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--complexity-weight",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--acidic-run-penalty",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--basic-run-penalty",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--charged-run-penalty",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--acidic-excess-penalty",
        type=float,
        default=0.80,
    )
    return parser.parse_args()


def parse_targets(args: argparse.Namespace, profiles) -> list[TargetWindow]:
    targets: list[TargetWindow] = []
    expected_len = len(profiles)
    for raw in args.target:
        if "=" not in raw:
            raise ValueError(f"Target must be label=SEQUENCE, got: {raw}")
        label, seq = raw.split("=", 1)
        seq = seq.strip().upper()
        if len(seq) != expected_len:
            raise ValueError(
                f"Target {label!r} has length {len(seq)}, expected {expected_len}"
            )
        db200k_score, _ = db200k_scan.score_window(
            seq,
            profiles,
            score_mode=args.score_mode,
            uncertainty_floor=args.uncertainty_floor,
        )
        seq_bonus, _ = compute_sequence_bonus(
            seq,
            surface_weight=args.surface_weight,
            flexibility_weight=args.flexibility_weight,
            polar_weight=args.polar_weight,
            gp_weight=args.gp_weight,
            hydrophobe_penalty=args.hydrophobe_penalty,
            complexity_weight=args.complexity_weight,
            transition_weight=args.transition_weight,
            repeat_penalty=args.repeat_penalty,
            acidic_run_penalty=args.acidic_run_penalty,
            basic_run_penalty=args.basic_run_penalty,
            charged_run_penalty=args.charged_run_penalty,
            acidic_excess_penalty=args.acidic_excess_penalty,
        )
        targets.append(
            TargetWindow(
                label=label,
                sequence=seq,
                db200k_score=db200k_score,
                seq_bonus=seq_bonus,
                final_score=db200k_score - seq_bonus,
            )
        )
    return targets


def main() -> None:
    args = parse_args()
    profiles = _db200k_cli.build_profiles_from_args(args)
    targets = parse_targets(args, profiles)
    window_len = len(profiles)
    total_windows = 0
    best_per_entry: dict[str, float] = {}

    for header, sequence in db200k_scan.iter_fasta_records(args.fasta):
        if len(sequence) < window_len:
            continue
        entry_best: float | None = None
        for start0 in range(len(sequence) - window_len + 1):
            window = sequence[start0 : start0 + window_len]
            if any(res not in db200k_scan.CENTER_ALPHABET_SET for res in window):
                continue
            total_windows += 1
            db200k_score, _ = db200k_scan.score_window(
                window,
                profiles,
                score_mode=args.score_mode,
                uncertainty_floor=args.uncertainty_floor,
            )
            seq_bonus, _ = compute_sequence_bonus(
                window,
                surface_weight=args.surface_weight,
                flexibility_weight=args.flexibility_weight,
                polar_weight=args.polar_weight,
                gp_weight=args.gp_weight,
                hydrophobe_penalty=args.hydrophobe_penalty,
                complexity_weight=args.complexity_weight,
                transition_weight=args.transition_weight,
                repeat_penalty=args.repeat_penalty,
                acidic_run_penalty=args.acidic_run_penalty,
                basic_run_penalty=args.basic_run_penalty,
                charged_run_penalty=args.charged_run_penalty,
                acidic_excess_penalty=args.acidic_excess_penalty,
            )
            final_score = db200k_score - seq_bonus
            if entry_best is None or final_score < entry_best:
                entry_best = final_score
            for target in targets:
                if final_score < target.final_score:
                    target.better_window_count += 1
                elif final_score == target.final_score:
                    target.equal_window_count += 1
                if window == target.sequence:
                    target.exact_match_count += 1
            if args.progress_every_windows and total_windows % args.progress_every_windows == 0:
                print(f"[scan] windows={total_windows}", flush=True)
        if entry_best is not None:
            best_per_entry[header] = entry_best

    for entry_best in best_per_entry.values():
        for target in targets:
            if entry_best < target.final_score:
                target.better_entry_count += 1
            elif entry_best == target.final_score:
                target.equal_entry_count += 1

    fields = [
        "label",
        "sequence",
        "db200k_score",
        "seq_bonus",
        "final_score",
        "window_rank",
        "window_percentile",
        "equal_window_count",
        "exact_match_count",
        "entry_rank",
        "entry_percentile",
        "equal_entry_count",
        "total_windows",
        "total_entries",
    ]

    rows: list[dict[str, object]] = []
    total_entries = len(best_per_entry)
    for target in targets:
        window_rank = target.better_window_count + 1
        entry_rank = target.better_entry_count + 1
        rows.append(
            {
                "label": target.label,
                "sequence": target.sequence,
                "db200k_score": f"{target.db200k_score:.6f}",
                "seq_bonus": f"{target.seq_bonus:.6f}",
                "final_score": f"{target.final_score:.6f}",
                "window_rank": window_rank,
                "window_percentile": f"{100.0 * window_rank / total_windows:.6f}",
                "equal_window_count": target.equal_window_count,
                "exact_match_count": target.exact_match_count,
                "entry_rank": entry_rank,
                "entry_percentile": f"{100.0 * entry_rank / total_entries:.6f}",
                "equal_entry_count": target.equal_entry_count,
                "total_windows": total_windows,
                "total_entries": total_entries,
            }
        )

    writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
