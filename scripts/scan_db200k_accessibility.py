#!/usr/bin/env python3
"""Rank DB200K motif hits with sequence accessibility proxies.

This script is meant for cases where raw DB200K compatibility alone is not
enough because buried, rigid windows can outrank exposed, flexible ones.

Workflow:
1. Build a DB200K query profile from `--query-seq`.
2. Scan a FASTA with rigid or semiglobal alignment.
3. For each hit, compute lightweight sequence-side proxies for:
   - accessibility / surface-likeness
   - flexibility / disorder-likeness
The output keeps the raw DB200K score visible and reports every component
separately, so weighting can be changed later without rerunning the scan.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contact_alignment import db200k_scan  # noqa: E402

try:
    import _db200k_cli  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - repo layout fallback
    from examples import _db200k_cli  # type: ignore  # noqa: E402


SEQ_SURFACE_PROPENSITY = {
    # Coarse surface/disorder-leaning weights.
    "A": 0.35,
    "C": -0.10,
    "D": 0.95,
    "E": 0.95,
    "F": -0.75,
    "G": 0.55,
    "H": 0.35,
    "I": -0.70,
    "K": 1.00,
    "L": -0.65,
    "M": -0.35,
    "N": 0.70,
    "P": 0.80,
    "Q": 0.75,
    "R": 1.00,
    "S": 0.80,
    "T": 0.55,
    "V": -0.55,
    "W": -1.00,
    "Y": -0.45,
}

SEQ_FLEX_PROPENSITY = {
    # Coarse flexibility/disorder-leaning weights.
    "A": 0.45,
    "C": -0.10,
    "D": 0.70,
    "E": 0.70,
    "F": -0.45,
    "G": 1.00,
    "H": 0.15,
    "I": -0.55,
    "K": 0.55,
    "L": -0.45,
    "M": -0.10,
    "N": 0.55,
    "P": 0.95,
    "Q": 0.55,
    "R": 0.45,
    "S": 0.80,
    "T": 0.55,
    "V": -0.40,
    "W": -0.90,
    "Y": -0.25,
}

POLAR_OR_CHARGED = set("DEHKR NQST".replace(" ", ""))
HYDROPHOBIC = set("AILMFWVYC")
ACIDIC = set("DE")
BASIC = set("KRH")
CHARGED = ACIDIC | BASIC
COARSE_CLASSES = {
    "acidic": set("DE"),
    "basic": set("KRH"),
    "polar": set("NQST"),
    "gp": set("GP"),
    "hydrophobic": set("AILMFWVYC"),
}


@dataclass
class HitRecord:
    header: str
    start: int
    end: int
    window: str
    db200k_score: float
    seq_surface_proxy: float
    seq_flex_proxy: float
    seq_polar_fraction: float
    seq_acidic_fraction: float
    seq_gp_fraction: float
    seq_hydrophobe_fraction: float
    seq_class_entropy: float
    seq_transition_rate: float
    seq_repeat_fraction: float
    seq_acidic_run_fraction: float
    seq_basic_fraction: float
    seq_basic_run_fraction: float
    seq_charged_run_fraction: float
    seq_bonus: float
    sequence_rank_score: float
    breakdown: list[tuple[int, str, float]]


@dataclass
class CandidateWindow:
    header: str
    start: int
    end: int
    window: str
    context_seq: str
    context_start: int
    seq_bonus: float
    seq_surface_proxy: float
    seq_flex_proxy: float
    seq_polar_fraction: float
    seq_acidic_fraction: float
    seq_gp_fraction: float
    seq_hydrophobe_fraction: float
    seq_class_entropy: float
    seq_transition_rate: float
    seq_repeat_fraction: float
    seq_acidic_run_fraction: float
    seq_basic_fraction: float
    seq_basic_run_fraction: float
    seq_charged_run_fraction: float


def mean_scale(window: str, scale: dict[str, float]) -> float:
    return sum(scale[aa] for aa in window) / len(window)


def normalized_scale(window: str, scale: dict[str, float]) -> float:
    values = list(scale.values())
    lo = min(values)
    hi = max(values)
    raw = mean_scale(window, scale)
    return (raw - lo) / (hi - lo)


def normalized_shannon_entropy(labels: list[str], alphabet_size: int) -> float:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    length = len(labels)
    if length <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    max_entropy = math.log2(min(alphabet_size, length))
    return entropy / max_entropy if max_entropy > 0.0 else 0.0


def coarse_class_labels(window: str) -> list[str]:
    labels: list[str] = []
    for aa in window:
        for label, residues in COARSE_CLASSES.items():
            if aa in residues:
                labels.append(label)
                break
        else:
            labels.append("other")
    return labels


def max_same_residue_run(window: str) -> int:
    if not window:
        return 0
    best = 1
    current = 1
    for i in range(1, len(window)):
        if window[i] == window[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def max_acidic_run(window: str) -> int:
    return max_class_run(window, ACIDIC)


def max_basic_run(window: str) -> int:
    return max_class_run(window, BASIC)


def max_charged_run(window: str) -> int:
    return max_class_run(window, CHARGED)


def max_class_run(window: str, residue_class: set[str]) -> int:
    best = 0
    current = 0
    for aa in window:
        if aa in residue_class:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def compute_sequence_proxies(window: str) -> dict[str, float]:
    length = len(window)
    acidic_fraction = sum(aa in ACIDIC for aa in window) / length
    basic_fraction = sum(aa in BASIC for aa in window) / length
    polar_fraction = sum(aa in POLAR_OR_CHARGED for aa in window) / length
    gp_fraction = sum(aa in {"G", "P"} for aa in window) / length
    hydrophobe_fraction = sum(aa in HYDROPHOBIC for aa in window) / length
    surface_proxy = normalized_scale(window, SEQ_SURFACE_PROPENSITY)
    flex_proxy = normalized_scale(window, SEQ_FLEX_PROPENSITY)
    class_labels = coarse_class_labels(window)
    class_entropy = normalized_shannon_entropy(class_labels, alphabet_size=5)
    transition_rate = (
        sum(class_labels[i] != class_labels[i - 1] for i in range(1, length)) / (length - 1)
        if length > 1
        else 0.0
    )
    return {
        "seq_surface_proxy": surface_proxy,
        "seq_flex_proxy": flex_proxy,
        "seq_polar_fraction": polar_fraction,
        "seq_acidic_fraction": acidic_fraction,
        "seq_basic_fraction": basic_fraction,
        "seq_gp_fraction": gp_fraction,
        "seq_hydrophobe_fraction": hydrophobe_fraction,
        "seq_class_entropy": class_entropy,
        "seq_transition_rate": transition_rate,
        "seq_repeat_fraction": max_same_residue_run(window) / length,
        "seq_acidic_run_fraction": max_acidic_run(window) / length,
        "seq_basic_run_fraction": max_basic_run(window) / length,
        "seq_charged_run_fraction": max_charged_run(window) / length,
    }


def compute_sequence_bonus(
    window: str,
    *,
    surface_weight: float,
    flexibility_weight: float,
    polar_weight: float,
    gp_weight: float,
    hydrophobe_penalty: float,
    complexity_weight: float,
    transition_weight: float,
    repeat_penalty: float,
    acidic_run_penalty: float,
    basic_run_penalty: float,
    charged_run_penalty: float,
    acidic_excess_penalty: float,
) -> tuple[float, dict[str, float]]:
    metrics = compute_sequence_proxies(window)
    bonus = (
        surface_weight * metrics["seq_surface_proxy"]
        + flexibility_weight * metrics["seq_flex_proxy"]
        + polar_weight * metrics["seq_polar_fraction"]
        + gp_weight * metrics["seq_gp_fraction"]
        + complexity_weight * metrics["seq_class_entropy"]
        + transition_weight * metrics["seq_transition_rate"]
        - hydrophobe_penalty * metrics["seq_hydrophobe_fraction"]
        - repeat_penalty * metrics["seq_repeat_fraction"]
        - acidic_run_penalty * metrics["seq_acidic_run_fraction"]
        - basic_run_penalty * metrics["seq_basic_run_fraction"]
        - charged_run_penalty * metrics["seq_charged_run_fraction"]
        - acidic_excess_penalty * max(0.0, metrics["seq_acidic_fraction"] - 0.25)
    )
    metrics["seq_bonus"] = bonus
    return bonus, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _db200k_cli.add_profile_args(parser)
    _db200k_cli.add_alignment_args(parser)
    parser.add_argument("--fasta", required=True, help="FASTA to scan.")
    parser.add_argument("--top-k", type=int, default=50, help="Top hits to keep after sequence-aware ranking.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional raw DB200K cutoff; retain windows with score <= threshold before ranking.",
    )
    parser.add_argument(
        "--surface-weight",
        type=float,
        default=0.25,
        help="Weight for sequence accessibility proxy.",
    )
    parser.add_argument(
        "--flexibility-weight",
        type=float,
        default=0.20,
        help="Weight for sequence flexibility/disorder proxy.",
    )
    parser.add_argument(
        "--polar-weight",
        type=float,
        default=0.10,
        help="Weight for polar/charged residue fraction.",
    )
    parser.add_argument(
        "--gp-weight",
        type=float,
        default=0.25,
        help="Weight for Gly/Pro fraction.",
    )
    parser.add_argument(
        "--hydrophobe-penalty",
        type=float,
        default=0.15,
        help="Penalty for hydrophobic content in the sequence-only accessibility bonus.",
    )
    parser.add_argument(
        "--complexity-weight",
        type=float,
        default=0.30,
        help="Weight for coarse class entropy; rewards mixed loop-like composition.",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=0.20,
        help="Weight for local chemistry transitions; rewards alternating residue classes.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.40,
        help="Penalty for long same-residue runs.",
    )
    parser.add_argument(
        "--acidic-run-penalty",
        type=float,
        default=0.60,
        help="Penalty for extended acidic runs.",
    )
    parser.add_argument(
        "--basic-run-penalty",
        type=float,
        default=0.55,
        help="Penalty for extended basic runs.",
    )
    parser.add_argument(
        "--charged-run-penalty",
        type=float,
        default=0.90,
        help="Penalty for long generic charged runs regardless of sign.",
    )
    parser.add_argument(
        "--acidic-excess-penalty",
        type=float,
        default=0.80,
        help="Penalty for very high acidic fraction.",
    )
    parser.add_argument(
        "--heuristic-prefilter-threshold",
        type=float,
        default=None,
        help="Only retain windows with sequence-only heuristic bonus >= this value before DB200K scoring.",
    )
    parser.add_argument(
        "--heuristic-prefilter-top-n",
        type=int,
        default=None,
        help="Retain only the top N windows by sequence-only heuristic bonus before DB200K scoring.",
    )
    parser.add_argument(
        "--report-heuristic-only",
        action="store_true",
        help="Only report heuristic prefilter selectivity and exit without DB200K scoring.",
    )
    parser.add_argument(
        "--heuristic-report-thresholds",
        type=str,
        default="0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated heuristic thresholds to count in report mode.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional TSV output path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--progress-every-windows",
        type=int,
        default=None,
        help="Optional progress print frequency.",
    )
    return parser.parse_args()

def iter_candidate_windows(
    args: argparse.Namespace,
    window_len: int,
) -> Iterable[CandidateWindow]:
    alignment_mode = args.alignment_mode
    target_flank = args.target_flank
    progress_every_windows = args.progress_every_windows
    heap: list[tuple[float, int, CandidateWindow]] = []
    seen = 0
    retained = 0
    idx = 0

    for header, sequence in db200k_scan.iter_fasta_records(args.fasta):
        if len(sequence) < window_len:
            continue
        for start0 in range(len(sequence) - window_len + 1):
            window = sequence[start0 : start0 + window_len]
            if any(res not in db200k_scan.CENTER_ALPHABET_SET for res in window):
                continue
            seen += 1
            seq_bonus, metrics = compute_sequence_bonus(
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
            if (
                args.heuristic_prefilter_threshold is not None
                and seq_bonus < args.heuristic_prefilter_threshold
            ):
                if progress_every_windows is not None and seen % progress_every_windows == 0:
                    print(
                        f"[heuristic-prefilter] windows={seen} retained={retained}",
                        flush=True,
                    )
                continue
            if alignment_mode == "rigid":
                context_start = start0
                context_seq = window
            else:
                context_start = max(0, start0 - target_flank)
                context_end = min(len(sequence), start0 + window_len + target_flank)
                context_seq = sequence[context_start:context_end]
                if any(res not in db200k_scan.CENTER_ALPHABET_SET for res in context_seq):
                    continue
            candidate = CandidateWindow(
                header=header,
                start=start0 + 1,
                end=start0 + window_len,
                window=window,
                context_seq=context_seq,
                context_start=context_start,
                seq_bonus=seq_bonus,
                seq_surface_proxy=metrics["seq_surface_proxy"],
                seq_flex_proxy=metrics["seq_flex_proxy"],
                seq_polar_fraction=metrics["seq_polar_fraction"],
                seq_acidic_fraction=metrics["seq_acidic_fraction"],
                seq_gp_fraction=metrics["seq_gp_fraction"],
                seq_hydrophobe_fraction=metrics["seq_hydrophobe_fraction"],
                seq_class_entropy=metrics["seq_class_entropy"],
                seq_transition_rate=metrics["seq_transition_rate"],
                seq_repeat_fraction=metrics["seq_repeat_fraction"],
                seq_acidic_run_fraction=metrics["seq_acidic_run_fraction"],
                seq_basic_fraction=metrics["seq_basic_fraction"],
                seq_basic_run_fraction=metrics["seq_basic_run_fraction"],
                seq_charged_run_fraction=metrics["seq_charged_run_fraction"],
            )
            retained += 1
            if args.heuristic_prefilter_top_n is None:
                yield candidate
            else:
                entry = (candidate.seq_bonus, idx, candidate)
                idx += 1
                if len(heap) < args.heuristic_prefilter_top_n:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
            if progress_every_windows is not None and seen % progress_every_windows == 0:
                print(
                    f"[heuristic-prefilter] windows={seen} retained={retained}",
                    flush=True,
                )

    if args.heuristic_prefilter_top_n is not None:
        shortlisted = [entry[2] for entry in heap]
        shortlisted.sort(key=lambda candidate: candidate.seq_bonus, reverse=True)
        for candidate in shortlisted:
            yield candidate


def report_heuristic_counts(args: argparse.Namespace, window_len: int) -> None:
    thresholds = [float(part) for part in args.heuristic_report_thresholds.split(",") if part.strip()]
    thresholds.sort()
    counts = {threshold: 0 for threshold in thresholds}
    total = 0
    for header, sequence in db200k_scan.iter_fasta_records(args.fasta):
        if len(sequence) < window_len:
            continue
        for start0 in range(len(sequence) - window_len + 1):
            window = sequence[start0 : start0 + window_len]
            if any(res not in db200k_scan.CENTER_ALPHABET_SET for res in window):
                continue
            total += 1
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
            for threshold in thresholds:
                if seq_bonus >= threshold:
                    counts[threshold] += 1
    print(f"windows_total\t{total}")
    for threshold in thresholds:
        kept = counts[threshold]
        pct = 100.0 * kept / total if total else 0.0
        print(f"threshold\t{threshold:.3f}\tcount\t{kept}\tpct\t{pct:.4f}")
    if args.heuristic_prefilter_top_n is not None and total:
        pct = 100.0 * args.heuristic_prefilter_top_n / total
        print(
            f"topn\t{args.heuristic_prefilter_top_n}\tpct\t{pct:.6f}"
        )


def score_candidate(
    candidate: CandidateWindow,
    profiles,
    args: argparse.Namespace,
) -> tuple[float, list[tuple[int, str, float]]]:
    prefilter_result: tuple[float, list[tuple[int, str, float]]] | None = None
    if args.prefilter_score_threshold is not None:
        prefilter_result = db200k_scan.score_window(
            candidate.window,
            profiles,
            score_mode=args.prefilter_score_mode,
            uncertainty_floor=args.uncertainty_floor,
        )
        if prefilter_result[0] > args.prefilter_score_threshold:
            raise ValueError("prefilter_rejected")

    if args.alignment_mode == "rigid":
        if prefilter_result is not None and args.prefilter_score_mode == args.score_mode:
            return prefilter_result
        return db200k_scan.score_window(
            candidate.window,
            profiles,
            score_mode=args.score_mode,
            uncertainty_floor=args.uncertainty_floor,
        )

    alignment = db200k_scan.score_window_semiglobal(
        candidate.context_seq,
        profiles,
        score_mode=args.score_mode,
        uncertainty_floor=args.uncertainty_floor,
        max_target_gaps=args.max_target_gaps,
        min_aligned_positions=args.min_aligned_positions,
        target_gap_penalty=args.target_gap_penalty,
        target_offset=candidate.context_start,
        peripheral_flank_weight=args.peripheral_flank_weight,
    )
    return alignment.score, alignment.breakdown


def iter_hits(args: argparse.Namespace, profiles) -> Iterable[HitRecord]:
    for candidate in iter_candidate_windows(args, len(profiles)):
        try:
            db200k_score, breakdown = score_candidate(candidate, profiles, args)
        except ValueError as exc:
            if str(exc) == "prefilter_rejected":
                continue
            raise
        if args.score_threshold is not None and db200k_score > args.score_threshold:
            continue
        yield HitRecord(
            header=candidate.header,
            start=candidate.start,
            end=candidate.end,
            window=candidate.window,
            db200k_score=db200k_score,
            seq_surface_proxy=candidate.seq_surface_proxy,
            seq_flex_proxy=candidate.seq_flex_proxy,
            seq_polar_fraction=candidate.seq_polar_fraction,
            seq_acidic_fraction=candidate.seq_acidic_fraction,
            seq_gp_fraction=candidate.seq_gp_fraction,
            seq_hydrophobe_fraction=candidate.seq_hydrophobe_fraction,
            seq_class_entropy=candidate.seq_class_entropy,
            seq_transition_rate=candidate.seq_transition_rate,
            seq_repeat_fraction=candidate.seq_repeat_fraction,
            seq_acidic_run_fraction=candidate.seq_acidic_run_fraction,
            seq_basic_fraction=candidate.seq_basic_fraction,
            seq_basic_run_fraction=candidate.seq_basic_run_fraction,
            seq_charged_run_fraction=candidate.seq_charged_run_fraction,
            seq_bonus=candidate.seq_bonus,
            sequence_rank_score=db200k_score - candidate.seq_bonus,
            breakdown=list(breakdown),
        )


def keep_top_k(hits: Iterable[HitRecord], top_k: int) -> list[HitRecord]:
    heap: list[tuple[float, int, HitRecord]] = []
    idx = 0
    for hit in hits:
        entry = (-hit.sequence_rank_score, idx, hit)
        idx += 1
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    ranked = [entry[2] for entry in heap]
    ranked.sort(key=lambda hit: hit.sequence_rank_score)
    return ranked


def write_hits(path: str, hits: list[HitRecord]) -> None:
    fields = [
        "sequence_rank_score",
        "db200k_score",
        "seq_bonus",
        "seq_surface_proxy",
        "seq_flex_proxy",
        "seq_polar_fraction",
        "seq_acidic_fraction",
        "seq_basic_fraction",
        "seq_gp_fraction",
        "seq_hydrophobe_fraction",
        "seq_class_entropy",
        "seq_transition_rate",
        "seq_repeat_fraction",
        "seq_acidic_run_fraction",
        "seq_basic_run_fraction",
        "seq_charged_run_fraction",
        "header",
        "start",
        "end",
        "window",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for hit in hits:
            writer.writerow(
                {
                    "sequence_rank_score": f"{hit.sequence_rank_score:.6f}",
                    "db200k_score": f"{hit.db200k_score:.6f}",
                    "seq_bonus": f"{hit.seq_bonus:.6f}",
                    "seq_surface_proxy": f"{hit.seq_surface_proxy:.6f}",
                    "seq_flex_proxy": f"{hit.seq_flex_proxy:.6f}",
                    "seq_polar_fraction": f"{hit.seq_polar_fraction:.6f}",
                    "seq_acidic_fraction": f"{hit.seq_acidic_fraction:.6f}",
                    "seq_basic_fraction": f"{hit.seq_basic_fraction:.6f}",
                    "seq_gp_fraction": f"{hit.seq_gp_fraction:.6f}",
                    "seq_hydrophobe_fraction": f"{hit.seq_hydrophobe_fraction:.6f}",
                    "seq_class_entropy": f"{hit.seq_class_entropy:.6f}",
                    "seq_transition_rate": f"{hit.seq_transition_rate:.6f}",
                    "seq_repeat_fraction": f"{hit.seq_repeat_fraction:.6f}",
                    "seq_acidic_run_fraction": f"{hit.seq_acidic_run_fraction:.6f}",
                    "seq_basic_run_fraction": f"{hit.seq_basic_run_fraction:.6f}",
                    "seq_charged_run_fraction": f"{hit.seq_charged_run_fraction:.6f}",
                    "header": hit.header,
                    "start": hit.start,
                    "end": hit.end,
                    "window": hit.window,
                }
            )


def print_hits(hits: list[HitRecord]) -> None:
    for hit in hits:
        line = (
            f"{hit.sequence_rank_score:.4f}\t"
            f"db200k={hit.db200k_score:.4f}\t"
            f"seq_bonus={hit.seq_bonus:.4f}\t"
            f"surf={hit.seq_surface_proxy:.3f}\t"
            f"flex={hit.seq_flex_proxy:.3f}\t"
            f"polar={hit.seq_polar_fraction:.3f}\t"
            f"acidic={hit.seq_acidic_fraction:.3f}\t"
            f"basic={hit.seq_basic_fraction:.3f}\t"
            f"gp={hit.seq_gp_fraction:.3f}\t"
            f"hydro={hit.seq_hydrophobe_fraction:.3f}\t"
            f"complex={hit.seq_class_entropy:.3f}\t"
            f"trans={hit.seq_transition_rate:.3f}\t"
            f"rep={hit.seq_repeat_fraction:.3f}\t"
            f"acidrun={hit.seq_acidic_run_fraction:.3f}\t"
            f"basicrun={hit.seq_basic_run_fraction:.3f}\t"
            f"chgrun={hit.seq_charged_run_fraction:.3f}"
        )
        line += f"\t{hit.header}\t{hit.start}-{hit.end}\t{hit.window}"
        print(line)


def main() -> None:
    args = parse_args()
    profiles = _db200k_cli.build_profiles_from_args(args)
    if args.report_heuristic_only:
        report_heuristic_counts(args, len(profiles))
        return
    hits = keep_top_k(iter_hits(args, profiles), args.top_k)
    print_hits(hits)
    if args.out:
        write_hits(args.out, hits)


if __name__ == "__main__":
    main()
