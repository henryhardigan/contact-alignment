from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path

from contact_alignment import db200k_scan


@dataclass(frozen=True)
class DesignedCandidate:
    sequence: str
    forward_score: float
    reciprocal_score: float
    total_score: float


def top_profile_choices(
    profiles: list[db200k_scan.PositionProfile],
    *,
    top_n: int = 4,
) -> list[list[tuple[str, float]]]:
    """Return the most favorable residue choices per query position."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")
    choices: list[list[tuple[str, float]]] = []
    for profile in profiles:
        ranked = sorted(
            (
                (residue, float(profile.energies[idx]))
                for idx, residue in enumerate(db200k_scan.CENTER_ALPHABET)
            ),
            key=lambda item: item[1],
        )[:top_n]
        choices.append(ranked)
    return choices


def beam_sample_favored_sequences(
    profiles: list[db200k_scan.PositionProfile],
    *,
    top_n: int = 4,
    beam_width: int = 1000,
) -> list[tuple[str, float]]:
    """Enumerate top candidate sequences by direct forward DB200K score."""
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")
    choices = top_profile_choices(profiles, top_n=top_n)
    beam: list[tuple[float, str]] = [(0.0, "")]
    for pos_choices in choices:
        next_beam: list[tuple[float, str]] = []
        for partial_score, partial_seq in beam:
            for residue, residue_score in pos_choices:
                entry = (partial_score + residue_score, partial_seq + residue)
                if len(next_beam) < beam_width:
                    heapq.heappush(next_beam, (-entry[0], entry[1]))
                elif -entry[0] > next_beam[0][0]:
                    heapq.heapreplace(next_beam, (-entry[0], entry[1]))
        beam = sorted([(-score, seq) for score, seq in next_beam], key=lambda item: item[0])
    return [(seq, score) for score, seq in beam]


def reciprocal_rescore_candidates(
    query_seq: str,
    db_root: str | Path,
    candidates: list[tuple[str, float]],
    *,
    profile_strategy: str = "hierarchical_5x5_3x3_1x1",
    one_by_one_mode: str = "directed",
    one_by_one_matrix_tsv: str | Path | None = None,
    strong_threshold: int = 100,
    strong_weight_3x3: float = 0.8,
    weak_weight_3x3: float = 0.5,
    shrinkage_prior_3x3: float = 25.0,
    shrinkage_prior_5x5: float = 10.0,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
    rescue_mode: str = "none",
    top_k: int | None = None,
) -> list[DesignedCandidate]:
    """Rescore candidate sequences by reciprocal DB200K compatibility."""
    resources = db200k_scan.load_profile_resources(
        db_root,
        profile_strategy=profile_strategy,
        one_by_one_mode=one_by_one_mode,
        one_by_one_matrix_tsv=one_by_one_matrix_tsv,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )
    rows: list[DesignedCandidate] = []
    for candidate_seq, forward_score in candidates:
        candidate_profiles = db200k_scan.build_query_profiles_from_resources(
            candidate_seq,
            resources,
            profile_strategy=profile_strategy,
            strong_threshold=strong_threshold,
            strong_weight_3x3=strong_weight_3x3,
            weak_weight_3x3=weak_weight_3x3,
            shrinkage_prior_3x3=shrinkage_prior_3x3,
            shrinkage_prior_5x5=shrinkage_prior_5x5,
        )
        reciprocal_score, _ = db200k_scan.score_window(
            query_seq,
            candidate_profiles,
            rescue_mode=rescue_mode,
        )
        rows.append(
            DesignedCandidate(
                sequence=candidate_seq,
                forward_score=forward_score,
                reciprocal_score=reciprocal_score,
                total_score=forward_score + reciprocal_score,
            )
        )
    rows.sort(key=lambda row: (row.total_score, row.forward_score, row.reciprocal_score, row.sequence))
    if top_k is not None:
        return rows[:top_k]
    return rows
