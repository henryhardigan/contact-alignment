"""Ungapped window-based alignment search (Phase 1).

This module implements the first phase of alignment discovery: finding the
best ungapped window pairs between two protein sequences. The algorithms
scan for optimal starting points that can later be extended with gaps.

Key functions:
- best_ungapped_window_pair: Find single best window using MJ scoring
- best_ungapped_window_pair_clustal: Find best window using Clustal scoring
- best_ungapped_window_pair_clustal_topk: Find top-K windows
- seed_windows: Generate filtered list of seed candidates

The window search can use exhaustive enumeration or random sampling for
large sequence pairs.
"""

import heapq
import random
from typing import Optional

from .amino_acid_properties import AA20, AA20_STR, apply_mj_overrides, has_charge_run
from .clustering import _kmer_hits, clustal_pair_score
from .scoring import context_bonus_aligned, get_mj_scorer

_ORIENTATION_FORWARD = "forward"
_ORIENTATION_REVERSE = "reverse"
_ORIENTATION_BOTH = "both"


def _normalize_orientation(orientation: str) -> str:
    o = orientation.lower()
    if o not in (_ORIENTATION_FORWARD, _ORIENTATION_REVERSE, _ORIENTATION_BOTH):
        raise ValueError(f"Unknown orientation {orientation!r}; use 'forward', 'reverse', or 'both'")
    return o


def _reverse_start_index(n2: int, window: int, j_rev: int) -> int:
    return n2 - j_rev - window


def overlaps_fraction(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Compute overlap fraction between two intervals.

    Returns the overlap length divided by the length of the shorter interval.

    Args:
        a_start: Start of first interval.
        a_end: End of first interval (inclusive).
        b_start: Start of second interval.
        b_end: End of second interval (inclusive).

    Returns:
        Overlap fraction in [0, 1].

    Example:
        >>> overlaps_fraction(0, 10, 5, 15)  # Overlap 5-10 = 6, shorter = 11
        0.545
    """
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    if right < left:
        return 0.0
    overlap = right - left + 1
    denom = min(a_end - a_start + 1, b_end - b_start + 1)
    if denom <= 0:
        return 0.0
    return overlap / denom


def best_ungapped_window_pair(
    seq1: str,
    seq2: str,
    mj: dict[tuple[str, str], float],
    window: int,
    mode: str = "min",
    max_evals: int = 0,
    rng_seed: int = 1,
    unknown_policy: str = "error",
    context_bonus: bool = False,
    orientation: str = "both",
    return_orientation: bool = False,
) -> tuple:
    """Find the best ungapped window pair using MJ scoring.

    Scans all (or sampled) window pairs and returns the one with the
    best (minimum or maximum) total MJ score.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        mj: MJ matrix dictionary.
        window: Window length.
        mode: 'min' for most negative (strongest complement) or 'max'.
        max_evals: If >0 and less than total pairs, use random sampling.
        rng_seed: Random seed for sampling.
        unknown_policy: How to handle unknown residues.
        context_bonus: Whether to apply context bonuses.
        orientation: 'forward', 'reverse', or 'both' (seq2 orientation).
        return_orientation: If True, append orientation label to return tuple.

    Returns:
        Tuple of (score, start_i, start_j, per_pos_scores) where:
            - start_i: 0-based start in seq1
            - start_j: 0-based start in seq2
            - per_pos_scores: List of scores for each window position
        Returns (None, None, None, None) if no valid windows.
        If return_orientation is True, appends orientation ('forward'/'reverse').

    Example:
        >>> score, i, j, per_pos = best_ungapped_window_pair(seq1, seq2, mj, 10)
        >>> print(f"Best window at ({i}, {j}) with score {score}")
    """
    orientation = _normalize_orientation(orientation)
    if window <= 0:
        return None, None, None, None
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return None, None, None, None

    def run(seq2_local: str) -> tuple[Optional[float], Optional[int], Optional[int], Optional[list[float]]]:
        total_pairs = (n1 - window + 1) * (len(seq2_local) - window + 1)
        use_sampling = max_evals and max_evals > 0 and total_pairs > max_evals

        mj_score = get_mj_scorer(mj)
        s1 = seq1
        s2 = seq2_local
        w = window

        if mode == "min":
            best = float("inf")
            def better(x: float, y: float) -> bool:
                return x < y
        else:
            best = float("-inf")
            def better(x: float, y: float) -> bool:
                return x > y

        best_i: Optional[int] = None
        best_j: Optional[int] = None
        best_per: Optional[list[float]] = None
        use_context = context_bonus

        if use_sampling:
            rng = random.Random(rng_seed)
            for _ in range(max_evals):
                i = rng.randrange(0, n1 - w + 1)
                j = rng.randrange(0, len(s2) - w + 1)
                per: list[float] = []
                s = 0.0
                valid = True
                for k in range(w):
                    try:
                        v = mj_score(s1[i + k], s2[j + k])
                    except KeyError:
                        if unknown_policy == "error":
                            raise
                        if unknown_policy == "skip":
                            valid = False
                            break
                        v = 0.0
                    per.append(v)  # type: ignore
                    s += v  # type: ignore
                if not valid:
                    continue
                if use_context:
                    s += sum(
                        context_bonus_aligned(
                            s1[i : i + w],
                            s2[j : j + w],
                            mj,
                            unknown_policy=unknown_policy,
                        )
                    )
                if better(s, best):
                    best = s
                    best_i = i
                    best_j = j
                    best_per = per
        else:
            for i in range(0, n1 - w + 1):
                for j in range(0, len(s2) - w + 1):
                    per = []
                    s = 0.0
                    invalid = False
                    for k in range(w):
                        try:
                            v = mj_score(s1[i + k], s2[j + k])
                        except KeyError:
                            if unknown_policy == "error":
                                raise
                            if unknown_policy == "skip":
                                invalid = True
                                break
                            v = 0.0
                        per.append(v)  # type: ignore
                        s += v  # type: ignore
                    if invalid:
                        continue
                    if use_context:
                        s += sum(
                            context_bonus_aligned(
                                s1[i : i + w],
                                s2[j : j + w],
                                mj,
                                unknown_policy=unknown_policy,
                            )
                        )
                    if better(s, best):
                        best = s
                        best_i = i
                        best_j = j
                        best_per = per

        return best, best_i, best_j, best_per

    def pack(
        score: Optional[float],
        i: Optional[int],
        j: Optional[int],
        per: Optional[list[float]],
        orient: str,
    ) -> tuple:
        if return_orientation:
            return score, i, j, per, orient
        return score, i, j, per

    best_f = run(seq2)
    if orientation == _ORIENTATION_FORWARD:
        return pack(best_f[0], best_f[1], best_f[2], best_f[3], _ORIENTATION_FORWARD)

    best_r = run(seq2[::-1])
    if best_r[2] is not None:
        best_r = (best_r[0], best_r[1], _reverse_start_index(n2, window, best_r[2]), best_r[3])

    if orientation == _ORIENTATION_REVERSE:
        return pack(best_r[0], best_r[1], best_r[2], best_r[3], _ORIENTATION_REVERSE)

    if best_f[0] is None:
        return pack(best_r[0], best_r[1], best_r[2], best_r[3], _ORIENTATION_REVERSE)
    if best_r[0] is None:
        return pack(best_f[0], best_f[1], best_f[2], best_f[3], _ORIENTATION_FORWARD)

    take_rev = best_r[0] < best_f[0] if mode == "min" else best_r[0] > best_f[0]
    if take_rev:
        return pack(best_r[0], best_r[1], best_r[2], best_r[3], _ORIENTATION_REVERSE)
    return pack(best_f[0], best_f[1], best_f[2], best_f[3], _ORIENTATION_FORWARD)


def best_ungapped_window_pair_clustal(
    seq1: str,
    seq2: str,
    *,
    window: int,
    max_evals: int = 0,
    rng_seed: int = 1,
    unknown_policy: str = "error",
    prefilter_min_strong: int = 0,
    prefilter_min_identity: int = 0,
    kmer_len: int = 0,
    kmer_min: int = 1,
    require_positions2: Optional[list[tuple[int, set[str]]]] = None,
    rank_by: str = "score",
    filter_charge_runs: bool = False,
    orientation: str = "both",
    return_orientation: bool = False,
) -> tuple:
    """Find best ungapped window pair using Clustal similarity scoring.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        window: Window length.
        max_evals: Maximum evaluations (0 = exhaustive).
        rng_seed: Random seed for sampling.
        unknown_policy: How to handle unknown residues.
        prefilter_min_strong: Minimum strong similarities required.
        prefilter_min_identity: Minimum identities required.
        kmer_len: K-mer length for prefiltering (0 = disabled).
        kmer_min: Minimum k-mer matches required.
        require_positions2: Position constraints as (offset, allowed_set) tuples.
        rank_by: 'score' or 'identity' for ranking windows.
        filter_charge_runs: Skip windows containing charge runs.
        orientation: 'forward', 'reverse', or 'both' (seq2 orientation).
        return_orientation: If True, append orientation label to return tuple.

    Returns:
        Tuple of (score, start_i, start_j, identity_count).
    """
    orientation = _normalize_orientation(orientation)
    if window <= 0:
        return None, None, None, None
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return None, None, None, None

    def rank_key(score: float, ident: int) -> tuple:
        if rank_by == "identity":
            return (ident, score)
        return (score,)

    def run(seq2_local: str) -> tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
        prefix: Optional[list[int]] = None
        if kmer_len and kmer_len > 0 and kmer_len <= window:
            seq1_kmers: set[str] = set()
            for i in range(0, n1 - kmer_len + 1):
                kmer = seq1[i : i + kmer_len]
                if any(c not in AA20 for c in kmer):
                    if unknown_policy == "error":
                        raise ValueError("Unknown residue in Clustal k-mer scan")
                    continue
                seq1_kmers.add(kmer)
            if not seq1_kmers:
                return None, None, None, None
            hits = _kmer_hits(seq2_local, kmer_len, seq1_kmers, unknown_policy=unknown_policy)
            if not hits:
                return None, None, None, None
            prefix = [0]
            for h in hits:
                prefix.append(prefix[-1] + (1 if h else 0))

        total_pairs = (n1 - window + 1) * (len(seq2_local) - window + 1)
        use_sampling = max_evals and max_evals > 0 and total_pairs > max_evals

        best = float("-inf")
        best_ident: Optional[int] = None
        best_i: Optional[int] = None
        best_j: Optional[int] = None

        def evaluate_window(i: int, j: int) -> Optional[tuple[float, int]]:
            if require_positions2:
                if any(seq2_local[j + off] not in allowed for off, allowed in require_positions2):
                    return None
            if prefix is not None:
                end = j + (window - kmer_len + 1)
                if end > len(prefix) - 1:
                    return None
                if prefix[end] - prefix[j] < kmer_min:
                    return None
            if filter_charge_runs:
                win1 = seq1[i : i + window]
                win2 = seq2_local[j : j + window]
                if has_charge_run(win1) or has_charge_run(win2):
                    return None

            s = 0.0
            strong = 0
            ident = 0
            for k in range(window):
                a = seq1[i + k]
                b = seq2_local[j + k]
                v = clustal_pair_score(a, b)
                if v is None:
                    if unknown_policy == "error":
                        raise ValueError("Unknown residue in Clustal scan")
                    if unknown_policy == "skip":
                        return None
                    v = 0.0
                if a == b and a in AA20:
                    ident += 1
                if v >= 0.5:
                    strong += 1
                s += v

            if prefilter_min_strong and strong < prefilter_min_strong:
                return None
            if prefilter_min_identity and ident < prefilter_min_identity:
                return None

            return s, ident

        if use_sampling:
            rng = random.Random(rng_seed)
            for _ in range(max_evals):
                i = rng.randrange(0, n1 - window + 1)
                j = rng.randrange(0, len(seq2_local) - window + 1)
                result = evaluate_window(i, j)
                if result is None:
                    continue
                s, ident = result
                if rank_key(s, ident) > rank_key(best, best_ident or 0):
                    best, best_ident, best_i, best_j = s, ident, i, j
        else:
            for i in range(0, n1 - window + 1):
                for j in range(0, len(seq2_local) - window + 1):
                    result = evaluate_window(i, j)
                    if result is None:
                        continue
                    s, ident = result
                    if rank_key(s, ident) > rank_key(best, best_ident or 0):
                        best, best_ident, best_i, best_j = s, ident, i, j

        if best_i is None or best_j is None:
            return None, None, None, None
        return best, best_i, best_j, best_ident

    def pack(
        score: Optional[float],
        i: Optional[int],
        j: Optional[int],
        ident: Optional[int],
        orient: str,
    ) -> tuple:
        if return_orientation:
            return score, i, j, ident, orient
        return score, i, j, ident

    best_f = run(seq2)
    if orientation == _ORIENTATION_FORWARD:
        return pack(best_f[0], best_f[1], best_f[2], best_f[3], _ORIENTATION_FORWARD)

    best_r = run(seq2[::-1])
    if best_r[2] is not None:
        best_r = (best_r[0], best_r[1], _reverse_start_index(n2, window, best_r[2]), best_r[3])

    if orientation == _ORIENTATION_REVERSE:
        return pack(best_r[0], best_r[1], best_r[2], best_r[3], _ORIENTATION_REVERSE)

    if best_f[0] is None:
        return pack(best_r[0], best_r[1], best_r[2], best_r[3], _ORIENTATION_REVERSE)
    if best_r[0] is None:
        return pack(best_f[0], best_f[1], best_f[2], best_f[3], _ORIENTATION_FORWARD)

    if rank_key(best_r[0], best_r[3] or 0) > rank_key(best_f[0], best_f[3] or 0):
        return pack(best_r[0], best_r[1], best_r[2], best_r[3], _ORIENTATION_REVERSE)
    return pack(best_f[0], best_f[1], best_f[2], best_f[3], _ORIENTATION_FORWARD)


def best_ungapped_window_pair_clustal_topk(
    seq1: str,
    seq2: str,
    *,
    window: int,
    max_evals: int = 0,
    rng_seed: int = 1,
    unknown_policy: str = "error",
    prefilter_min_strong: int = 0,
    prefilter_min_identity: int = 0,
    kmer_len: int = 0,
    kmer_min: int = 1,
    require_positions2: Optional[list[tuple[int, set[str]]]] = None,
    rank_by: str = "identity",
    topk: int = 3,
    filter_charge_runs: bool = False,
    orientation: str = "both",
    include_orientation: bool = False,
) -> list[tuple]:
    """Find top-K best ungapped windows using Clustal scoring.

    Similar to best_ungapped_window_pair_clustal but returns multiple
    top-scoring windows for subsequent evaluation.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        window: Window length.
        max_evals: Maximum evaluations (0 = exhaustive).
        rng_seed: Random seed for sampling.
        unknown_policy: How to handle unknown residues.
        prefilter_min_strong: Minimum strong similarities required.
        prefilter_min_identity: Minimum identities required.
        kmer_len: K-mer length for prefiltering.
        kmer_min: Minimum k-mer matches required.
        require_positions2: Position constraints.
        rank_by: 'identity' or 'score' for ranking.
        topk: Number of top windows to return.
        filter_charge_runs: Skip windows with charge runs.
        orientation: 'forward', 'reverse', or 'both' (seq2 orientation).
        include_orientation: If True, append orientation label to tuples.

    Returns:
        List of (score, identity_count, start_i, start_j) tuples,
        sorted by ranking criterion (best first).
        If include_orientation is True, tuples include orientation at end.
    """
    orientation = _normalize_orientation(orientation)
    if window <= 0:
        return []
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return []

    def run(seq2_local: str) -> list[tuple[float, int, int, int]]:
        prefix: Optional[list[int]] = None
        if kmer_len and kmer_len > 0 and kmer_len <= window:
            seq1_kmers: set[str] = set()
            for i in range(0, n1 - kmer_len + 1):
                kmer = seq1[i : i + kmer_len]
                if any(c not in AA20 for c in kmer):
                    if unknown_policy == "error":
                        raise ValueError("Unknown residue in Clustal k-mer scan")
                    continue
                seq1_kmers.add(kmer)
            if not seq1_kmers:
                return []
            hits = _kmer_hits(seq2_local, kmer_len, seq1_kmers, unknown_policy=unknown_policy)
            if not hits:
                return []
            prefix = [0]
            for h in hits:
                prefix.append(prefix[-1] + (1 if h else 0))

        total_pairs = (n1 - window + 1) * (len(seq2_local) - window + 1)
        use_sampling = max_evals and max_evals > 0 and total_pairs > max_evals
        rng = random.Random(rng_seed)

        heap: list[tuple[tuple[int, float], float, int, int, int]] = []

        def push_hit(score: float, ident: int, i: int, j: int) -> None:
            key = (ident, score) if rank_by == "identity" else (int(score * 1000), score)
            item = (key, score, ident, i, j)
            if len(heap) < topk:
                heapq.heappush(heap, item)
            else:
                if key > heap[0][0]:
                    heapq.heapreplace(heap, item)

        def evaluate_window(i: int, j: int) -> Optional[tuple[float, int]]:
            if require_positions2:
                if any(seq2_local[j + off] not in allowed for off, allowed in require_positions2):
                    return None
            if prefix is not None:
                end = j + (window - kmer_len + 1)
                if end > len(prefix) - 1:
                    return None
                if prefix[end] - prefix[j] < kmer_min:
                    return None
            if filter_charge_runs:
                win1 = seq1[i : i + window]
                win2 = seq2_local[j : j + window]
                if has_charge_run(win1) or has_charge_run(win2):
                    return None

            s = 0.0
            strong = 0
            ident = 0
            for k in range(window):
                a = seq1[i + k]
                b = seq2_local[j + k]
                v = clustal_pair_score(a, b)
                if v is None:
                    if unknown_policy == "error":
                        raise ValueError("Unknown residue in Clustal scan")
                    if unknown_policy == "skip":
                        return None
                    v = 0.0
                if a == b and a in AA20:
                    ident += 1
                if v >= 0.5:
                    strong += 1
                s += v

            if prefilter_min_strong and strong < prefilter_min_strong:
                return None
            if prefilter_min_identity and ident < prefilter_min_identity:
                return None

            return s, ident

        if use_sampling:
            for _ in range(max_evals):
                i = rng.randrange(0, n1 - window + 1)
                j = rng.randrange(0, len(seq2_local) - window + 1)
                result = evaluate_window(i, j)
                if result:
                    push_hit(result[0], result[1], i, j)
        else:
            for i in range(0, n1 - window + 1):
                for j in range(0, len(seq2_local) - window + 1):
                    result = evaluate_window(i, j)
                    if result:
                        push_hit(result[0], result[1], i, j)

        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(score, ident, i, j) for _key, score, ident, i, j in results]

    def rank_key(score: float, ident: int) -> tuple:
        return (ident, score) if rank_by == "identity" else (int(score * 1000), score)

    def with_orientation(
        hits: list[tuple[float, int, int, int]],
        orient: str,
    ) -> list[tuple]:
        if include_orientation:
            return [(score, ident, i, j, orient) for score, ident, i, j in hits]
        return hits  # type: ignore[return-value]

    hits_f = run(seq2)
    if orientation == _ORIENTATION_FORWARD:
        return with_orientation(hits_f, _ORIENTATION_FORWARD)

    hits_r = run(seq2[::-1])
    hits_r = [
        (score, ident, i, _reverse_start_index(n2, window, j)) for score, ident, i, j in hits_r
    ]
    if orientation == _ORIENTATION_REVERSE:
        return with_orientation(hits_r, _ORIENTATION_REVERSE)

    merged_tagged = (
        [(score, ident, i, j, _ORIENTATION_FORWARD) for score, ident, i, j in hits_f]
        + [(score, ident, i, j, _ORIENTATION_REVERSE) for score, ident, i, j in hits_r]
    )
    merged_tagged.sort(key=lambda x: rank_key(x[0], x[1]), reverse=True)
    merged_tagged = merged_tagged[:topk]
    if include_orientation:
        return merged_tagged  # type: ignore[return-value]
    return [(score, ident, i, j) for score, ident, i, j, _orient in merged_tagged]


def _seed_windows_single(
    seq1: str,
    seq2: str,
    mj: dict[tuple[str, str], float],
    *,
    window: int,
    score_max: float,
    kmax: int,
    kmin: int,
    prefilter_len: int = 0,
    prefilter_score_max: Optional[float] = None,
    prefilter_kmax: int = 0,
    prefilter_kmin: int = 0,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> list[tuple[float, int, int]]:
    if window <= 0:
        return []
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return []

    hits_np = _seed_windows_numpy(
        seq1, seq2, mj,
        window=window,
        score_max=score_max,
        kmax=kmax,
        kmin=kmin,
        prefilter_len=prefilter_len,
        prefilter_score_max=prefilter_score_max,
        prefilter_kmax=prefilter_kmax,
        prefilter_kmin=prefilter_kmin,
        unknown_policy=unknown_policy,
        context_bonus=context_bonus,
    )
    if hits_np is not None:
        return hits_np

    mj_score = get_mj_scorer(mj)

    def score_window(i: int, j: int, win: int) -> Optional[float]:
        s = 0.0
        for k in range(win):
            try:
                v = mj_score(seq1[i + k], seq2[j + k])
            except KeyError:
                if unknown_policy == "error":
                    raise
                if unknown_policy == "skip":
                    return None
                v = 0.0
            s += v  # type: ignore
        if context_bonus:
            s += sum(
                context_bonus_aligned(
                    seq1[i : i + win],
                    seq2[j : j + win],
                    mj,
                    unknown_policy=unknown_policy,
                )
            )
        return s

    hits: list[tuple[float, int, int]] = []
    use_prefilter = (
        prefilter_len
        and prefilter_len > 0
        and prefilter_len < window
        and n1 >= prefilter_len
        and n2 >= prefilter_len
    )

    if use_prefilter:
        pf_score_max = prefilter_score_max if prefilter_score_max is not None else score_max
        pf_hits: list[tuple[float, int, int]] = []

        for i in range(0, n1 - prefilter_len + 1):
            for j in range(0, n2 - prefilter_len + 1):
                s = score_window(i, j, prefilter_len)
                if s is None:
                    continue
                pf_hits.append((s, i, j))

        if not pf_hits:
            return []

        pf_hits.sort(key=lambda x: x[0])
        pf_filtered = [h for h in pf_hits if h[0] <= pf_score_max]

        if len(pf_filtered) < prefilter_kmin:
            pf_filtered = pf_hits[: min(prefilter_kmin, len(pf_hits))]
        if prefilter_kmax and len(pf_filtered) > prefilter_kmax:
            pf_filtered = pf_filtered[:prefilter_kmax]

        candidates = [(i, j) for _, i, j in pf_filtered]
        for i, j in candidates:
            if i + window > n1 or j + window > n2:
                continue
            s = score_window(i, j, window)
            if s is None:
                continue
            hits.append((s, i, j))
    else:
        for i in range(0, n1 - window + 1):
            for j in range(0, n2 - window + 1):
                s = score_window(i, j, window)
                if s is None:
                    continue
                hits.append((s, i, j))

    if not hits:
        return []

    hits.sort(key=lambda x: x[0])
    filtered = [h for h in hits if h[0] <= score_max]

    if len(filtered) < kmin:
        filtered = hits[: min(kmin, len(hits))]
    if kmax and len(filtered) > kmax:
        filtered = filtered[:kmax]

    return filtered


def seed_windows(
    seq1: str,
    seq2: str,
    mj: dict[tuple[str, str], float],
    *,
    window: int,
    score_max: float,
    kmax: int,
    kmin: int,
    prefilter_len: int = 0,
    prefilter_score_max: Optional[float] = None,
    prefilter_kmax: int = 0,
    prefilter_kmin: int = 0,
    unknown_policy: str = "error",
    context_bonus: bool = False,
    orientation: str = "both",
    include_orientation: bool = False,
) -> list[tuple]:
    """Generate filtered list of seed windows for alignment.

    Scans for window pairs that pass the score threshold and returns
    them sorted by score. Can use prefiltering with shorter windows
    for efficiency.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        mj: MJ matrix dictionary.
        window: Window length for final scoring.
        score_max: Maximum score threshold (more negative = stricter).
        kmax: Maximum number of seeds to return (0 = unlimited).
        kmin: Minimum seeds to keep even if they don't pass threshold.
        prefilter_len: Shorter window length for prefiltering (0 = disabled).
        prefilter_score_max: Score threshold for prefiltering.
        prefilter_kmax: Max candidates from prefilter.
        prefilter_kmin: Min candidates from prefilter.
        unknown_policy: How to handle unknown residues.
        context_bonus: Whether to apply context bonuses.
        orientation: 'forward', 'reverse', or 'both' (seq2 orientation).
        include_orientation: If True, append orientation label to tuples.

    Returns:
        List of (score, start_i, start_j) tuples sorted by score
        (most negative first).
        If include_orientation is True, tuples include orientation at end.
    """
    orientation = _normalize_orientation(orientation)
    if window <= 0:
        return []
    n2 = len(seq2)
    if len(seq1) < window or n2 < window:
        return []

    def with_orientation(
        hits: list[tuple[float, int, int]],
        orient: str,
    ) -> list[tuple]:
        if include_orientation:
            return [(score, i, j, orient) for score, i, j in hits]
        return hits  # type: ignore[return-value]

    hits_f = _seed_windows_single(
        seq1,
        seq2,
        mj,
        window=window,
        score_max=score_max,
        kmax=kmax,
        kmin=kmin,
        prefilter_len=prefilter_len,
        prefilter_score_max=prefilter_score_max,
        prefilter_kmax=prefilter_kmax,
        prefilter_kmin=prefilter_kmin,
        unknown_policy=unknown_policy,
        context_bonus=context_bonus,
    )
    if orientation == _ORIENTATION_FORWARD:
        return with_orientation(hits_f, _ORIENTATION_FORWARD)

    hits_r = _seed_windows_single(
        seq1,
        seq2[::-1],
        mj,
        window=window,
        score_max=score_max,
        kmax=kmax,
        kmin=kmin,
        prefilter_len=prefilter_len,
        prefilter_score_max=prefilter_score_max,
        prefilter_kmax=prefilter_kmax,
        prefilter_kmin=prefilter_kmin,
        unknown_policy=unknown_policy,
        context_bonus=context_bonus,
    )
    hits_r = [(score, i, _reverse_start_index(n2, window, j)) for score, i, j in hits_r]
    if orientation == _ORIENTATION_REVERSE:
        return with_orientation(hits_r, _ORIENTATION_REVERSE)

    merged_tagged = (
        [(score, i, j, _ORIENTATION_FORWARD) for score, i, j in hits_f]
        + [(score, i, j, _ORIENTATION_REVERSE) for score, i, j in hits_r]
    )
    merged_tagged.sort(key=lambda x: x[0])
    if kmax and len(merged_tagged) > kmax:
        merged_tagged = merged_tagged[:kmax]
    if include_orientation:
        return merged_tagged  # type: ignore[return-value]
    return [(score, i, j) for score, i, j, _orient in merged_tagged]


def _seed_windows_numpy(
    seq1: str,
    seq2: str,
    mj: dict[tuple[str, str], float],
    *,
    window: int,
    score_max: float,
    kmax: int,
    kmin: int,
    prefilter_len: int,
    prefilter_score_max: Optional[float],
    prefilter_kmax: int,
    prefilter_kmin: int,
    unknown_policy: str,
    context_bonus: bool,
) -> Optional[list[tuple[float, int, int]]]:
    """NumPy-accelerated seed window finding.

    Returns None if NumPy is not available or context_bonus is enabled.
    Otherwise returns the same result as seed_windows but faster.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    if context_bonus:
        return None  # Context bonus not supported in NumPy path

    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return []

    # Convert sequences to indices
    aa_to_idx = {aa: i for i, aa in enumerate(AA20_STR)}
    unk_idx = len(AA20_STR)
    s1_idx = np.fromiter(
        (aa_to_idx.get(ch, unk_idx) for ch in seq1.upper()),
        dtype=np.int16,
        count=n1,
    )
    s2_idx = np.fromiter(
        (aa_to_idx.get(ch, unk_idx) for ch in seq2.upper()),
        dtype=np.int16,
        count=n2,
    )

    if unknown_policy == "error":
        if (s1_idx == unk_idx).any() or (s2_idx == unk_idx).any():
            raise KeyError("Unknown residue encountered with --unknown error")

    # Build MJ score matrix
    mj_arr = np.zeros((unk_idx + 1, unk_idx + 1), dtype=np.float32)
    for i, a in enumerate(AA20_STR):
        for j, b in enumerate(AA20_STR):
            val = mj.get((a, b), mj.get((b, a)))
            val = apply_mj_overrides(a, b, val)
            if val is not None:
                mj_arr[i, j] = float(val)

    nwin = n2 - window + 1
    if nwin <= 0:
        return []

    # Create sliding windows using strides
    s2_stride = s2_idx.strides[0]
    windows_full = np.lib.stride_tricks.as_strided(
        s2_idx, shape=(nwin, window), strides=(s2_stride, s2_stride)
    )
    invalid_full = None
    if unknown_policy == "skip":
        invalid_full = (windows_full == unk_idx).any(axis=1)

    # Check for prefiltering
    use_prefilter = (
        prefilter_len
        and prefilter_len > 0
        and prefilter_len < window
        and n1 >= prefilter_len
        and n2 >= prefilter_len
    )

    if use_prefilter:
        pf_len = prefilter_len
        pf_score_max = prefilter_score_max if prefilter_score_max is not None else score_max
        pf_nwin = n2 - pf_len + 1
        windows_pf = np.lib.stride_tricks.as_strided(
            s2_idx, shape=(pf_nwin, pf_len), strides=(s2_stride, s2_stride)
        )
        invalid_pf = None
        if unknown_policy == "skip":
            invalid_pf = (windows_pf == unk_idx).any(axis=1)

    def heap_push(
        heap: list[tuple[float, float, int, int]],
        score: float,
        i: int,
        j: int,
        max_size: int,
    ) -> None:
        key = -score  # Min-heap for most negative
        if len(heap) < max_size:
            heapq.heappush(heap, (key, score, i, j))
            return
        if key > heap[0][0]:
            heapq.heapreplace(heap, (key, score, i, j))

    filtered_heap: list[tuple[float, float, int, int]] = []
    overall_heap: list[tuple[float, float, int, int]] = []
    all_filtered: list[tuple[float, int, int]] = []
    filtered_count = 0

    for i in range(0, n1 - window + 1):
        s1w = s1_idx[i : i + window]
        if unknown_policy == "skip" and (s1w == unk_idx).any():
            continue

        if use_prefilter:
            s1_pf = s1_idx[i : i + pf_len]
            if unknown_policy == "skip" and (s1_pf == unk_idx).any():
                continue
            pf_scores = mj_arr[s1_pf[:, None], windows_pf.T].sum(axis=0)
            if invalid_pf is not None:
                pf_scores = pf_scores.copy()
                pf_scores[invalid_pf] = np.inf

            pf_mask = pf_scores <= pf_score_max
            if pf_mask.sum() < prefilter_kmin:
                if pf_nwin <= prefilter_kmin:
                    pf_idx = np.arange(pf_nwin, dtype=np.int64)
                else:
                    pf_idx = np.argpartition(pf_scores, prefilter_kmin - 1)[:prefilter_kmin]
            else:
                pf_idx = np.nonzero(pf_mask)[0]
                if prefilter_kmax and len(pf_idx) > prefilter_kmax:
                    tmp = pf_scores.copy()
                    tmp[~pf_mask] = np.inf
                    pf_idx = np.argpartition(tmp, prefilter_kmax - 1)[:prefilter_kmax]

            if len(pf_idx) == 0:
                continue
            pf_idx = pf_idx[pf_idx <= n2 - window]
            if len(pf_idx) == 0:
                continue

            windows_sel = windows_full[pf_idx]
            scores = mj_arr[s1w[:, None], windows_sel.T].sum(axis=0)
            if invalid_full is not None:
                invalid_sel = invalid_full[pf_idx]
                if invalid_sel.any():
                    scores = scores.copy()
                    scores[invalid_sel] = np.inf
            j_candidates = pf_idx
        else:
            scores = mj_arr[s1w[:, None], windows_full.T].sum(axis=0)
            if invalid_full is not None:
                scores = scores.copy()
                scores[invalid_full] = np.inf
            j_candidates = None

        if j_candidates is None:
            j_idx = np.arange(nwin, dtype=np.int64)
        else:
            j_idx = j_candidates

        # Track minimum kmin scores
        if kmin > 0:
            if len(scores) <= kmin:
                idx = np.arange(len(scores))
            else:
                idx = np.argpartition(scores, kmin - 1)[:kmin]
            for jj in idx:
                score = float(scores[jj])
                heap_push(overall_heap, score, i, int(j_idx[jj]), kmin)

        # Track scores passing threshold
        if score_max is not None:
            mask = scores <= score_max
            filtered_count += int(mask.sum())
            if kmax == 0:
                idx = np.nonzero(mask)[0]
                for jj in idx:
                    all_filtered.append((float(scores[jj]), i, int(j_idx[jj])))
            elif mask.any():
                if mask.sum() > kmax:
                    tmp = scores.copy()
                    tmp[~mask] = np.inf
                    idx = np.argpartition(tmp, kmax - 1)[:kmax]
                else:
                    idx = np.nonzero(mask)[0]
                for jj in idx:
                    score = float(scores[jj])
                    heap_push(filtered_heap, score, i, int(j_idx[jj]), kmax)

    # Select final results
    if filtered_count >= kmin:
        if kmax == 0:
            hits = all_filtered
        else:
            hits = [(s, i, j) for _, s, i, j in filtered_heap]
    else:
        hits = [(s, i, j) for _, s, i, j in overall_heap]

    hits.sort(key=lambda x: x[0])
    return hits


def seed_null_best_scores(
    seq1: str,
    seq2: str,
    mj: dict[tuple[str, str], float],
    *,
    window: int,
    n: int,
    unknown_policy: str = "error",
    seed: Optional[int] = None,
    context_bonus: bool = False,
    orientation: str = "both",
) -> list[float]:
    """Generate null distribution for best window scores.

    Creates a null by shuffling seq2 and finding the best ungapped
    window pair for each shuffle.

    Args:
        seq1: First sequence (held constant).
        seq2: Second sequence (shuffled each iteration).
        mj: MJ matrix dictionary.
        window: Window length.
        n: Number of null samples.
        unknown_policy: How to handle unknown residues.
        seed: Random seed.
        context_bonus: Whether to apply context bonuses.
        orientation: 'forward', 'reverse', or 'both' (seq2 orientation).

    Returns:
        List of best window scores from shuffled sequences.
    """
    rng = random.Random(seed)
    scores: list[float] = []
    base = list(seq2)
    for _ in range(n):
        rng.shuffle(base)
        shuf = "".join(base)
        score, _, _, _ = best_ungapped_window_pair(
            seq1,
            shuf,
            mj,
            window=window,
            mode="min",
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
            orientation=orientation,
        )
        if score is not None:
            scores.append(float(score))
    return scores
