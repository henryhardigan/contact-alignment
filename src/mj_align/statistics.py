"""Statistical analysis and null distribution generation.

This module provides functions for assessing the statistical significance
of alignment scores through:

- Empirical null distributions via shuffling and circular shifts
- Analytic null distributions using probability theory
- Hypothesis testing functions (hypergeometric, Poisson-binomial)
- Quantile calculations and window scoring

These tools help distinguish biologically meaningful alignments from
chance similarities.
"""

import math
import random
from collections.abc import Iterable
from typing import Optional

from .amino_acid_properties import AA20
from .scoring import score_aligned


def shuffle_preserve_gaps(seq: str, gap_char: str = "-") -> str:
    """Shuffle residues while keeping gap positions fixed.

    Creates a random permutation of non-gap residues while maintaining
    gap positions unchanged. Useful for generating null distributions
    that preserve alignment structure.

    Args:
        seq: Sequence to shuffle (may contain gaps).
        gap_char: Gap character to preserve. Default '-'.

    Returns:
        Shuffled sequence with gaps in original positions.

    Example:
        >>> random.seed(42)
        >>> shuffle_preserve_gaps("A-BC--D")
        'D-CB--A'  # Gaps at positions 2, 5, 6 preserved
    """
    residues = [c for c in seq if c != gap_char]
    random.shuffle(residues)

    out: list[str] = []
    r = 0
    for c in seq:
        if c == gap_char:
            out.append(gap_char)
        else:
            out.append(residues[r])
            r += 1
    return "".join(out)


def null_distribution(
    fixed_seq: str,
    shuffle_seq: str,
    mj: dict[tuple[str, str], float],
    n_iter: int = 1000,
    *,
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> list[float]:
    """Generate null distribution by shuffling one sequence.

    Creates an empirical null distribution by repeatedly shuffling
    one sequence (preserving gaps) and scoring against the fixed sequence.

    Args:
        fixed_seq: Sequence to hold constant.
        shuffle_seq: Sequence to shuffle each iteration.
        mj: MJ matrix dictionary.
        n_iter: Number of shuffle iterations. Default 1000.
        gap_char: Gap character. Default '-'.
        gap_penalty: Gap penalty for scoring. Default None (ignore gaps).
        unknown_policy: How to handle unknown residues.
        context_bonus: Whether to apply context bonuses.

    Returns:
        List of total scores from each shuffled alignment.

    Example:
        >>> null_scores = null_distribution(seq1, seq2, mj, n_iter=1000)
        >>> p_value = sum(1 for s in null_scores if s <= observed) / len(null_scores)
    """
    scores: list[float] = []
    for _ in range(n_iter):
        shuf = shuffle_preserve_gaps(shuffle_seq, gap_char=gap_char)
        total, _ = score_aligned(
            fixed_seq,
            shuf,
            mj,
            gap_char=gap_char,
            gap_penalty=gap_penalty,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        scores.append(total)
    return scores


def circular_shift(seq: str, k: int) -> str:
    """Circularly shift a sequence by k positions.

    Rotates the sequence so that positions wrap around. Useful for
    generating null distributions that preserve sequence composition
    and local structure.

    Args:
        seq: Sequence to shift.
        k: Number of positions to shift (positive = right shift).

    Returns:
        Circularly shifted sequence.

    Example:
        >>> circular_shift("ABCD", 1)
        'DABC'
        >>> circular_shift("ABCD", 3)
        'BCDA'
    """
    if not seq:
        return seq
    k = k % len(seq)
    if k == 0:
        return seq
    return seq[-k:] + seq[:-k]


def circular_shift_null(
    fixed_seq: str,
    shift_seq: str,
    mj: dict[tuple[str, str], float],
    *,
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    context_bonus: bool = False,
) -> list[float]:
    """Generate null distribution by circular shifts.

    Creates a null distribution using all (or sampled) circular shifts
    of one sequence against the fixed sequence. The observed alignment
    (shift k=0) is excluded.

    Args:
        fixed_seq: Sequence to hold constant.
        shift_seq: Sequence to circularly shift.
        mj: MJ matrix dictionary.
        gap_char: Gap character. Default '-'.
        gap_penalty: Gap penalty for scoring.
        unknown_policy: How to handle unknown residues.
        n_samples: If provided and < (L-1), randomly sample this many shifts.
        seed: Random seed for sampling.
        context_bonus: Whether to apply context bonuses.

    Returns:
        List of scores from shifted alignments.

    Raises:
        ValueError: If sequences have different lengths.
    """
    L = len(fixed_seq)
    if L != len(shift_seq):
        raise ValueError("Aligned sequences must be same length")
    if L < 2:
        return []

    # Generate shift indices (excluding k=0)
    shifts = list(range(1, L))
    if n_samples is not None and n_samples > 0 and n_samples < len(shifts):
        rng = random.Random(seed)
        shifts = rng.sample(shifts, n_samples)

    scores: list[float] = []
    for k in shifts:
        shifted = circular_shift(shift_seq, k)
        total, _ = score_aligned(
            fixed_seq,
            shifted,
            mj,
            gap_char=gap_char,
            gap_penalty=gap_penalty,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        scores.append(total)
    return scores


def quantile(values: list[float], q: float) -> float:
    """Compute quantile using linear interpolation.

    Args:
        values: List of numeric values.
        q: Quantile to compute (0.0 to 1.0).

    Returns:
        The q-th quantile value.

    Raises:
        ValueError: If values list is empty.

    Example:
        >>> quantile([1, 2, 3, 4, 5], 0.5)  # Median
        3.0
        >>> quantile([1, 2, 3, 4, 5], 0.25)
        2.0
    """
    if not values:
        raise ValueError("quantile() requires a non-empty list")
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    xs = sorted(values)
    n = len(xs)
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def best_window_score(
    per_pos: list[Optional[float]],
    window: int,
    *,
    mode: str = "min",
    none_as: Optional[float] = 0.0,
) -> tuple[Optional[float], Optional[int]]:
    """Find the best contiguous window score over per-position values.

    Scans all windows of the given length and returns the best score
    (minimum or maximum depending on mode).

    Args:
        per_pos: Per-position scores (None for gaps/skipped positions).
        window: Window size.
        mode: Optimization mode:
            - 'min': Find most negative window (strongest complement)
            - 'max': Find most positive window
        none_as: Value to use for None positions. If None, skip windows
            containing None values.

    Returns:
        Tuple of (best_score, 1-based_start_position) or (None, None)
        if no valid windows exist.

    Raises:
        ValueError: If window <= 0.

    Example:
        >>> per_pos = [-5.0, -10.0, -15.0, -8.0, -3.0]
        >>> best_window_score(per_pos, 3, mode="min")
        (-33.0, 2)  # Positions 2-4 sum to -33
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(per_pos):
        return None, None

    best: Optional[float] = None
    best_start: Optional[int] = None

    for start in range(0, len(per_pos) - window + 1):
        chunk = per_pos[start: start + window]

        # Handle None values
        if none_as is None:
            if any(v is None for v in chunk):
                continue
            s = float(sum(chunk))  # type: ignore
        else:
            s = float(sum((none_as if v is None else float(v)) for v in chunk))

        # Update best
        if best is None:
            best, best_start = s, start + 1  # 1-based index
        else:
            if mode == "min" and s < best:
                best, best_start = s, start + 1
            elif mode == "max" and s > best:
                best, best_start = s, start + 1

    return best, best_start


def scan_null_best_window(
    fixed_seq: str,
    other_seq: str,
    mj: dict[tuple[str, str], float],
    *,
    window: int,
    null_method: str,
    n: int,
    shuffle_which: str,
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    seed: Optional[int] = None,
    mode: str = "min",
    context_bonus: bool = False,
) -> list[float]:
    """Generate scan-aware null distribution over best window scores.

    For each null draw, generates a random alignment (shuffle or shift)
    then finds the best window score within that alignment. This accounts
    for the multiple testing inherent in window scanning.

    Args:
        fixed_seq: Sequence to hold constant.
        other_seq: Sequence to randomize.
        mj: MJ matrix dictionary.
        window: Window size for best-window calculation.
        null_method: 'shuffle' or 'shift'.
        n: Number of null samples.
        shuffle_which: Which sequence to randomize ('seq1' or 'seq2').
        gap_char: Gap character.
        gap_penalty: Gap penalty for scoring.
        unknown_policy: How to handle unknown residues.
        seed: Random seed.
        mode: 'min' or 'max' for best window selection.
        context_bonus: Whether to apply context bonuses.

    Returns:
        List of best-window scores (one per null sample).

    Raises:
        ValueError: If null_method or shuffle_which is invalid.
    """
    if null_method not in {"shuffle", "shift"}:
        raise ValueError("null_method must be 'shuffle' or 'shift'")
    if shuffle_which not in {"seq1", "seq2"}:
        raise ValueError("shuffle_which must be 'seq1' or 'seq2'")

    s1 = fixed_seq
    s2 = other_seq
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    scores: list[float] = []

    if null_method == "shuffle":
        rng = random.Random(seed)

        if shuffle_which == "seq2":
            fixed = s1
            shuf_base = s2
        else:
            fixed = s2
            shuf_base = s1

        # Pre-compute gap positions for efficient shuffling
        gap_positions = set(i for i, c in enumerate(shuf_base) if c == gap_char)
        residues = [c for c in shuf_base if c != gap_char]

        for _ in range(n):
            rng.shuffle(residues)
            out = []
            r = 0
            for i in range(len(shuf_base)):
                if i in gap_positions:
                    out.append(gap_char)
                else:
                    out.append(residues[r])
                    r += 1
            shuf = "".join(out)

            total, per_pos = score_aligned(
                fixed,
                shuf,
                mj,
                gap_char=gap_char,
                gap_penalty=gap_penalty,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )
            best, _ = best_window_score(per_pos, window, mode=mode, none_as=0.0)
            if best is not None:
                scores.append(best)

    else:  # shift
        L = len(s1)
        if L < 2:
            return []

        # Generate shift indices
        shifts = list(range(1, L))
        if n > 0 and n < len(shifts):
            rng = random.Random(seed)
            shifts = rng.sample(shifts, n)

        if shuffle_which == "seq2":
            fixed = s1
            shift_base = s2
        else:
            fixed = s2
            shift_base = s1

        for k in shifts:
            shifted = circular_shift(shift_base, k)
            total, per_pos = score_aligned(
                fixed,
                shifted,
                mj,
                gap_char=gap_char,
                gap_penalty=gap_penalty,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )
            best, _ = best_window_score(per_pos, window, mode=mode, none_as=0.0)
            if best is not None:
                scores.append(best)

    return scores


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Compute Jaccard index (intersection over union) for two sets.

    Args:
        a: First collection of integers.
        b: Second collection of integers.

    Returns:
        Jaccard index in [0, 1]. Returns 1.0 if both sets are empty.

    Example:
        >>> jaccard([1, 2, 3], [2, 3, 4])
        0.5  # Intersection {2,3} / Union {1,2,3,4} = 2/4
    """
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


def hypergeom_p_at_least(k: int, N: int, K: int, n: int) -> float:
    """Compute P(X >= k) for hypergeometric distribution.

    Calculates the probability of observing at least k successes when
    drawing n items without replacement from a population of N items
    containing K successes.

    This is appropriate for testing whether anchor overlap between two
    sets is greater than expected by chance.

    Args:
        k: Observed overlap (successes drawn).
        N: Population size (total eligible positions).
        K: Number of success states in population (anchors in set A).
        n: Number of draws (anchors in set B).

    Returns:
        Tail probability P(X >= k), or NaN for invalid parameters.

    Example:
        >>> # 5 positions, set A has 2 anchors, set B has 2 anchors
        >>> # P(overlap >= 1) under random placement
        >>> hypergeom_p_at_least(k=1, N=5, K=2, n=2)
        0.7
    """
    if k <= 0:
        return 1.0
    if N <= 0 or K < 0 or n < 0 or K > N or n > N:
        return float("nan")

    max_x = min(K, n)
    if k > max_x:
        return 0.0

    denom = math.comb(N, n)
    num = 0
    for x in range(k, max_x + 1):
        num += math.comb(K, x) * math.comb(N - K, n - x)
    return num / denom


def poisson_binomial_p_ge(ps: list[float], x: int) -> float:
    """Compute P(X >= x) for Poisson-binomial distribution via DP.

    The Poisson-binomial distribution is the sum of independent
    Bernoulli random variables with potentially different success
    probabilities. Used for computing analytic p-values when each
    position has a different complement probability.

    Args:
        ps: List of per-trial success probabilities.
        x: Threshold count.

    Returns:
        P(X >= x) where X = sum of Bernoulli(p_i).

    Example:
        >>> # Three positions with different complement probabilities
        >>> poisson_binomial_p_ge([0.3, 0.5, 0.7], x=2)
        0.59  # P(at least 2 successes)
    """
    if x <= 0:
        return 1.0
    n = len(ps)
    if n == 0:
        return 0.0
    if x > n:
        return 0.0

    # dp[k] = P(sum == k) after processing i items
    dp = [0.0] * (n + 1)
    dp[0] = 1.0

    for p in ps:
        # Update backwards to avoid overwriting values we still need
        for k in range(n, 0, -1):
            dp[k] = dp[k] * (1.0 - p) + dp[k - 1] * p
        dp[0] = dp[0] * (1.0 - p)

    return sum(dp[x:])


def per_position_complement_prob_uniform(
    fixed_seq_aln: str,
    mj: dict[tuple[str, str], float],
    *,
    thr: float,
    gap_char: str = "-",
    unknown_policy: str = "error",
) -> list[Optional[float]]:
    """Compute per-position probability of complement under uniform model.

    For each position, calculates P(MJ(fixed[i], B) <= thr) where
    B is uniformly distributed over the 20 standard amino acids.
    This provides a taxon-agnostic baseline for significance testing.

    Args:
        fixed_seq_aln: Aligned sequence with fixed residues.
        mj: MJ matrix dictionary.
        thr: MJ threshold for complement detection.
        gap_char: Gap character. Default '-'.
        unknown_policy: How to handle unknown residues.

    Returns:
        List of per-position probabilities (None for gaps/unknown).

    Raises:
        ValueError: If unknown residue with 'error' policy.

    Example:
        >>> probs = per_position_complement_prob_uniform("ACDEF", mj, thr=-25.0)
        >>> probs[0]  # P(random residue complements A at threshold -25)
        0.15
    """
    s = fixed_seq_aln.strip().upper()
    out: list[Optional[float]] = []

    aa_list = sorted(AA20)
    for i, a in enumerate(s, start=1):
        if a == gap_char:
            out.append(None)
            continue
        if a not in AA20:
            if unknown_policy == "error":
                raise ValueError(
                    f"Unknown residue {a!r} at pos {i} in fixed sequence"
                )
            if unknown_policy == "skip":
                out.append(None)
                continue
            # unknown_policy == "zero"
            out.append(0.0)
            continue

        # Count how many of 20 AAs have MJ <= thr with residue a
        ok = 0
        for b in aa_list:
            val = mj.get((a, b), mj.get((b, a)))
            if val is None:
                if unknown_policy == "error":
                    raise ValueError(f"Missing MJ value for pair {a},{b}")
                if unknown_policy == "skip":
                    continue
                val = 0.0
            if float(val) <= thr:
                ok += 1
        out.append(ok / 20.0)

    return out


def modelc_uniform_anchor_stats(
    seq1_aln: str,
    seq2_aln: str,
    mj: dict[tuple[str, str], float],
    *,
    thr: float,
    gap_char: str = "-",
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> tuple[int, float, float, float, float]:
    """Compute analytic anchor statistics under uniform partner model.

    Model C assumes partner residues are uniformly distributed over
    the 20 amino acids. Computes observed anchor count and tail
    probabilities in both conditioning directions.

    Args:
        seq1_aln: First aligned sequence.
        seq2_aln: Second aligned sequence.
        mj: MJ matrix dictionary.
        thr: MJ threshold for anchor detection.
        gap_char: Gap character.
        unknown_policy: How to handle unknown residues.
        context_bonus: Whether to apply context bonuses.

    Returns:
        Tuple of (x_obs, ex_seq1_fixed, p_seq1_fixed, ex_seq2_fixed, p_seq2_fixed):
            - x_obs: Observed number of anchor positions
            - ex_seq1_fixed: E[X] with seq1 fixed, partner uniform
            - p_seq1_fixed: P(X >= x_obs | seq1 fixed)
            - ex_seq2_fixed: E[X] with seq2 fixed, partner uniform
            - p_seq2_fixed: P(X >= x_obs | seq2 fixed)

    Raises:
        ValueError: If sequences have different lengths.
    """

    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    # Get observed anchors
    _total, per_pos = score_aligned(
        s1,
        s2,
        mj,
        gap_char=gap_char,
        gap_penalty=None,
        unknown_policy=unknown_policy,
        context_bonus=context_bonus,
    )
    anchors = [
        i for i, v in enumerate(per_pos, start=1) if v is not None and float(v) <= thr
    ]
    x_obs = len(anchors)

    # Per-position complement probabilities for each sequence
    p1 = per_position_complement_prob_uniform(
        s1, mj, thr=thr, gap_char=gap_char, unknown_policy=unknown_policy
    )
    p2 = per_position_complement_prob_uniform(
        s2, mj, thr=thr, gap_char=gap_char, unknown_policy=unknown_policy
    )

    # Identify eligible positions (scored in per_pos)
    elig = [i for i, v in enumerate(per_pos, start=1) if v is not None]

    # Filter to positions with valid probabilities
    elig2 = [i for i in elig if p1[i - 1] is not None and p2[i - 1] is not None]
    ps1 = [float(p1[i - 1]) for i in elig2]  # type: ignore
    ps2 = [float(p2[i - 1]) for i in elig2]  # type: ignore

    # Recompute observed anchors if eligibility tightened
    if len(elig2) != len(elig):
        anchors_set = set(anchors)
        x_obs = sum(1 for i in elig2 if i in anchors_set)

    # Compute expected values and tail probabilities
    ex1 = float(sum(ps1))
    ex2 = float(sum(ps2))
    p_seq1_fixed = poisson_binomial_p_ge(ps1, x_obs)
    p_seq2_fixed = poisson_binomial_p_ge(ps2, x_obs)

    return x_obs, ex1, p_seq1_fixed, ex2, p_seq2_fixed
