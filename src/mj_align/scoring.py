"""MJ matrix handling and alignment scoring functions.

This module provides the core scoring functionality for protein sequence
alignments using Miyazawa-Jernigan (MJ) contact potentials. It includes:

- MJ matrix loading from CSV files
- Alignment scoring with gap handling
- Context bonus calculations for neighboring residue interactions
- Anchor detection and scoring based on MJ thresholds

The MJ matrix represents statistical contact potentials between amino acid
residue types derived from known protein structures.
"""

import csv
from collections.abc import Iterable
from typing import Callable, Optional

from .amino_acid_properties import (
    AA20,
    AROMATICS,
    HYDROPHOBE_OFFSET_WEIGHT,
    HYDROPHOBES,
    NEG_CHARGES,
    POS_CHARGES,
    apply_mj_overrides,
)


def load_mj_csv(path: str) -> dict[tuple[str, str], float]:
    """Load an MJ matrix from a CSV file.

    Expected CSV format:
        - First row: header with blank cell followed by amino acid codes
        - First column: amino acid codes
        - Body: numeric MJ values

    Args:
        path: Path to the MJ matrix CSV file.

    Returns:
        Dictionary mapping (amino_acid_1, amino_acid_2) tuples to
        their MJ interaction scores.

    Raises:
        ValueError: If no valid MJ values are found in the file.

    Example:
        >>> mj = load_mj_csv("mj_matrix.csv")
        >>> mj[("A", "V")]  # Get score for Ala-Val interaction
        -2.5
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [h.strip().upper() for h in header[1:]]  # Skip first empty cell
        mj: dict[tuple[str, str], float] = {}

        for row in reader:
            if not row or not row[0].strip():
                continue
            raa = row[0].strip().upper()  # Row amino acid
            for caa, v in zip(cols, row[1:]):
                v = v.strip()
                if v:
                    mj[(raa, caa)] = float(v)

    if not mj:
        raise ValueError(f"No MJ values loaded from {path!r}")
    return mj


def get_mj_scorer(
    mj: dict[tuple[str, str], float]
) -> Callable[[str, str], Optional[float]]:
    """Create a scorer function from an MJ matrix dictionary.

    Returns a callable that looks up pair scores with automatic
    handling of key order (tries both (a,b) and (b,a)) and applies
    MJ overrides for special residue pairs.

    Args:
        mj: MJ matrix as a dictionary mapping residue pairs to scores.

    Returns:
        A function that takes two amino acids and returns their
        interaction score, or raises KeyError if not found.

    Example:
        >>> mj = load_mj_csv("mj_matrix.csv")
        >>> scorer = get_mj_scorer(mj)
        >>> scorer("A", "V")
        -2.5
    """
    if hasattr(mj, "score") and callable(getattr(mj, "score")):
        # Handle objects with .score() method
        score_func = getattr(mj, "score")
        def _score(a: str, b: str) -> Optional[float]:
            return apply_mj_overrides(a, b, score_func(a, b))
        return _score

    if isinstance(mj, dict):
        def _score(a: str, b: str) -> Optional[float]:
            # Try direct key first
            if (a, b) in mj:
                return apply_mj_overrides(a, b, mj[(a, b)])
            # Fall back to reversed key
            if (b, a) in mj:
                return apply_mj_overrides(a, b, mj[(b, a)])
            raise KeyError((a, b))
        return _score

    raise TypeError(f"Unsupported MJ matrix type: {type(mj)}")


def _mj_pair_score(
    a: str,
    b: str,
    mj: dict[tuple[str, str], float],
    unknown_policy: str,
) -> Optional[float]:
    """Get MJ score for a residue pair with policy-based error handling.

    Internal helper function that looks up pair scores and handles
    unknown residue pairs according to the specified policy.

    Args:
        a: First amino acid.
        b: Second amino acid.
        mj: MJ matrix dictionary.
        unknown_policy: How to handle unknown pairs:
            - 'error': Raise ValueError
            - 'skip': Return None
            - 'zero': Return 0.0

    Returns:
        The MJ score, None if skipped, or 0.0 if using zero policy.

    Raises:
        ValueError: If pair not found and unknown_policy is 'error'.
    """
    val = mj.get((a, b), mj.get((b, a)))
    val = apply_mj_overrides(a, b, val)
    if val is None:
        if unknown_policy == "error":
            raise ValueError(f"Unknown pair {a},{b}")
        if unknown_policy == "skip":
            return None
        return 0.0  # unknown_policy == "zero"
    return float(val)


def score_aligned(
    seq1_aln: str,
    seq2_aln: str,
    mj: dict[tuple[str, str], float],
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> tuple[float, list[Optional[float]]]:
    """Score two aligned sequences position-by-position using MJ matrix.

    Computes total alignment score and per-position scores. Gap positions
    can be optionally penalized or ignored.

    Args:
        seq1_aln: First aligned sequence (may contain gap characters).
        seq2_aln: Second aligned sequence (must be same length as seq1_aln).
        mj: MJ matrix dictionary mapping residue pairs to scores.
        gap_char: Character representing gaps in alignment. Default '-'.
        gap_penalty: Penalty for gap positions. If None, gaps are ignored.
        unknown_policy: How to handle unknown residues:
            - 'error': Raise on unknown residues
            - 'skip': Skip position (per_pos=None)
            - 'zero': Treat as score 0
        context_bonus: Whether to apply context bonuses for neighboring
            residue interactions. Default False.

    Returns:
        Tuple of (total_score, per_position_scores) where per_position_scores
        is a list with float scores or None for skipped positions.

    Raises:
        ValueError: If sequences have different lengths or unknown residues
            encountered with 'error' policy.

    Example:
        >>> mj = load_mj_csv("mj_matrix.csv")
        >>> total, per_pos = score_aligned("ACDEF", "ACDEF", mj)
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    total = 0.0
    per_pos: list[Optional[float]] = []
    idx1 = 0
    idx2 = 0

    for i, (a, b) in enumerate(zip(s1, s2), start=1):
        # Handle gap positions
        if a == gap_char or b == gap_char:
            if gap_penalty is None:
                if a != gap_char:
                    idx1 += 1
                if b != gap_char:
                    idx2 += 1
                per_pos.append(None)
                continue
            total += gap_penalty
            per_pos.append(gap_penalty)
            if a != gap_char:
                idx1 += 1
            if b != gap_char:
                idx2 += 1
            continue

        # Score aligned residue pair
        val = mj.get((a, b), mj.get((b, a)))
        val = apply_mj_overrides(a, b, val)
        if val is None:
            if unknown_policy == "error":
                raise ValueError(f"Unknown pair {a},{b} at pos {i}")
            if unknown_policy == "skip":
                per_pos.append(None)
                continue
            val = 0.0  # unknown_policy == "zero"

        total += val
        per_pos.append(val)
        idx1 += 1
        idx2 += 1

    # Apply context bonuses if requested
    if context_bonus:
        bonuses = context_bonus_aligned(
            s1, s2, mj, gap_char=gap_char, unknown_policy=unknown_policy
        )
        for i, bonus in enumerate(bonuses):
            if bonus == 0.0:
                continue
            val = per_pos[i]
            if val is None:
                continue
            per_pos[i] = float(val) + bonus
            total += bonus

    return total, per_pos


def score_aligned_with_gaps(
    aln1: str,
    aln2: str,
    mj: dict[tuple[str, str], float],
    *,
    gap_open: float,
    gap_ext: float,
    gap_char: str = "-",
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> float:
    """Score aligned sequences using affine gap penalties (open + extend).

    Uses separate penalties for opening a new gap versus extending an
    existing gap, which better models biological gap formation.

    Args:
        aln1: First aligned sequence.
        aln2: Second aligned sequence.
        mj: MJ matrix dictionary.
        gap_open: Penalty for opening a new gap.
        gap_ext: Penalty for extending an existing gap.
        gap_char: Gap character. Default '-'.
        unknown_policy: How to handle unknown residues.
        context_bonus: Whether to apply context bonuses.

    Returns:
        Total alignment score including gap penalties.

    Raises:
        ValueError: If sequences have different lengths.
    """
    s1 = aln1.strip().upper()
    s2 = aln2.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    total = 0.0
    gap1 = 0  # Track if we're in a gap in seq1
    gap2 = 0  # Track if we're in a gap in seq2

    for i, (a, b) in enumerate(zip(s1, s2), start=1):
        if a == gap_char and b == gap_char:
            continue
        if a == gap_char:
            total += gap_ext if gap1 else gap_open
            gap1 = 1
            gap2 = 0
            continue
        if b == gap_char:
            total += gap_ext if gap2 else gap_open
            gap2 = 1
            gap1 = 0
            continue
        gap1 = 0
        gap2 = 0
        val = _mj_pair_score(a, b, mj, unknown_policy)
        if val is None:
            continue
        total += float(val)

    if context_bonus:
        bonuses = context_bonus_aligned(
            s1, s2, mj, gap_char=gap_char, unknown_policy=unknown_policy
        )
        total += sum(bonuses)
    return total


def context_bonus_aligned(
    seq1_aln: str,
    seq2_aln: str,
    mj: dict[tuple[str, str], float],
    *,
    gap_char: str = "-",
    unknown_policy: str = "error",
) -> list[float]:
    """Compute context bonuses for neighboring residue interactions.

    Applies additional scoring based on structural considerations for
    residue pairs in neighboring positions (±1):

    - Charge opposites (K/R vs D/E): +0.25 × MJ score
    - Proline with aromatic at ±1: +0.5 × MJ score
    - Proline with proline at ±1: +0.5 × MJ score
    - Hydrophobe with hydrophobe at ±1: +0.6 × MJ score

    Args:
        seq1_aln: First aligned sequence.
        seq2_aln: Second aligned sequence.
        mj: MJ matrix dictionary.
        gap_char: Gap character. Default '-'.
        unknown_policy: How to handle unknown residues.

    Returns:
        List of bonus values for each alignment position.

    Raises:
        ValueError: If sequences have different lengths.
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")
    bonuses = [0.0] * len(s1)

    def pair_score(a: str, b: str) -> Optional[float]:
        """Helper to get pair score with gap handling."""
        if a == gap_char or b == gap_char:
            return None
        if a not in AA20 or b not in AA20:
            if unknown_policy == "error":
                raise ValueError(f"Unknown pair {a},{b}")
            if unknown_policy == "skip":
                return None
            return 0.0
        return _mj_pair_score(a, b, mj, unknown_policy)

    n = len(s1)
    for i in range(n):
        a = s1[i]
        b = s2[i]
        if a == gap_char or b == gap_char:
            continue
        if a not in AA20 or b not in AA20:
            continue

        # Charge opposites within ±1 (half weight per neighbor)
        if a in POS_CHARGES or a in NEG_CHARGES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                b2 = s2[j]
                if (a in POS_CHARGES and b2 in NEG_CHARGES) or (
                    a in NEG_CHARGES and b2 in POS_CHARGES
                ):
                    val = pair_score(a, b2)
                    if val is not None:
                        bonuses[i] += 0.25 * float(val)

        if b in POS_CHARGES or b in NEG_CHARGES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                a2 = s1[j]
                if (b in POS_CHARGES and a2 in NEG_CHARGES) or (
                    b in NEG_CHARGES and a2 in POS_CHARGES
                ):
                    val = pair_score(b, a2)
                    if val is not None:
                        bonuses[i] += 0.25 * float(val)

        # Proline aligned to opposing ±1 aromatic (full weight)
        if a == "P":
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                b2 = s2[j]
                if b2 in AROMATICS:
                    val = pair_score("P", b2)
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)
                if b2 == "P":
                    val = pair_score("P", "P")
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)
        if b == "P":
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                a2 = s1[j]
                if a2 in AROMATICS:
                    val = pair_score("P", a2)
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)
                if a2 == "P":
                    val = pair_score("P", "P")
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)

        # Hydrophobe aligned to opposing ±1 hydrophobe
        if a in HYDROPHOBES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                b2 = s2[j]
                if b2 in HYDROPHOBES:
                    val = pair_score(a, b2)
                    if val is not None:
                        bonuses[i] += HYDROPHOBE_OFFSET_WEIGHT * float(val)
        if b in HYDROPHOBES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                a2 = s1[j]
                if a2 in HYDROPHOBES:
                    val = pair_score(b, a2)
                    if val is not None:
                        bonuses[i] += HYDROPHOBE_OFFSET_WEIGHT * float(val)

    return bonuses


def anchors_by_threshold(
    per_pos: list[Optional[float]],
    thr: float = -25.0,
) -> list[int]:
    """Find anchor positions where MJ score indicates strong complement.

    Anchors are positions with MJ scores at or below the threshold,
    indicating favorable (complementary) interactions.

    Args:
        per_pos: Per-position scores from score_aligned().
        thr: Score threshold for anchor detection. More negative values
            indicate stronger interactions. Default -25.0.

    Returns:
        List of 1-based position indices that qualify as anchors.

    Example:
        >>> per_pos = [-30.0, -10.0, None, -28.0, -5.0]
        >>> anchors_by_threshold(per_pos, thr=-25.0)
        [1, 4]  # Positions 1 and 4 have scores <= -25
    """
    return [i for i, v in enumerate(per_pos, start=1) if v is not None and v <= thr]


def anchor_score(
    per_pos: list[Optional[float]],
    anchors: Iterable[int],
) -> tuple[float, list[tuple[int, float]]]:
    """Sum MJ scores over specified anchor positions.

    Args:
        per_pos: Per-position scores from score_aligned().
        anchors: Iterable of 1-based position indices.

    Returns:
        Tuple of (total_anchor_score, list_of_used_positions) where
        list_of_used_positions contains (position, score) pairs.

    Example:
        >>> per_pos = [-30.0, -10.0, None, -28.0]
        >>> anchor_score(per_pos, [1, 4])
        (-58.0, [(1, -30.0), (4, -28.0)])
    """
    total = 0.0
    used: list[tuple[int, float]] = []
    for i in anchors:
        if i < 1 or i > len(per_pos):
            continue
        v = per_pos[i - 1]
        if v is None:
            continue
        total += float(v)
        used.append((i, float(v)))
    return total, used


def top_contributors(
    seq1_aln: str,
    seq2_aln: str,
    per_pos: list[Optional[float]],
    n: int = 10,
) -> list[tuple[float, int, str, str]]:
    """Find the n most favorable (most negative) aligned positions.

    Args:
        seq1_aln: First aligned sequence.
        seq2_aln: Second aligned sequence.
        per_pos: Per-position scores from score_aligned().
        n: Number of top contributors to return. Default 10.

    Returns:
        List of (score, 1-based_position, residue1, residue2) tuples,
        sorted by score (most negative first).

    Example:
        >>> top_contributors("AKW", "VEF", [-5, -30, -25], n=2)
        [(-30.0, 2, 'K', 'E'), (-25.0, 3, 'W', 'F')]
    """
    s1 = seq1_aln
    s2 = seq2_aln
    hits: list[tuple[float, int, str, str]] = []
    for i, v in enumerate(per_pos, start=1):
        if v is None:
            continue
        hits.append((float(v), i, s1[i - 1], s2[i - 1]))

    hits.sort(key=lambda x: x[0])  # Most negative first
    return hits[: max(0, n)]
