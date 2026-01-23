"""Dynamic programming for gapped alignment extension (Phase 2).

This module implements the second phase of alignment: extending ungapped
seeds with gaps using dynamic programming. Key features:

- Affine gap penalties (separate open/extend costs)
- Band-constrained DP for efficiency
- Support for proline run constraints
- Bi-directional extension from seed positions

The Stage 2 algorithm takes a fixed seed window and extends it in both
directions, allowing gaps while respecting various constraints.
"""

import random
from typing import Optional

from .scoring import _mj_pair_score, score_aligned_with_gaps


def proline_run_mask(seq: str) -> list[bool]:
    """Create a mask for positions in proline runs of length >= 2.

    Identifies consecutive proline (P) residues that form runs,
    which may have special structural significance.

    Args:
        seq: Protein sequence.

    Returns:
        List of booleans where True indicates position is part
        of a PP+ run.

    Example:
        >>> proline_run_mask("ACPPDEF")
        [False, False, True, True, False, False, False]
    """
    s = seq.strip().upper()
    mask = [False] * len(s)
    i = 0
    while i < len(s):
        if s[i] != "P":
            i += 1
            continue
        j = i + 1
        while j < len(s) and s[j] == "P":
            j += 1
        if j - i >= 2:  # Run of length >= 2
            for k in range(i, j):
                mask[k] = True
        i = j
    return mask


def proline_run_ids(seq: str) -> list[int]:
    """Assign run IDs to positions in proline runs.

    Each proline run (length >= 2) gets a unique ID starting from 0.
    Positions not in proline runs get -1.

    Args:
        seq: Protein sequence.

    Returns:
        List of run IDs (-1 for non-run positions).

    Example:
        >>> proline_run_ids("APPBPPC")
        [-1, 0, 0, -1, 1, 1, -1]
    """
    s = seq.strip().upper()
    ids = [-1] * len(s)
    i = 0
    run_id = 0
    while i < len(s):
        if s[i] != "P":
            i += 1
            continue
        j = i + 1
        while j < len(s) and s[j] == "P":
            j += 1
        if j - i >= 2:
            for k in range(i, j):
                ids[k] = run_id
            run_id += 1
        i = j
    return ids


def alignment_gaps_cover_all_proline_runs_seq2(
    aln1: str,
    aln2: str,
    *,
    gap_char: str = "-",
) -> tuple[bool, int]:
    """Check if alignment gaps cover all proline runs in seq2.

    For biological reasons, we may require that every proline run
    in seq2 has at least one position covered by a gap in seq1.

    Args:
        aln1: Aligned sequence 1.
        aln2: Aligned sequence 2.
        gap_char: Gap character.

    Returns:
        Tuple of (all_covered, gap_count) where:
            - all_covered: True if every PP+ run has at least one gap
            - gap_count: Number of gaps that hit proline runs
    """
    s1 = aln1.strip().upper()
    s2 = aln2.strip().upper()
    "".join(c for c in s1 if c != gap_char)
    s2_nogap = "".join(c for c in s2 if c != gap_char)

    ids2 = proline_run_ids(s2_nogap)
    need2 = set(i for i in ids2 if i >= 0)  # Run IDs that need coverage

    i1 = 0  # Position in ungapped seq1
    i2 = 0  # Position in ungapped seq2
    gaps = 0

    for a, b in zip(s1, s2):
        if a == gap_char and b == gap_char:
            continue
        if a == gap_char:
            # Gap in seq1 - check if it covers a proline run in seq2
            if i2 < len(ids2) and ids2[i2] >= 0:
                need2.discard(ids2[i2])
                gaps += 1
            i2 += 1
            continue
        if b == gap_char:
            i1 += 1
            continue
        i1 += 1
        i2 += 1

    if need2:
        return False, gaps
    return True, gaps


def force_gap_per_proline_run_seq2(
    aln1: str,
    aln2: str,
    *,
    gap_char: str = "-",
) -> tuple[str, str, bool]:
    """Insert gaps to ensure every proline run in seq2 has coverage.

    Modifies the alignment to insert gaps in aln1 at the first position
    of each proline run that doesn't already have a gap.

    Args:
        aln1: Aligned sequence 1.
        aln2: Aligned sequence 2.
        gap_char: Gap character.

    Returns:
        Tuple of (modified_aln1, modified_aln2, was_changed).
    """
    s1 = aln1.strip().upper()
    s2 = aln2.strip().upper()
    s2_nogap = "".join(c for c in s2 if c != gap_char)

    ids2 = proline_run_ids(s2_nogap)
    if not ids2:
        return s1, s2, False

    # Find alignment positions for each proline run
    run_positions: dict[int, list[int]] = {}
    i2 = 0
    for idx, b in enumerate(s2):
        if b == gap_char:
            continue
        if i2 < len(ids2) and ids2[i2] >= 0:
            run_positions.setdefault(ids2[i2], []).append(idx)
        i2 += 1

    # Identify runs that need gaps inserted
    insert_positions: list[int] = []
    for run_id, positions in run_positions.items():
        if not positions:
            continue
        # Check if any position already has a gap in aln1
        if any(s1[pos] == gap_char for pos in positions):
            continue
        # Need to insert gap at first position
        insert_positions.append(positions[0])

    if not insert_positions:
        return s1, s2, False

    # Insert gaps
    s1_list = list(s1)
    s2_list = list(s2)
    for pos in sorted(insert_positions, reverse=True):
        s1_list.insert(pos, gap_char)
        s2_list.insert(pos, s2_list[pos])

    return "".join(s1_list), "".join(s2_list), True


def _stage2_dp_tables(
    s1: str,
    s2: str,
    mj: dict[tuple[str, str], float],
    *,
    anchor_i: int,
    anchor_j: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_in_seq1_only: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str,
    enforce_end_delta_zero: bool = False,
) -> tuple[
    list[list[Optional[float]]],
    list[list[Optional[tuple[int, int, int, str, int]]]],
    list[dict[tuple[int, int, int, str, int], tuple[tuple[int, int, int, str, int], str]]],
]:
    """Build DP tables for gapped alignment extension.

    Uses a state-based DP with tracking of:
    - Position in both sequences
    - Number of gaps used
    - Current state (Match, Gap1, Gap2, Start)
    - Current gap length

    Args:
        s1: First sequence (reversed for left extension).
        s2: Second sequence.
        mj: MJ matrix dictionary.
        anchor_i: Starting position in s1.
        anchor_j: Starting position in s2.
        max_len: Maximum extension length.
        band: Band width for diagonal constraint.
        switch_pen: Penalty for changing alignment trajectory.
        gap_open: Gap opening penalty.
        gap_ext: Gap extension penalty.
        gap_in_seq1_only: Only allow gaps in seq1 (for proline constraints).
        max_gaps: Maximum number of gap openings.
        max_gap_len: Maximum length of any single gap.
        unknown_policy: How to handle unknown residues.
        enforce_end_delta_zero: Require final positions to be on diagonal.

    Returns:
        Tuple of (best_scores, best_keys, backpointers) where:
            - best_scores[t][g] = best score at length t with g gaps
            - best_keys[t][g] = DP key achieving that score
            - backpointers[t] = dict mapping keys to (prev_key, move)
    """
    n1 = len(s1)
    n2 = len(s2)

    # best_scores[t][g] = best score achievable at extension length t with g gaps
    best_scores: list[list[Optional[float]]] = [
        [None] * (max_gaps + 1) for _ in range(max_len + 1)
    ]
    # best_keys[t][g] = DP key (i, j, g, state, gap_len) for best score
    best_keys: list[list[Optional[tuple[int, int, int, str, int]]]] = [
        [None] * (max_gaps + 1) for _ in range(max_len + 1)
    ]
    # Type alias for DP key: (i, j, gap_count, state, gap_len)
    DPKey = tuple[int, int, int, str, int]
    # back[t] = dict mapping keys to (previous_key, move_type)
    back: list[dict[DPKey, tuple[DPKey, str]]] = []

    # DP tables: dp[t] = dict of {key: score}
    dp: list[dict[DPKey, float]] = []

    # Initialize: start at anchor with score 0
    dp0: dict[DPKey, float] = {}
    back0: dict[DPKey, tuple[DPKey, str]] = {}
    dp0[(anchor_i, anchor_j, 0, "S", 0)] = 0.0
    dp.append(dp0)
    back.append(back0)

    # Fill DP table
    for t in range(1, max_len + 1):
        cur: dict[DPKey, float] = {}
        cur_back: dict[DPKey, tuple[DPKey, str]] = {}
        prev = dp[t - 1]

        for (i, j, g, state, gl), prev_score in prev.items():
            delta_prev = j - i

            # MATCH: advance both sequences
            if i < n1 and j < n2:
                delta_cur = (j + 1) - (i + 1)
                if abs(delta_cur) <= band:
                    val = _mj_pair_score(s1[i], s2[j], mj, unknown_policy)
                    if val is not None:
                        score = prev_score + val
                        if state != "S" and delta_cur != delta_prev:
                            score += switch_pen
                        key = (i + 1, j + 1, g, "M", 0)
                        if key not in cur or score < cur[key]:
                            cur[key] = score
                            cur_back[key] = ((i, j, g, state, gl), "M")

            # GAP IN SEQ1: advance j only
            if j < n2:
                delta_cur = (j + 1) - i
                if abs(delta_cur) <= band:
                    if state == "G1":
                        # Extend existing gap
                        if gl < max_gap_len:
                            score = prev_score + gap_ext
                            if delta_cur != delta_prev:
                                score += switch_pen
                            key = (i, j + 1, g, "G1", gl + 1)
                            if key not in cur or score < cur[key]:
                                cur[key] = score
                                cur_back[key] = ((i, j, g, state, gl), "G1")
                    else:
                        # Open new gap
                        if g < max_gaps:
                            score = prev_score + gap_open
                            if state != "S" and delta_cur != delta_prev:
                                score += switch_pen
                            key = (i, j + 1, g + 1, "G1", 1)
                            if key not in cur or score < cur[key]:
                                cur[key] = score
                                cur_back[key] = ((i, j, g, state, gl), "G1")

            # GAP IN SEQ2: advance i only (if allowed)
            if i < n1 and not gap_in_seq1_only:
                delta_cur = j - (i + 1)
                if abs(delta_cur) <= band:
                    if state == "G2":
                        # Extend existing gap
                        if gl < max_gap_len:
                            score = prev_score + gap_ext
                            if delta_cur != delta_prev:
                                score += switch_pen
                            key = (i + 1, j, g, "G2", gl + 1)
                            if key not in cur or score < cur[key]:
                                cur[key] = score
                                cur_back[key] = ((i, j, g, state, gl), "G2")
                    else:
                        # Open new gap
                        if g < max_gaps:
                            score = prev_score + gap_open
                            if state != "S" and delta_cur != delta_prev:
                                score += switch_pen
                            key = (i + 1, j, g + 1, "G2", 1)
                            if key not in cur or score < cur[key]:
                                cur[key] = score
                                cur_back[key] = ((i, j, g, state, gl), "G2")

        dp.append(cur)
        back.append(cur_back)

        # Track best scores at this length
        for key, score in cur.items():
            _, _, g, _, _ = key
            if g > max_gaps:
                continue
            if enforce_end_delta_zero:
                i, j, _, _, _ = key
                if (j - i) != 0:
                    continue
            if best_scores[t][g] is None or score < best_scores[t][g]:  # type: ignore
                best_scores[t][g] = score
                best_keys[t][g] = key

    return best_scores, best_keys, back


def _stage2_backtrace(
    s1: str,
    s2: str,
    back: list[dict[tuple[int, int, int, str, int], tuple[tuple[int, int, int, str, int], str]]],
    key: tuple[int, int, int, str, int],
    t: int,
) -> tuple[str, str, tuple[int, int]]:
    """Reconstruct alignment from DP backpointers.

    Args:
        s1: First sequence.
        s2: Second sequence.
        back: Backpointer tables from _stage2_dp_tables.
        key: Final DP key.
        t: Final extension length.

    Returns:
        Tuple of (aln1, aln2, start_positions) where start_positions
        is (start_i, start_j) in the original sequences.
    """
    aln1 = []
    aln2 = []
    cur_key = key

    while t > 0:
        prev_key, move = back[t].get(cur_key, ((0, 0, 0, "S", 0), "S"))
        i, j, _, _, _ = cur_key

        if move == "M":
            aln1.append(s1[i - 1])
            aln2.append(s2[j - 1])
        elif move == "G1":
            aln1.append("-")
            aln2.append(s2[j - 1])
        elif move == "G2":
            aln1.append(s1[i - 1])
            aln2.append("-")
        else:
            break

        cur_key = prev_key
        t -= 1

    # Reverse since we built backwards
    aln1_str = "".join(reversed(aln1))
    aln2_str = "".join(reversed(aln2))
    start_i, start_j, _, _, _ = cur_key

    return aln1_str, aln2_str, (start_i, start_j)


def stage2_extend_fixed_core(
    s1: str,
    s2: str,
    mj: dict[tuple[str, str], float],
    *,
    anchor_i: int,
    anchor_j: int,
    seed_len: int,
    min_len: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_proline_force_runs_seq2: bool = False,
    gap_proline_force_post: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> tuple[Optional[float], Optional[str], Optional[str], Optional[tuple[int, int]]]:
    """Extend a fixed seed using bidirectional DP.

    Takes a seed window and extends it in both directions using
    gapped dynamic programming, then combines the results.

    Args:
        s1: First sequence.
        s2: Second sequence.
        mj: MJ matrix dictionary.
        anchor_i: Start of seed in s1 (0-based).
        anchor_j: Start of seed in s2 (0-based).
        seed_len: Length of fixed seed.
        min_len: Minimum total alignment length.
        max_len: Maximum total alignment length.
        band: Band width for DP.
        switch_pen: Penalty for trajectory changes.
        gap_open: Gap opening penalty.
        gap_ext: Gap extension penalty.
        gap_proline_force_runs_seq2: Require gaps at proline runs.
        gap_proline_force_post: Insert gaps post-hoc for proline runs.
        max_gaps: Maximum number of gap openings.
        max_gap_len: Maximum single gap length.
        unknown_policy: How to handle unknown residues.
        context_bonus: Whether to apply context bonuses.

    Returns:
        Tuple of (score, aln1, aln2, start_positions) or all None if failed.
    """
    if min_len <= 0 or max_len <= 0 or max_len < min_len:
        raise ValueError("Invalid length range for stage2 DP")

    n1 = len(s1)
    n2 = len(s2)
    if n1 == 0 or n2 == 0:
        return None, None, None, None
    if anchor_i < 0 or anchor_j < 0 or anchor_i > n1 or anchor_j > n2:
        return None, None, None, None
    if anchor_i + seed_len > n1 or anchor_j + seed_len > n2:
        return None, None, None, None
    if (gap_proline_force_runs_seq2 or gap_proline_force_post) and max_gaps <= 0:
        return None, None, None, None

    # Score the fixed core
    core_score = 0.0
    for k in range(seed_len):
        val = _mj_pair_score(
            s1[anchor_i + k], s2[anchor_j + k], mj, unknown_policy
        )
        if val is None:
            return None, None, None, None
        core_score += float(val)

    # Prepare sequences for left and right extension
    left1 = s1[:anchor_i][::-1]  # Reverse for left extension
    left2 = s2[:anchor_j][::-1]
    right1 = s1[anchor_i + seed_len:]
    right2 = s2[anchor_j + seed_len:]

    # Right DP (extension from end of seed)
    r_scores, r_keys, r_back = _stage2_dp_tables(
        right1, right2, mj,
        anchor_i=0,
        anchor_j=0,
        max_len=max_len,
        band=band,
        switch_pen=switch_pen,
        gap_open=gap_open,
        gap_ext=gap_ext,
        gap_in_seq1_only=gap_proline_force_runs_seq2,
        max_gaps=max_gaps,
        max_gap_len=max_gap_len,
        unknown_policy=unknown_policy,
    )

    # Left DP (extension from start of seed, reversed)
    l_scores, l_keys, l_back = _stage2_dp_tables(
        left1, left2, mj,
        anchor_i=0,
        anchor_j=0,
        max_len=max_len,
        band=band,
        switch_pen=switch_pen,
        gap_open=gap_open,
        gap_ext=gap_ext,
        gap_in_seq1_only=gap_proline_force_runs_seq2,
        max_gaps=max_gaps,
        max_gap_len=max_gap_len,
        unknown_policy=unknown_policy,
        enforce_end_delta_zero=True,
    )

    # Include zero-length left extension
    l_scores[0][0] = 0.0
    l_keys[0][0] = (0, 0, 0, "S", 0)

    # Find best combination of left and right extensions
    best_score: Optional[float] = None
    best_pair: Optional[tuple[int, int, int, int]] = None

    for t_left in range(0, max_len + 1):
        for g_left in range(0, max_gaps + 1):
            if l_scores[t_left][g_left] is None:
                continue
            if l_keys[t_left][g_left] is None:
                continue

            for t_right in range(0, max_len + 1):
                for g_right in range(0, max_gaps - g_left + 1):
                    if r_scores[t_right][g_right] is None:
                        continue
                    if r_keys[t_right][g_right] is None:
                        continue

                    total_len = seed_len + t_left + t_right
                    if total_len < min_len or total_len > max_len:
                        continue

                    score = (
                        core_score
                        + float(l_scores[t_left][g_left])  # type: ignore
                        + float(r_scores[t_right][g_right])  # type: ignore
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (t_left, g_left, t_right, g_right)

    if best_score is None or best_pair is None:
        return None, None, None, None

    # Reconstruct alignment
    t_left, g_left, t_right, g_right = best_pair

    aln1_r, aln2_r, start_r = _stage2_backtrace(
        right1, right2, r_back, r_keys[t_right][g_right], t_right  # type: ignore
    )

    if t_left == 0:
        aln1_l = ""
        aln2_l = ""
        start_i = anchor_i
        start_j = anchor_j
    else:
        aln1_l_rev, aln2_l_rev, start_l_rev = _stage2_backtrace(
            left1, left2, l_back, l_keys[t_left][g_left], t_left  # type: ignore
        )
        aln1_l = aln1_l_rev[::-1]
        aln2_l = aln2_l_rev[::-1]
        start_i = anchor_i - start_l_rev[0]
        start_j = anchor_j - start_l_rev[1]

    # Combine alignment pieces
    core_aln1 = s1[anchor_i : anchor_i + seed_len]
    core_aln2 = s2[anchor_j : anchor_j + seed_len]
    aln1 = aln1_l + core_aln1 + aln1_r
    aln2 = aln2_l + core_aln2 + aln2_r

    # Check/enforce proline constraints
    if gap_proline_force_runs_seq2:
        ok, gaps = alignment_gaps_cover_all_proline_runs_seq2(aln1, aln2)
        if not ok or gaps == 0:
            return None, None, None, None

    if gap_proline_force_post:
        forced1, forced2, changed = force_gap_per_proline_run_seq2(aln1, aln2)
        if not changed:
            return None, None, None, None
        forced_score = score_aligned_with_gaps(
            forced1, forced2, mj,
            gap_open=gap_open,
            gap_ext=gap_ext,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        return forced_score, forced1, forced2, (start_i, start_j)

    if context_bonus:
        best_score = score_aligned_with_gaps(
            aln1, aln2, mj,
            gap_open=gap_open,
            gap_ext=gap_ext,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )

    return best_score, aln1, aln2, (start_i, start_j)


def stage2_best_from_seed(
    seq1: str,
    seq2: str,
    mj: dict[tuple[str, str], float],
    *,
    seed_i: int,
    seed_j: int,
    seed_len: int,
    flank: int,
    min_len: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_proline_force_runs_seq2: bool = False,
    gap_proline_force_post: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str = "error",
    reanchor: bool = False,
    context_bonus: bool = False,
) -> tuple[
    Optional[float],
    Optional[str],
    Optional[str],
    Optional[tuple[int, int]],
    Optional[tuple[int, int]],
]:
    """Find best alignment extension from a seed position.

    Extracts a flanked region around the seed and runs Stage 2 DP.
    Can optionally try all possible anchor positions within the
    flanked region.

    Args:
        seq1: First full sequence.
        seq2: Second full sequence.
        mj: MJ matrix dictionary.
        seed_i: Seed start in seq1.
        seed_j: Seed start in seq2.
        seed_len: Seed length.
        flank: Extra positions to include on each side.
        min_len: Minimum alignment length.
        max_len: Maximum alignment length.
        band: DP band width.
        switch_pen: Trajectory change penalty.
        gap_open: Gap open penalty.
        gap_ext: Gap extend penalty.
        gap_proline_force_runs_seq2: Require gaps at proline runs.
        gap_proline_force_post: Insert gaps post-hoc.
        max_gaps: Maximum gap openings.
        max_gap_len: Maximum gap length.
        unknown_policy: How to handle unknown residues.
        reanchor: If True, try all anchor positions in flanked region.
        context_bonus: Whether to apply context bonuses.

    Returns:
        Tuple of (score, aln1, aln2, start_positions, anchor_positions)
        or all None if failed.
    """
    # Extract flanked subsequences
    s1_start = max(0, seed_i - flank)
    s1_end = min(len(seq1), seed_i + seed_len + flank)
    s2_start = max(0, seed_j - flank)
    s2_end = min(len(seq2), seed_j + seed_len + flank)

    sub1 = seq1[s1_start:s1_end]
    sub2 = seq2[s2_start:s2_end]

    if not reanchor:
        # Use original anchor position
        score, aln1, aln2, start = stage2_extend_fixed_core(
            sub1, sub2, mj,
            anchor_i=seed_i - s1_start,
            anchor_j=seed_j - s2_start,
            seed_len=seed_len,
            min_len=min_len,
            max_len=max_len,
            band=band,
            switch_pen=switch_pen,
            gap_open=gap_open,
            gap_ext=gap_ext,
            gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
            gap_proline_force_post=gap_proline_force_post,
            max_gaps=max_gaps,
            max_gap_len=max_gap_len,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        if score is None or start is None:
            return None, None, None, None, None

        start_i, start_j = start
        return (
            score,
            aln1,
            aln2,
            (s1_start + start_i, s2_start + start_j),
            (seed_i, seed_j),
        )

    # Try all anchor positions
    best_score: Optional[float] = None
    best: Optional[tuple[str, str, tuple[int, int]]] = None
    best_anchor: Optional[tuple[int, int]] = None

    max_i = len(sub1) - seed_len
    max_j = len(sub2) - seed_len
    if max_i < 0 or max_j < 0:
        return None, None, None, None, None

    for ai in range(0, max_i + 1):
        for aj in range(0, max_j + 1):
            score, aln1, aln2, start = stage2_extend_fixed_core(
                sub1, sub2, mj,
                anchor_i=ai,
                anchor_j=aj,
                seed_len=seed_len,
                min_len=min_len,
                max_len=max_len,
                band=band,
                switch_pen=switch_pen,
                gap_open=gap_open,
                gap_ext=gap_ext,
                gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
                gap_proline_force_post=gap_proline_force_post,
                max_gaps=max_gaps,
                max_gap_len=max_gap_len,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )
            if score is None or start is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best = (aln1, aln2, start)  # type: ignore
                best_anchor = (ai, aj)

    if best_score is None or best is None or best_anchor is None:
        return None, None, None, None, None

    aln1, aln2, start = best
    start_i, start_j = start
    anchor_i, anchor_j = best_anchor
    return (
        best_score,
        aln1,
        aln2,
        (s1_start + start_i, s2_start + start_j),
        (s1_start + anchor_i, s2_start + anchor_j),
    )


def stage2_null_scores(
    seq1: str,
    seq2: str,
    mj: dict[tuple[str, str], float],
    *,
    seed_i: int,
    seed_j: int,
    seed_len: int,
    flank: int,
    min_len: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_proline_force_runs_seq2: bool = False,
    gap_proline_force_post: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str,
    n: int,
    seed: Optional[int] = None,
    reanchor: bool = False,
    context_bonus: bool = False,
) -> list[float]:
    """Generate null distribution for Stage 2 alignment scores.

    Shuffles both subsequences and runs Stage 2 DP to build an
    empirical null distribution.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        mj: MJ matrix dictionary.
        seed_i: Seed position in seq1.
        seed_j: Seed position in seq2.
        seed_len: Seed length.
        flank: Flank size.
        min_len: Minimum alignment length.
        max_len: Maximum alignment length.
        band: DP band width.
        switch_pen: Trajectory penalty.
        gap_open: Gap open penalty.
        gap_ext: Gap extend penalty.
        gap_proline_force_runs_seq2: Require gaps at proline runs.
        gap_proline_force_post: Insert gaps post-hoc.
        max_gaps: Maximum gaps.
        max_gap_len: Maximum gap length.
        unknown_policy: How to handle unknown residues.
        n: Number of null samples.
        seed: Random seed.
        reanchor: Try all anchor positions.
        context_bonus: Apply context bonuses.

    Returns:
        List of scores from null alignments.
    """
    rng = random.Random(seed)

    s1_start = max(0, seed_i - flank)
    s1_end = min(len(seq1), seed_i + seed_len + flank)
    s2_start = max(0, seed_j - flank)
    s2_end = min(len(seq2), seed_j + seed_len + flank)

    base2 = list(seq2[s2_start:s2_end])
    base1 = list(seq1[s1_start:s1_end])

    scores: list[float] = []
    for _ in range(n):
        rng.shuffle(base2)
        rng.shuffle(base1)
        sub1 = "".join(base1)
        sub2 = "".join(base2)

        if reanchor:
            max_i = len(sub1) - seed_len
            max_j = len(sub2) - seed_len
            best_score: Optional[float] = None
            if max_i >= 0 and max_j >= 0:
                for ai in range(0, max_i + 1):
                    for aj in range(0, max_j + 1):
                        score, _, _, _ = stage2_extend_fixed_core(
                            sub1, sub2, mj,
                            anchor_i=ai,
                            anchor_j=aj,
                            seed_len=seed_len,
                            min_len=min_len,
                            max_len=max_len,
                            band=band,
                            switch_pen=switch_pen,
                            gap_open=gap_open,
                            gap_ext=gap_ext,
                            gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
                            gap_proline_force_post=gap_proline_force_post,
                            max_gaps=max_gaps,
                            max_gap_len=max_gap_len,
                            unknown_policy=unknown_policy,
                            context_bonus=context_bonus,
                        )
                        if score is None:
                            continue
                        if best_score is None or score < best_score:
                            best_score = score
            score = best_score
        else:
            score, _, _, _ = stage2_extend_fixed_core(
                sub1, sub2, mj,
                anchor_i=seed_i - s1_start,
                anchor_j=seed_j - s2_start,
                seed_len=seed_len,
                min_len=min_len,
                max_len=max_len,
                band=band,
                switch_pen=switch_pen,
                gap_open=gap_open,
                gap_ext=gap_ext,
                gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
                gap_proline_force_post=gap_proline_force_post,
                max_gaps=max_gaps,
                max_gap_len=max_gap_len,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )

        if score is not None:
            scores.append(score)

    return scores
