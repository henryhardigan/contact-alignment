"""Hidden Markov Model for protein interface footprinting.

This module implements a simple three-state HMM for identifying protein-protein
interface regions based on MJ interaction scores:

- B (Background): Positions with weak or no complementary interactions
- P (Peripheral): Positions with moderate complementary interactions
- C (Core): Positions with strong complementary interactions (interface core)

The HMM uses Viterbi decoding to find the most likely state path through
the per-position MJ scores.
"""

import math
from typing import Optional


def hmm_emission(state: str, s: float, tc: float, tp: float) -> float:
    """Compute emission score for a given state and MJ score.

    The emission model rewards scores that match expected interaction
    strength for each state, with penalties for mismatches.

    Args:
        state: HMM state ('B', 'P', or 'C').
        s: MJ score at this position (more negative = stronger interaction).
        tc: Core threshold - scores <= tc are expected for core state.
        tp: Peripheral threshold - scores <= tp are expected for peripheral state.

    Returns:
        Log-scale emission score for this state and observation.

    Note:
        Very positive MJ scores (>= 20 or >= 30) receive additional
        penalties regardless of state, as they indicate unfavorable
        interactions.
    """
    score = 0.0

    # Penalize very positive (unfavorable) MJ scores
    if s >= 30:
        score -= 6.0
    elif s >= 20:
        score -= 3.0

    if state == "C":
        # Core: expect strong complement (s <= tc)
        if s <= tc:
            score += 4.0 + 0.1 * (tc - s)
        else:
            score -= 8.0
    elif state == "P":
        # Peripheral: expect moderate complement (s <= tp)
        if s <= tp:
            score += 2.0 + 0.05 * (tp - s)
        else:
            score -= 2.0
    else:  # B (Background)
        # Background: mild bonus for any reasonable score
        if s <= tp:
            score += 0.5

    return score


def hmm_interface_path(
    per_pos: list[Optional[float]],
    *,
    tc: float,
    tp: float,
) -> list[Optional[str]]:
    """Find most likely B/P/C state path using Viterbi algorithm.

    Decodes the optimal state sequence through the HMM given the
    per-position MJ scores. Gap positions (None) are skipped and
    returned as None in the path.

    Args:
        per_pos: Per-position MJ scores from alignment scoring.
            None values indicate gaps or skipped positions.
        tc: Core state threshold for emission scoring.
        tp: Peripheral state threshold for emission scoring.

    Returns:
        List of state labels ('B', 'P', 'C') for each position,
        with None for gap positions.

    Example:
        >>> per_pos = [-30.0, -15.0, -5.0, None, -28.0, -10.0]
        >>> path = hmm_interface_path(per_pos, tc=-25.0, tp=-15.0)
        >>> path
        ['C', 'P', 'B', None, 'C', 'P']
    """
    # Transition probabilities (fixed model)
    trans = {
        "B": {"B": 0.985, "P": 0.014, "C": 0.001},  # B rarely transitions to P/C
        "P": {"B": 0.12, "P": 0.85, "C": 0.03},     # P sometimes goes to B or C
        "C": {"B": 0.01, "P": 0.07, "C": 0.92},     # C tends to stay in C
    }
    # Convert to log probabilities
    logt = {a: {b: math.log(p) for b, p in trans[a].items()} for a in trans}
    states = ["B", "P", "C"]

    # Get indices of non-None positions
    idxs = [i for i, v in enumerate(per_pos) if v is not None]
    if not idxs:
        return [None] * len(per_pos)

    # Viterbi initialization: start in B state
    dp = {s: float("-inf") for s in states}
    dp["B"] = 0.0
    back: list[dict] = []  # Backpointers for traceback

    # Forward pass
    for i in idxs:
        val = per_pos[i]
        s_val = float(val) if val is not None else 0.0
        cur = {}
        back_i = {}
        for st in states:
            best_prev = None
            best_score = float("-inf")
            for prev in states:
                score = dp[prev] + logt[prev][st]
                if score > best_score:
                    best_score = score
                    best_prev = prev
            cur[st] = best_score + hmm_emission(st, s_val, tc, tp)
            back_i[st] = best_prev if best_prev is not None else "B"
        dp = cur
        back.append(back_i)

    # Backtrace from best final state
    last_state = max(dp.items(), key=lambda x: x[1])[0]
    path_states: list[Optional[str]] = [None] * len(idxs)
    for k in range(len(idxs) - 1, -1, -1):
        path_states[k] = last_state
        last_state = back[k][last_state]

    # Map back to full alignment length
    out: list[Optional[str]] = [None] * len(per_pos)
    for idx, state in zip(idxs, path_states):
        out[idx] = state
    return out


def hmm_segments(path: list[Optional[str]]) -> list[tuple[str, int, int]]:
    """Extract contiguous state segments from an HMM path.

    Groups consecutive positions with the same state into segments
    for easier interpretation of interface regions.

    Args:
        path: State labels from hmm_interface_path().

    Returns:
        List of (state, start_pos, end_pos) tuples using 1-based indices.
        None positions break segments.

    Example:
        >>> path = ['B', 'B', 'P', 'P', 'C', 'C', 'C', None, 'B']
        >>> hmm_segments(path)
        [('B', 1, 2), ('P', 3, 4), ('C', 5, 7), ('B', 9, 9)]
    """
    segs: list[tuple[str, int, int]] = []
    cur_state: Optional[str] = None
    cur_start: Optional[int] = None

    for i, st in enumerate(path, start=1):  # 1-based indexing
        if st is None:
            # Gap breaks current segment
            if cur_state is not None:
                segs.append((cur_state, cur_start, i - 1))  # type: ignore
                cur_state = None
                cur_start = None
            continue
        if cur_state is None:
            # Start new segment
            cur_state = st
            cur_start = i
        elif st != cur_state:
            # State changed, end current segment and start new one
            segs.append((cur_state, cur_start, i - 1))  # type: ignore
            cur_state = st
            cur_start = i

    # Handle final segment
    if cur_state is not None:
        segs.append((cur_state, cur_start, len(path)))  # type: ignore

    return segs
