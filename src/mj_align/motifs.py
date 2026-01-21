"""ScanProsite-style pattern generation and parsing.

This module provides utilities for working with protein sequence motifs
in ScanProsite format:

- Pattern generation from aligned sequences and anchor positions
- Pattern parsing into position-specific residue sets
- Pattern matching against FASTA databases
- Complement motif generation based on MJ interaction thresholds

ScanProsite pattern syntax:
    - Single residue: 'A', 'K', etc.
    - Any residue: 'x'
    - Repeat: 'x(n)' for n positions
    - Alternative: '[AV]' matches A or V
    - Positions separated by '-'

Example patterns:
    - 'R-x-R-x(2)-F-P' : R, any, R, any 2, F, P
    - 'K-[RK]-x-D' : K, R or K, any, D
"""

import itertools
from collections.abc import Iterable
from typing import Optional

from .amino_acid_properties import (
    AA20,
    AROMATICS,
    CLUSTAL_STRONG_GROUPS,
    NEG_CHARGES,
    POS_CHARGES,
    apply_mj_overrides,
)
from .fasta_io import read_fasta_all


def scanprosite_motif_from_anchors(
    seq_aln: str,
    anchors: Iterable[int],
) -> str:
    """Build a ScanProsite-style motif from aligned sequence and anchors.

    Creates a pattern where anchor positions emit the concrete residue
    and non-anchor positions emit 'x' (any residue). Consecutive x's
    are compressed to x(n) notation.

    Args:
        seq_aln: Aligned sequence (may contain gaps).
        anchors: 1-based indices of anchor positions.

    Returns:
        ScanProsite pattern string (e.g., 'R-x-R-x(2)-F-P').

    Example:
        >>> scanprosite_motif_from_anchors("RXRXXFP", [1, 3, 6, 7])
        'R-x-R-x(2)-F-P'
    """
    s = seq_aln.strip().upper()
    a_set = set(int(i) for i in anchors)

    # Build raw token list
    tokens: list[str] = []
    for i, ch in enumerate(s, start=1):  # 1-based indexing
        if i in a_set and ch in AA20:
            tokens.append(ch)
        else:
            tokens.append("x")

    # Trim leading/trailing x's
    while tokens and tokens[0] == "x":
        tokens.pop(0)
    while tokens and tokens[-1] == "x":
        tokens.pop()

    # Compress x-runs
    out: list[str] = []
    run = 0
    for t in tokens + ["__END__"]:
        if t == "x":
            run += 1
            continue
        if run:
            out.append("x" if run == 1 else f"x({run})")
            run = 0
        if t != "__END__":
            out.append(t)

    return "-".join(out)


def combined_aligned_regex(
    seq1_aln: str,
    seq2_aln: str,
    *,
    gap_char: str = "-",
) -> str:
    """Combine two aligned sequences into a ScanProsite-style regex.

    For each position:
    - If identical: emit single residue
    - If different but both valid: emit [AB] bracket set
    - If gap or unknown: emit 'x'

    Args:
        seq1_aln: First aligned sequence.
        seq2_aln: Second aligned sequence (same length).
        gap_char: Gap character. Default '-'.

    Returns:
        Combined ScanProsite pattern.

    Raises:
        ValueError: If sequences have different lengths.

    Example:
        >>> combined_aligned_regex("ACDEF", "ACDKF")
        'A-C-D-[EK]-F'
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    tokens: list[str] = []
    for a, b in zip(s1, s2):
        if a == gap_char or b == gap_char:
            tokens.append("x")
            continue
        if a not in AA20 or b not in AA20:
            tokens.append("x")
            continue
        if a == b:
            tokens.append(a)
        else:
            pair = "".join(sorted({a, b}))
            tokens.append("[" + pair + "]")

    # Trim leading/trailing x's
    while tokens and tokens[0] == "x":
        tokens.pop(0)
    while tokens and tokens[-1] == "x":
        tokens.pop()

    # Compress x-runs
    out: list[str] = []
    run = 0
    for t in tokens + ["__END__"]:
        if t == "x":
            run += 1
            continue
        if run:
            out.append("x" if run == 1 else f"x({run})")
            run = 0
        if t != "__END__":
            out.append(t)

    return "-".join(out)


def combined_aligned_strong_regex(
    seq1_aln: str,
    seq2_aln: str,
    *,
    gap_char: str = "-",
) -> str:
    """Combine aligned sequences with Clustal strong-similarity expansion.

    Like combined_aligned_regex but expands bracket sets to include
    other residues that share strong similarity with both original
    residues.

    Args:
        seq1_aln: First aligned sequence.
        seq2_aln: Second aligned sequence.
        gap_char: Gap character. Default '-'.

    Returns:
        Combined ScanProsite pattern with expanded similarity.

    Raises:
        ValueError: If sequences have different lengths.
    """
    from .clustering import _clustal_any_pair, _clustal_strong_pair

    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    tokens: list[str] = []
    for a, b in zip(s1, s2):
        if a == gap_char or b == gap_char:
            tokens.append("x")
            continue
        if a not in AA20 or b not in AA20:
            tokens.append("x")
            continue

        if a == b:
            # For identical residues, expand to shared strong groups
            groups = [g for g in CLUSTAL_STRONG_GROUPS if a in g]
            if groups:
                common = set(groups[0])
                for g in groups[1:]:
                    common &= g
                if common and len(common) > 1:
                    tokens.append("[" + "".join(sorted(common)) + "]")
                else:
                    tokens.append(a)
            else:
                tokens.append(a)
            continue

        # Different residues
        base = {a, b}
        if not _clustal_any_pair(a, b):
            tokens.append("x")
            continue

        # Expand to include residues similar to both
        expanded = set(base)
        use_strong = _clustal_strong_pair(a, b)
        for cand in AA20:
            count = 0
            for aa in base:
                if use_strong:
                    if _clustal_strong_pair(aa, cand):
                        count += 1
                else:
                    if _clustal_any_pair(aa, cand):
                        count += 1
            if count >= 2:
                expanded.add(cand)

        if len(expanded) == 1:
            tokens.append(next(iter(expanded)))
        else:
            tokens.append("[" + "".join(sorted(expanded)) + "]")

    # Trim leading/trailing x's
    while tokens and tokens[0] == "x":
        tokens.pop(0)
    while tokens and tokens[-1] == "x":
        tokens.pop()

    # Compress x-runs
    out: list[str] = []
    run = 0
    for t in tokens + ["__END__"]:
        if t == "x":
            run += 1
            continue
        if run:
            out.append("x" if run == 1 else f"x({run})")
            run = 0
        if t != "__END__":
            out.append(t)

    return "-".join(out)


def parse_scanprosite_pattern(pattern: str) -> list[str]:
    """Parse a ScanProsite pattern into per-position tokens.

    Handles basic pattern syntax with single residues, 'x' wildcards,
    and x(n) repeat notation.

    Args:
        pattern: ScanProsite pattern string (dash-separated).

    Returns:
        List of single-character tokens ('X' for wildcard, 'A' for residues).

    Raises:
        ValueError: If pattern contains invalid tokens.

    Example:
        >>> parse_scanprosite_pattern("R-x(2)-K")
        ['R', 'X', 'X', 'K']
    """
    tokens: list[str] = []
    for raw in pattern.strip().upper().split("-"):
        raw = raw.strip()
        if not raw:
            continue

        if raw == "X":
            tokens.append("X")
            continue
        if raw.startswith("X(") and raw.endswith(")"):
            n_str = raw[2:-1].strip()
            if not n_str.isdigit():
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            n = int(n_str)
            if n <= 0:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            tokens.extend(["X"] * n)
            continue
        if len(raw) == 1 and raw in AA20:
            tokens.append(raw)
            continue

        raise ValueError(f"Unsupported ScanProsite token: {raw}")

    return tokens


def parse_scanprosite_pattern_with_sets(
    pattern: str,
) -> list[Optional[set[str]]]:
    """Parse ScanProsite pattern into per-position residue sets.

    Handles full pattern syntax including bracket sets like [RK].

    Args:
        pattern: ScanProsite pattern string.

    Returns:
        List where each element is:
            - Set of allowed residues for that position
            - None for wildcard ('x') positions

    Raises:
        ValueError: If pattern contains invalid tokens.

    Example:
        >>> parse_scanprosite_pattern_with_sets("R-[KR]-x-D")
        [{'R'}, {'K', 'R'}, None, {'D'}]
    """
    tokens: list[Optional[set[str]]] = []
    for raw in pattern.strip().upper().split("-"):
        raw = raw.strip()
        if not raw:
            continue

        if raw == "X":
            tokens.append(None)
            continue
        if raw.startswith("X(") and raw.endswith(")"):
            n_str = raw[2:-1].strip()
            if not n_str.isdigit():
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            n = int(n_str)
            if n <= 0:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            tokens.extend([None] * n)
            continue
        if raw.startswith("[") and raw.endswith("]") and len(raw) >= 3:
            body = raw[1:-1].strip()
            if not body:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            allowed = {c for c in body if c in AA20}
            if not allowed:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            tokens.append(allowed)
            continue
        if len(raw) == 1 and raw in AA20:
            tokens.append({raw})
            continue

        raise ValueError(f"Unsupported ScanProsite token: {raw}")

    return tokens


def scanprosite_forms_in_fasta(
    pattern: str,
    fasta_path: str,
    *,
    unknown_policy: str,
    name_filter: Optional[str] = None,
) -> dict[str, int]:
    """Search a FASTA file for all instances of a ScanProsite pattern.

    Scans through sequences and counts occurrences of each concrete
    form that matches the pattern.

    Args:
        pattern: ScanProsite pattern to search for.
        fasta_path: Path to FASTA file.
        unknown_policy: How to handle unknown residues (error/skip/zero).
        name_filter: Optional substring to filter FASTA entries.

    Returns:
        Dictionary mapping matched sequence forms to their counts.

    Raises:
        ValueError: If unknown residue found with 'error' policy.

    Example:
        >>> counts = scanprosite_forms_in_fasta("R-x-R", "proteins.fasta", unknown_policy="skip")
        >>> counts
        {'RAR': 15, 'RKR': 8, 'RRR': 3}
    """
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    if not tokens:
        return {}

    window = len(tokens)
    counts: dict[str, int] = {}

    for _name, seq in read_fasta_all(fasta_path):
        if name_filter and name_filter not in _name:
            continue

        s = seq.upper()
        if len(s) < window:
            continue

        # Scan all positions
        for i in range(0, len(s) - window + 1):
            ok = True
            for k, allowed in enumerate(tokens):
                aa = s[i + k]
                if aa not in AA20:
                    if unknown_policy == "skip":
                        ok = False
                        break
                    if unknown_policy == "error":
                        raise ValueError(
                            f"Unknown residue '{aa}' in FASTA at {fasta_path}"
                        )
                    # unknown_policy == "zero"
                    if allowed is not None:
                        ok = False
                        break
                    continue
                if allowed is not None and aa not in allowed:
                    ok = False
                    break

            if ok:
                form = s[i : i + window]
                counts[form] = counts.get(form, 0) + 1

    return counts


def scanprosite_expected_forms(pattern: str) -> Optional[set[str]]:
    """Enumerate all possible concrete forms for a pattern.

    Args:
        pattern: ScanProsite pattern string.

    Returns:
        Set of all possible sequences matching the pattern,
        or None if pattern contains wildcard positions.

    Example:
        >>> scanprosite_expected_forms("A-[KR]-D")
        {'AKD', 'ARD'}
        >>> scanprosite_expected_forms("A-x-D")  # Contains wildcard
        None
    """
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    if not tokens:
        return set()

    # Check for wildcard positions
    for t in tokens:
        if t is None:
            return None

    # Generate all combinations
    pools: list[list[str]] = []
    for t in tokens:
        pools.append(sorted(t))  # type: ignore

    return {"".join(p) for p in itertools.product(*pools)}


def scanprosite_complement_motif(
    pattern: str,
    mj: dict[tuple[str, str], float],
    *,
    thr: float,
    set_mode: str = "all",
    context_offset: bool = False,
    top_k: int = 0,
) -> str:
    """Generate a complement motif based on MJ interaction thresholds.

    For each position in the input pattern, find residues that have
    favorable MJ interactions (score <= threshold) with the pattern
    residues.

    Args:
        pattern: Input ScanProsite pattern.
        mj: MJ matrix dictionary.
        thr: MJ threshold for complement detection.
        set_mode: For bracket sets, require complement to match:
            - 'all': All residues in set (intersection)
            - 'any': Any residue in set (union)
        context_offset: Include neighboring position residues in scoring.
        top_k: If >0, limit complement to top_k best-scoring residues.

    Returns:
        ScanProsite pattern for the complement motif.

    Raises:
        ValueError: If set_mode is invalid.

    Example:
        >>> mj = load_mj_csv("mj_matrix.csv")
        >>> scanprosite_complement_motif("W-x-W", mj, thr=-25.0)
        '[AG]-x-[AG]'
    """
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    out: list[str] = []
    aa_list = sorted(AA20)

    if set_mode not in {"all", "any"}:
        raise ValueError("set_mode must be 'all' or 'any'")

    def allowed_for_residue(res: str) -> set[str]:
        """Find residues that complement a single residue."""
        allowed_set: set[str] = set()
        for b in aa_list:
            val = mj.get((res, b), mj.get((b, res)))
            val = apply_mj_overrides(res, b, val)
            if val is None:
                continue
            if float(val) <= thr:
                allowed_set.add(b)
        return allowed_set

    def allowed_for_set(res_set: set[str]) -> set[str]:
        """Find residues that complement a set of residues."""
        if set_mode == "all":
            allowed = set(aa_list)
            for aa in res_set:
                allowed &= allowed_for_residue(aa)
            return allowed
        # set_mode == "any"
        allowed_any: set[str] = set()
        for aa in res_set:
            allowed_any |= allowed_for_residue(aa)
        return allowed_any

    for idx, t in enumerate(tokens):
        if t is None:
            out.append("x")
            continue

        allowed = allowed_for_set(t)

        # Optional: include context from neighboring positions
        if context_offset:
            neighbor_sets: list[set[str]] = []
            if idx - 1 >= 0 and tokens[idx - 1] is not None:
                neighbor_sets.append(tokens[idx - 1])  # type: ignore
            if idx + 1 < len(tokens) and tokens[idx + 1] is not None:
                neighbor_sets.append(tokens[idx + 1])  # type: ignore
            for nset in neighbor_sets:
                for aa in nset:
                    if aa in POS_CHARGES or aa in NEG_CHARGES or aa == "P" or aa in AROMATICS:
                        allowed |= allowed_for_residue(aa)

        # Optional: limit to top_k best scoring
        if top_k and allowed:
            sources = set(t)
            if context_offset:
                for nset in neighbor_sets:
                    for aa in nset:
                        if aa in POS_CHARGES or aa in NEG_CHARGES or aa == "P" or aa in AROMATICS:
                            sources.add(aa)

            scored = []
            for b in sorted(allowed):
                best = None
                for aa in sources:
                    val = mj.get((aa, b), mj.get((b, aa)))
                    val = apply_mj_overrides(aa, b, val)
                    if val is None:
                        continue
                    v = float(val)
                    if best is None or v < best:
                        best = v
                if best is not None:
                    scored.append((best, b))

            scored.sort(key=lambda x: (x[0], x[1]))
            allowed = {b for _, b in scored[:top_k]}

        # Format output token
        if not allowed:
            out.append("x")
        elif len(allowed) == 1:
            out.append(sorted(allowed)[0])
        else:
            out.append("[" + "".join(sorted(allowed)) + "]")

    # Compress consecutive x into x(n)
    compressed: list[str] = []
    run = 0
    for tok in out + ["__END__"]:
        if tok == "x":
            run += 1
            continue
        if run:
            compressed.append("x" if run == 1 else f"x({run})")
            run = 0
        if tok != "__END__":
            compressed.append(tok)

    # Trim leading/trailing x runs
    while compressed and compressed[0].startswith("x"):
        compressed.pop(0)
    while compressed and compressed[-1].startswith("x"):
        compressed.pop()

    return "-".join(compressed)


def avg_mj_score_pattern_to_seq(
    pattern: str,
    seq: str,
    mj: dict[tuple[str, str], float],
    *,
    unknown_policy: str,
) -> Optional[float]:
    """Compute average MJ score between a degenerate pattern and a sequence.

    For positions with multiple allowed residues (bracket sets), averages
    the scores across all allowed residues.

    Args:
        pattern: ScanProsite pattern (may contain bracket sets and wildcards).
        seq: Concrete sequence to score against.
        mj: MJ matrix dictionary.
        unknown_policy: How to handle unknown residues.

    Returns:
        Average total MJ score, or None if incompatible lengths or errors.

    Example:
        >>> avg_mj_score_pattern_to_seq("[AV]-K-D", "LKE", mj, unknown_policy="error")
        -45.5  # Average of scores for A and V against L at position 1, etc.
    """
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    s = seq.strip().upper()
    if len(tokens) != len(s):
        return None

    total = 0.0
    aa_list = sorted(AA20)

    for t, b in zip(tokens, s):
        if b not in AA20:
            if unknown_policy == "error":
                raise ValueError(f"Unknown residue {b!r} in sequence")
            if unknown_policy == "skip":
                return None
            # unknown_policy == "zero"
            continue

        allowed = aa_list if t is None else sorted(t)
        if not allowed:
            return None

        vals = []
        for a in allowed:
            val = mj.get((a, b), mj.get((b, a)))
            val = apply_mj_overrides(a, b, val)
            if val is None:
                continue
            vals.append(float(val))

        if not vals:
            return None
        total += sum(vals) / len(vals)

    return total
