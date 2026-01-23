"""Clustal-style sequence similarity scoring.

This module provides functions for computing Clustal-style similarity
between aligned protein sequences, including:

- Identity, strong similarity, and weak similarity classifications
- Position-by-position scoring with standard Clustal symbols (*, :, .)
- Anchor position detection based on similarity thresholds
- Pairwise residue scoring functions

The Clustal scoring follows standard conventions:
    - '*' (identity): Exact amino acid match (score: 1.0)
    - ':' (strong): Amino acids in same strong similarity group (score: 0.5)
    - '.' (weak): Amino acids in same weak similarity group (score: 0.25)
    - ' ' (none): No similarity (score: 0.0)

References:
    - Clustal Omega: https://www.ebi.ac.uk/Tools/msa/clustalo/
    - Thompson et al. (1994) CLUSTAL W
"""

from typing import Optional

from .amino_acid_properties import (
    AA20,
    CLUSTAL_STRONG_GROUPS,
    CLUSTAL_WEAK_GROUPS,
)


def _clustal_strong_pair(a: str, b: str) -> bool:
    """Check if two amino acids belong to the same strong similarity group.

    Args:
        a: First amino acid single-letter code.
        b: Second amino acid single-letter code.

    Returns:
        True if both residues are in any of the Clustal strong groups.
    """
    return any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS)


def _clustal_any_pair(a: str, b: str) -> bool:
    """Check if two amino acids share any similarity group (strong or weak).

    Args:
        a: First amino acid single-letter code.
        b: Second amino acid single-letter code.

    Returns:
        True if both residues are in any Clustal similarity group.
    """
    return any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS + CLUSTAL_WEAK_GROUPS)


def clustal_pair_score(a: str, b: str) -> Optional[float]:
    """Score a single amino acid pair using Clustal rules.

    Args:
        a: First amino acid single-letter code.
        b: Second amino acid single-letter code.

    Returns:
        Score based on similarity:
            - 1.0 for identity
            - 0.5 for strong similarity
            - 0.25 for weak similarity
            - 0.0 for no similarity
            - None if either residue is not a standard amino acid

    Example:
        >>> clustal_pair_score("A", "A")
        1.0
        >>> clustal_pair_score("S", "T")  # Strong similarity
        0.5
        >>> clustal_pair_score("A", "K")  # No similarity
        0.0
    """
    if a not in AA20 or b not in AA20:
        return None
    if a == b:
        return 1.0
    if any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS):
        return 0.5
    if any(a in g and b in g for g in CLUSTAL_WEAK_GROUPS):
        return 0.25
    return 0.0


def clustal_similarity(
    seq1_aln: str,
    seq2_aln: str,
    *,
    gap_char: str = "-",
) -> tuple[str, float, float, int]:
    """Compute Clustal-style similarity line and scores for an alignment.

    Generates the characteristic Clustal alignment annotation where:
        - '*' marks identical residues
        - ':' marks strongly similar residues
        - '.' marks weakly similar residues
        - ' ' marks dissimilar or gap positions

    Args:
        seq1_aln: First aligned sequence.
        seq2_aln: Second aligned sequence (same length as seq1_aln).
        gap_char: Character used for gaps. Default '-'.

    Returns:
        Tuple of (symbols, score, score_normalized, n_eligible) where:
            - symbols: String of similarity symbols for each position
            - score: Sum of position scores
            - score_normalized: Score divided by eligible positions
            - n_eligible: Number of non-gap aligned positions

    Raises:
        ValueError: If sequences have different lengths.

    Example:
        >>> symbols, score, norm, n = clustal_similarity("ACDEF", "ACDEK")
        >>> print(symbols)
        '****.'
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    symbols = []
    score = 0.0
    n_eligible = 0

    for a, b in zip(s1, s2):
        # Skip gap positions
        if a == gap_char or b == gap_char:
            symbols.append(" ")
            continue
        # Skip unknown residues
        if a not in AA20 or b not in AA20:
            symbols.append(" ")
            continue

        n_eligible += 1

        if a == b:
            symbols.append("*")
            score += 1.0
            continue
        if any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS):
            symbols.append(":")
            score += 0.5
            continue
        if any(a in g and b in g for g in CLUSTAL_WEAK_GROUPS):
            symbols.append(".")
            score += 0.25
            continue
        symbols.append(" ")

    score_norm = score / n_eligible if n_eligible else 0.0
    return "".join(symbols), score, score_norm, n_eligible


def clustal_anchor_positions(
    s1: str,
    s2: str,
    mode: str = "strong",
) -> tuple[list[int], list[int]]:
    """Find anchor positions based on Clustal similarity criteria.

    Identifies positions that meet specified similarity criteria and
    returns both anchor positions and all eligible positions.

    Args:
        s1: First aligned sequence.
        s2: Second aligned sequence.
        mode: Anchor criterion:
            - 'identity': Only exact matches
            - 'strong': Only strong similarity (not identity)
            - 'strong+identity': Strong similarity or identity
            - 'weak': Only weak similarity (not strong or identity)
            - 'any': Any similarity (identity, strong, or weak)

    Returns:
        Tuple of (anchor_positions, eligible_positions) using 1-based indices.

    Raises:
        ValueError: If mode is not recognized.

    Example:
        >>> anchors, eligible = clustal_anchor_positions("ACDEF", "ACDEK", "identity")
        >>> anchors  # Positions with exact matches
        [1, 2, 3, 4]
    """
    anchors: list[int] = []
    eligible: list[int] = []

    for i, (a, b) in enumerate(zip(s1, s2), start=1):  # 1-based indexing
        if a not in AA20 or b not in AA20:
            continue
        eligible.append(i)

        is_id = (a == b)
        is_strong = any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS)
        is_weak = any(a in g and b in g for g in CLUSTAL_WEAK_GROUPS)

        if mode == "identity":
            keep = is_id
        elif mode == "strong":
            keep = is_strong and not is_id
        elif mode == "strong+identity":
            keep = is_strong or is_id
        elif mode == "weak":
            keep = is_weak and not is_strong and not is_id
        elif mode == "any":
            keep = is_weak or is_strong or is_id
        else:
            raise ValueError(f"Unknown clustal anchor mode: {mode}")

        if keep:
            anchors.append(i)

    return anchors, eligible


def clustal_entry_key(header: str) -> tuple[str, str]:
    """Extract base ID and entry ID from a FASTA header.

    Parses UniProt-style headers to extract identifiers for grouping
    and identification purposes.

    Args:
        header: FASTA header line (without leading '>').

    Returns:
        Tuple of (base_id, entry_id) where:
            - base_id: Identifier before underscore (e.g., 'P12345')
            - entry_id: Full entry identifier (e.g., 'P12345_HUMAN')

    Example:
        >>> clustal_entry_key("sp|P12345|PROT_HUMAN Description")
        ('PROT', 'PROT_HUMAN')
    """
    head = header.split()[0]  # First whitespace-separated token
    if "|" in head:
        parts = head.split("|")
        entry = parts[2] if len(parts) >= 3 else head
    else:
        entry = head
    base = entry.split("_")[0]  # Part before species suffix
    return base, entry


def _kmer_hits(
    seq: str,
    k: int,
    kmer_set: set[str],
    *,
    unknown_policy: str,
) -> list[bool]:
    """Find positions where k-mers match a reference set.

    Scans a sequence and marks positions where the k-mer starting
    at that position exists in the reference set.

    Args:
        seq: Sequence to scan.
        k: K-mer length.
        kmer_set: Set of reference k-mers to match against.
        unknown_policy: How to handle unknown residues in k-mers.

    Returns:
        List of booleans indicating match at each position.

    Raises:
        ValueError: If unknown residue found with 'error' policy.
    """
    hits: list[bool] = []
    if k <= 0 or len(seq) < k:
        return hits

    for i in range(0, len(seq) - k + 1):
        kmer = seq[i : i + k]
        # Check for unknown residues
        if any(c not in AA20 for c in kmer):
            if unknown_policy == "error":
                raise ValueError("Unknown residue in Clustal k-mer scan")
            hits.append(False)
            continue
        hits.append(kmer in kmer_set)

    return hits


def parse_clustal_require(s: Optional[str]) -> Optional[list[tuple[int, set[str]]]]:
    """Parse position requirement specification for Clustal searching.

    Parses strings like '3=W,4=L' or '3=[WF],4=L' into structured
    constraints for position-specific amino acid requirements.

    Args:
        s: Requirement string in format 'pos=residue(s),...'.
            Uses 1-based positions. Residues can be single letters
            or bracketed sets like [WF].

    Returns:
        List of (0-based_offset, allowed_residues_set) tuples,
        or None if input is empty/None.

    Raises:
        ValueError: If format is invalid or residues are unknown.

    Example:
        >>> parse_clustal_require("3=W,4=[LI]")
        [(2, {'W'}), (3, {'L', 'I'})]
    """
    if not s:
        return None

    items: list[tuple[int, set[str]]] = []
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid clustal require token: {raw}")

        pos_str, aa_str = raw.split("=", 1)
        pos_str = pos_str.strip()
        aa_str = aa_str.strip().upper()

        if not pos_str.isdigit():
            raise ValueError(f"Invalid clustal require position: {pos_str}")
        if not aa_str:
            raise ValueError("Empty clustal require residues")

        # Handle bracketed sets
        if aa_str.startswith("[") and aa_str.endswith("]"):
            aa_str = aa_str[1:-1].strip()

        if not aa_str or any(c not in AA20 for c in aa_str):
            raise ValueError(f"Invalid clustal require residues: {aa_str}")

        pos0 = int(pos_str) - 1  # Convert to 0-based
        if pos0 < 0:
            raise ValueError(f"Invalid clustal require position: {pos_str}")

        items.append((pos0, set(aa_str)))

    return items or None
