"""Amino acid constants, classifications, and property-related utilities.

This module provides fundamental constants and utility functions for working
with protein sequences, including:

- Standard amino acid definitions (AA20)
- Chemical property classifications (aromatics, charges, hydrophobes)
- Clustal similarity groups (strong and weak)
- MJ matrix override rules for special residue pairs

These constants and functions are used throughout the mj_align package for
sequence analysis and scoring.

References:
    - Clustal Omega: https://www.ebi.ac.uk/Tools/msa/clustalo/
    - Thompson et al. (1994) CLUSTAL W
"""

from typing import Optional

# Standard 20 amino acids (canonical order for reproducibility)
AA20_STR: str = "ACDEFGHIKLMNPQRSTVWY"
"""String containing all 20 standard amino acids in canonical order."""

AA20: set[str] = set(AA20_STR)
"""Set of the 20 standard amino acid single-letter codes."""

# Chemical property classifications
AROMATICS: set[str] = {"W", "F", "Y"}
"""Aromatic amino acids: Tryptophan (W), Phenylalanine (F), Tyrosine (Y)."""

SMALL_NEUTRAL: set[str] = {"A", "G"}
"""Small neutral amino acids: Alanine (A), Glycine (G)."""

POS_CHARGES: set[str] = {"K", "R"}
"""Positively charged amino acids: Lysine (K), Arginine (R)."""

NEG_CHARGES: set[str] = {"D", "E"}
"""Negatively charged amino acids: Aspartic acid (D), Glutamic acid (E)."""

HYDROPHOBES: set[str] = {"A", "I", "L", "M", "V", "F", "W", "Y"}
"""Hydrophobic amino acids."""

HYDROPHOBE_OFFSET_WEIGHT: float = 0.6
"""Weight applied to hydrophobe context bonuses in scoring."""

# Clustal similarity groups for sequence alignment scoring
CLUSTAL_STRONG_GROUPS = [
    set("STA"),    # Small hydroxyl/sulfhydryl
    set("NEQK"),   # Amide/charged
    set("NHQK"),   # Amide/basic
    set("NDEQ"),   # Acidic/amide
    set("QHRK"),   # Basic/amide
    set("MILV"),   # Aliphatic
    set("MILF"),   # Aliphatic/aromatic
    set("HY"),     # Aromatic/basic
    set("FYW"),    # Aromatic
]
"""Clustal strong similarity groups - amino acids within a group score ':' in alignments."""

CLUSTAL_WEAK_GROUPS = [
    set("CSA"),    # Small
    set("ATV"),    # Small aliphatic
    set("SAG"),    # Small
    set("STNK"),   # Polar/basic
    set("STPA"),   # Small/polar
    set("SGND"),   # Small/acidic
    set("SNDEQK"), # Polar
    set("NDEQHK"), # Polar/charged
    set("NEQHRK"), # Polar/basic
    set("FVLIM"),  # Hydrophobic
    set("HFY"),    # Aromatic
]
"""Clustal weak similarity groups - amino acids within a group score '.' in alignments."""


def apply_mj_overrides(a: str, b: str, val: Optional[float]) -> Optional[float]:
    """Apply custom MJ score overrides for special residue pairs.

    Certain amino acid pairs have adjusted interaction scores based on
    structural considerations. Currently implements:
    - Aromatic + small neutral pairs: returns -8.0 (favorable interaction)

    Args:
        a: First amino acid (single-letter code).
        b: Second amino acid (single-letter code).
        val: Original MJ matrix value (may be None if pair not in matrix).

    Returns:
        The overridden score if a special rule applies, otherwise the
        original value unchanged.

    Example:
        >>> apply_mj_overrides("W", "A", -5.0)  # Aromatic + small neutral
        -8.0
        >>> apply_mj_overrides("K", "E", -10.0)  # No override
        -10.0
    """
    # Aromatics paired with small neutral residues get favorable score
    if a in AROMATICS and b in SMALL_NEUTRAL:
        return -8.0
    if b in AROMATICS and a in SMALL_NEUTRAL:
        return -8.0
    return val


def has_charge_run(seq: str, run_len: int = 3) -> bool:
    """Detect runs of charged residues in a protein sequence.

    Identifies consecutive stretches of positively charged (K/R) or
    negatively charged (D/E) amino acids that meet or exceed the
    specified length threshold.

    Args:
        seq: Protein sequence (single-letter amino acid codes).
        run_len: Minimum length of consecutive charged residues to
            detect. Default is 3.

    Returns:
        True if the sequence contains a run of positive or negative
        charges of at least run_len consecutive residues.

    Example:
        >>> has_charge_run("ACKKKLM", run_len=3)  # Has KKK run
        True
        >>> has_charge_run("ACKKLM", run_len=3)   # KK is only 2
        False
        >>> has_charge_run("ACDDDLM", run_len=3)  # Has DDD run
        True
    """
    if run_len <= 1:
        return False
    s = seq.strip().upper()
    count_pos = 0  # Count of consecutive positive charges (K/R)
    count_neg = 0  # Count of consecutive negative charges (D/E)
    for c in s:
        if c in "KR":
            count_pos += 1
            count_neg = 0
        elif c in "DE":
            count_neg += 1
            count_pos = 0
        else:
            count_pos = 0
            count_neg = 0
        if count_pos >= run_len or count_neg >= run_len:
            return True
    return False
