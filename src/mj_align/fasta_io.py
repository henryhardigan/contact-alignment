"""FASTA file input/output operations.

This module provides utilities for reading and parsing FASTA-formatted
protein sequence files. FASTA is a standard text-based format for
representing nucleotide or protein sequences.

FASTA format:
    >sequence_name description
    MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKA
    LGISPDLIQKITGSLLLTTDSIIYPSQWKPAEFNNKILNNKIKEYILKNNFIDEKNIIKKV
    ...
"""

from collections.abc import Generator
from typing import Optional


def read_fasta_all(path: str) -> Generator[tuple[str, str], None, None]:
    """Read all entries from a FASTA file.

    Parses a multi-entry FASTA file and yields each sequence with its
    header. Handles multi-line sequences by concatenating continuation
    lines.

    Args:
        path: Path to the FASTA file.

    Yields:
        Tuples of (header, sequence) where header is the text after '>'
        and sequence is the concatenated amino acid string.

    Raises:
        ValueError: If no FASTA header is found in the file.

    Example:
        >>> for name, seq in read_fasta_all("proteins.fasta"):
        ...     print(f"{name}: {len(seq)} residues")
        sp|P12345|PROT_HUMAN: 350 residues
        sp|Q67890|PROT_MOUSE: 348 residues
    """
    name: Optional[str] = None
    seq_parts: list = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Yield previous entry if exists
                if name is not None:
                    yield name, "".join(seq_parts)
                # Start new entry
                name = line[1:].strip()
                seq_parts = []
            else:
                # Continuation of sequence
                seq_parts.append(line)

        # Yield final entry
        if name is not None:
            yield name, "".join(seq_parts)

    # Validate file had at least one entry
    if name is None:
        raise ValueError(f"No FASTA header found in {path}")


def read_fasta_entry(path: str, name_filter: str) -> tuple[str, str]:
    """Read a specific entry from a FASTA file by name substring match.

    Searches through a FASTA file for entries whose header contains the
    specified filter string. Requires exactly one match to avoid ambiguity.

    Args:
        path: Path to the FASTA file.
        name_filter: Substring to search for in FASTA headers.

    Returns:
        Tuple of (header, sequence) for the matching entry.

    Raises:
        ValueError: If no entries match or multiple entries match the filter.

    Example:
        >>> name, seq = read_fasta_entry("swissprot.fasta", "P12345")
        >>> print(name)
        sp|P12345|PROT_HUMAN Protein description
    """
    matches = []
    for name, seq in read_fasta_all(path):
        if name_filter in name:
            matches.append((name, seq))

    if not matches:
        raise ValueError(f"No FASTA records match: {name_filter}")
    if len(matches) > 1:
        sample = "; ".join(n for n, _ in matches[:3])
        raise ValueError(
            f"Multiple FASTA records match: {name_filter} (n={len(matches)}). "
            f"Examples: {sample}"
        )
    return matches[0]
