# fasta_io.py - FASTA File Input/Output Operations

## Purpose

This module provides utilities for reading and parsing FASTA-formatted protein sequence files. FASTA is the standard text-based format for representing biological sequences.

## FASTA Format Overview

```
>sequence_name description text
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKA
LGISPDLIQKITGSLLLTTDSIIYPSQWKPAEFNNKILNNKIKEYILKNNFIDEKNIIKKV
...
>another_sequence more description
ACDEFGHIKLMNPQRSTVWY...
```

**Key features:**
- Header line starts with `>`
- Header contains sequence name/ID and optional description
- Sequence can span multiple lines
- Multiple entries per file

## Key Functions

### `read_fasta_all(path: str) -> Generator[tuple[str, str], None, None]`

Reads all entries from a FASTA file.

**Yields:** `(header, sequence)` tuples where:
- `header`: Text after `>` (name and description)
- `sequence`: Concatenated amino acid string (all lines joined)

**Example:**
```python
for name, seq in read_fasta_all("proteins.fasta"):
    print(f"{name}: {len(seq)} residues")

# Output:
# sp|P12345|PROT_HUMAN: 350 residues
# sp|Q67890|PROT_MOUSE: 348 residues
```

**Memory efficiency:** Uses a generator, so it doesn't load the entire file into memory at once.

### `read_fasta_entry(path: str, name_filter: str) -> tuple[str, str]`

Reads a specific entry by name substring match.

**Parameters:**
- `path`: Path to FASTA file
- `name_filter`: Substring to search for in headers

**Returns:** `(header, sequence)` for the matching entry

**Raises:**
- `ValueError`: If no entries match or multiple entries match

**Example:**
```python
# Find entry containing "P12345"
name, seq = read_fasta_entry("swissprot.fasta", "P12345")
print(f"Found: {name}")
```

**Why require unique matches?**

Ambiguous matches can lead to subtle bugs. If you need a specific entry, the filter should uniquely identify it. If multiple matches are expected, use `read_fasta_all` and filter manually.

## Error Handling

```python
# No matching entries
try:
    name, seq = read_fasta_entry("proteins.fasta", "NONEXISTENT")
except ValueError as e:
    print(f"Not found: {e}")

# Multiple matching entries
try:
    name, seq = read_fasta_entry("proteins.fasta", "HUMAN")  # Too generic
except ValueError as e:
    print(f"Ambiguous: {e}")
    # Message includes count and examples
```

## Common Usage Patterns

### Processing a database

```python
from mj_align import read_fasta_all

# Count entries
count = sum(1 for _ in read_fasta_all("database.fasta"))

# Find longest sequence
longest = max(read_fasta_all("database.fasta"), key=lambda x: len(x[1]))

# Filter by length
long_seqs = [(n, s) for n, s in read_fasta_all("database.fasta") if len(s) > 500]
```

### Working with UniProt entries

```python
from mj_align import read_fasta_entry

# UniProt format: sp|ACCESSION|ENTRY_NAME Description
name, seq = read_fasta_entry("uniprot.fasta", "P12345")

# Parse the header
parts = name.split("|")
db = parts[0]        # "sp" or "tr"
accession = parts[1] # "P12345"
entry = parts[2].split()[0]  # "PROT_HUMAN"
```

### Loading sequences for alignment

```python
from mj_align import read_fasta_all, score_aligned

# Load two sequences from separate files
name1, seq1 = next(read_fasta_all("protein1.fasta"))
name2, seq2 = next(read_fasta_all("protein2.fasta"))

# Or from same file with different filters
name1, seq1 = read_fasta_entry("proteins.fasta", "HUMAN")
name2, seq2 = read_fasta_entry("proteins.fasta", "MOUSE")
```

## Implementation Notes

1. **Line handling**: Empty lines are skipped, sequences are stripped of whitespace
2. **Case preservation**: Headers preserve original case, sequences are returned as-is (score functions will uppercase)
3. **Memory**: Only one entry is held in memory at a time (generator-based)
4. **Encoding**: Files are read with default encoding (usually UTF-8)

## Dependencies

None (uses only Python standard library)
