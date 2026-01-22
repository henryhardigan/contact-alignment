# motifs.py - ScanProsite-Style Pattern Generation and Parsing

## Purpose

This module provides utilities for working with protein sequence motifs in ScanProsite format. It can generate patterns from alignments, parse existing patterns, search databases for pattern matches, and create complement motifs.

## ScanProsite Pattern Syntax

| Element | Meaning | Example |
|---------|---------|---------|
| Single letter | Specific residue | `A`, `K`, `W` |
| `x` | Any residue | `x` |
| `x(n)` | Any n residues | `x(3)` = any 3 |
| `[ABC]` | Any of A, B, or C | `[KR]` = K or R |
| `-` | Position separator | `R-x-R` |

**Example patterns:**
- `R-x-R-x(2)-F-P`: R, any, R, any 2, F, P
- `K-[RK]-x-D`: K, (R or K), any, D

## Pattern Generation

### `scanprosite_motif_from_anchors(seq_aln, anchors) -> str`

Builds a pattern from aligned sequence and anchor positions.

**Algorithm:**
1. Anchor positions emit the concrete residue
2. Non-anchor positions emit `x`
3. Consecutive x's are compressed to `x(n)` notation
4. Leading/trailing x's are trimmed

**Example:**
```python
scanprosite_motif_from_anchors("RXRXXFP", [1, 3, 6, 7])
# Returns "R-x-R-x(2)-F-P"
```

### `combined_aligned_regex(seq1_aln, seq2_aln, ...) -> str`

Combines two aligned sequences into a pattern.

**Per-position rules:**
- Identical residues: emit single residue
- Different valid residues: emit `[AB]` bracket set
- Gap or unknown: emit `x`

**Example:**
```python
combined_aligned_regex("ACDEF", "ACDKF")
# Returns "A-C-D-[EK]-F"
```

### `combined_aligned_strong_regex(seq1_aln, seq2_aln, ...) -> str`

Like `combined_aligned_regex` but expands bracket sets to include residues with strong Clustal similarity to both originals.

## Pattern Parsing

### `parse_scanprosite_pattern(pattern) -> list[str]`

Parses pattern into per-position tokens.

**Returns:** List of single characters (`'X'` for wildcard, amino acid letter otherwise)

**Example:**
```python
parse_scanprosite_pattern("R-x(2)-K")
# Returns ["R", "X", "X", "K"]
```

### `parse_scanprosite_pattern_with_sets(pattern) -> list[Optional[set[str]]]`

Parses pattern into per-position residue sets.

**Returns:** List where each element is:
- Set of allowed residues for constrained positions
- `None` for wildcard positions

**Example:**
```python
parse_scanprosite_pattern_with_sets("R-[KR]-x-D")
# Returns [{'R'}, {'K', 'R'}, None, {'D'}]
```

## Pattern Searching

### `scanprosite_forms_in_fasta(pattern, fasta_path, *, unknown_policy, name_filter=None) -> dict[str, int]`

Searches a FASTA file for all instances of a pattern.

**Returns:** Dictionary mapping matched sequence forms to their counts.

**Example:**
```python
counts = scanprosite_forms_in_fasta("R-x-R", "proteins.fasta", unknown_policy="skip")
# Returns {'RAR': 15, 'RKR': 8, 'RRR': 3, ...}
```

### `scanprosite_expected_forms(pattern) -> Optional[set[str]]`

Enumerates all possible concrete forms for a pattern.

**Returns:**
- Set of all possible matching sequences if pattern is fully constrained
- `None` if pattern contains wildcard positions

**Example:**
```python
scanprosite_expected_forms("A-[KR]-D")
# Returns {'AKD', 'ARD'}

scanprosite_expected_forms("A-x-D")
# Returns None (wildcard position)
```

## Complement Motif Generation

### `scanprosite_complement_motif(pattern, mj, *, thr, set_mode='all', context_offset=False, top_k=0) -> str`

Generates a complement motif based on MJ interaction thresholds.

**For each position:** Find residues with favorable MJ interactions (score <= threshold) with the pattern residues.

**Parameters:**
- `thr`: MJ threshold for complement detection
- `set_mode`: For bracket sets:
  - `'all'`: Complement must work for ALL residues (intersection)
  - `'any'`: Complement works for ANY residue (union)
- `context_offset`: Include neighboring positions in scoring
- `top_k`: If >0, limit to top_k best-scoring complements

**Example:**
```python
# Find what complements aromatic residues
complement = scanprosite_complement_motif("W-x-W", mj, thr=-25.0)
# Returns something like "[AG]-x-[AG]" (small neutrals complement aromatics)
```

## Pattern Scoring

### `avg_mj_score_pattern_to_seq(pattern, seq, mj, *, unknown_policy) -> Optional[float]`

Computes average MJ score between a degenerate pattern and a sequence.

**Algorithm:** For positions with multiple allowed residues, averages scores across all allowed options.

**Example:**
```python
avg_mj_score_pattern_to_seq("[AV]-K-D", "LKE", mj, unknown_policy="error")
# Averages MJ(A,L) and MJ(V,L) for position 1, etc.
```

## Example Workflow

```python
from mj_align import (
    score_aligned, anchors_by_threshold,
    scanprosite_motif_from_anchors, scanprosite_complement_motif,
    scanprosite_forms_in_fasta
)

# 1. Score alignment and find anchors
total, per_pos = score_aligned(seq1, seq2, mj)
anchors = anchors_by_threshold(per_pos, thr=-25.0)

# 2. Generate pattern from anchors
pattern = scanprosite_motif_from_anchors(seq1, anchors)
print(f"Pattern: {pattern}")  # e.g., "R-x(2)-W-x-K"

# 3. Generate complement pattern
complement = scanprosite_complement_motif(pattern, mj, thr=-25.0)
print(f"Complement: {complement}")

# 4. Search database for complement matches
hits = scanprosite_forms_in_fasta(complement, "proteome.fasta", unknown_policy="skip")
print(f"Found {sum(hits.values())} instances of {len(hits)} forms")
```

## Dependencies

- `amino_acid_properties`: AA20, AROMATICS, CLUSTAL_STRONG_GROUPS, NEG_CHARGES, POS_CHARGES, apply_mj_overrides
- `fasta_io`: read_fasta_all
