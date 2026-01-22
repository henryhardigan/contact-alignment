# clustering.py - Clustal-Style Sequence Similarity Scoring

## Purpose

This module provides Clustal-style similarity scoring between aligned protein sequences. While MJ scoring measures complementarity (puzzle piece fitting), Clustal scoring measures similarity (evolutionary relatedness).

## Clustal Scoring System

| Symbol | Meaning | Score | Criterion |
|--------|---------|-------|-----------|
| `*` | Identity | 1.0 | Exact match (A=A) |
| `:` | Strong | 0.5 | Same strong similarity group |
| `.` | Weak | 0.25 | Same weak similarity group |
| ` ` | None | 0.0 | No similarity |

## Key Functions

### `clustal_pair_score(a: str, b: str) -> Optional[float]`

Scores a single amino acid pair using Clustal rules.

**Returns:**
- `1.0` for identity
- `0.5` for strong similarity
- `0.25` for weak similarity
- `0.0` for no similarity
- `None` if either residue is not a standard amino acid

**Example:**
```python
clustal_pair_score("A", "A")  # Returns 1.0 (identity)
clustal_pair_score("S", "T")  # Returns 0.5 (strong - both in STA group)
clustal_pair_score("A", "V")  # Returns 0.25 (weak - both in ATV group)
clustal_pair_score("A", "K")  # Returns 0.0 (no similarity)
```

### `clustal_similarity(seq1_aln, seq2_aln, *, gap_char='-') -> tuple[str, float, float, int]`

Computes Clustal-style similarity line and scores for an alignment.

**Returns:** `(symbols, score, score_normalized, n_eligible)`
- `symbols`: String of similarity symbols (`*`, `:`, `.`, ` `)
- `score`: Sum of position scores
- `score_normalized`: Score divided by eligible positions
- `n_eligible`: Number of non-gap aligned positions

**Example:**
```python
symbols, score, norm, n = clustal_similarity("ACDEF", "ACDEK")
print(symbols)  # "****." - 4 identities, 1 weak similarity (E/K in NDEQHK)
print(f"Score: {score}/{n} = {norm:.2f}")
```

### `clustal_anchor_positions(s1, s2, mode='strong') -> tuple[list[int], list[int]]`

Finds anchor positions based on Clustal similarity criteria.

**Parameters:**
- `mode`: Anchor criterion
  - `'identity'`: Only exact matches
  - `'strong'`: Only strong similarity (not identity)
  - `'strong+identity'`: Strong similarity or identity
  - `'weak'`: Only weak similarity (not strong or identity)
  - `'any'`: Any similarity (identity, strong, or weak)

**Returns:** `(anchor_positions, eligible_positions)` using 1-based indices

**Example:**
```python
anchors, eligible = clustal_anchor_positions("ACDEF", "ACDEK", "identity")
# anchors = [1, 2, 3, 4] - positions with exact matches
# eligible = [1, 2, 3, 4, 5] - all non-gap positions
```

### `clustal_entry_key(header: str) -> tuple[str, str]`

Extracts identifiers from a FASTA header for grouping.

**Example:**
```python
clustal_entry_key("sp|P12345|PROT_HUMAN Description")
# Returns ("PROT", "PROT_HUMAN")
```

## Helper Functions

### `_clustal_strong_pair(a, b) -> bool`

Checks if two amino acids belong to the same strong similarity group.

### `_clustal_any_pair(a, b) -> bool`

Checks if two amino acids share any similarity group (strong or weak).

### `_kmer_hits(seq, k, kmer_set, *, unknown_policy) -> list[bool]`

Finds positions where k-mers match a reference set.

**Use case:** Prefiltering in window search to quickly identify potentially similar regions.

### `parse_clustal_require(s: Optional[str]) -> Optional[list[tuple[int, set[str]]]]`

Parses position requirement specifications for constrained searches.

**Format:** `'pos=residue(s),...'` where positions are 1-based

**Example:**
```python
parse_clustal_require("3=W,4=[LI]")
# Returns [(2, {'W'}), (3, {'L', 'I'})]  # 0-based offsets
```

## When to Use Clustal vs MJ Scoring

| Use Case | Scoring Method |
|----------|----------------|
| Finding homologous sequences | Clustal |
| Predicting binding interfaces | MJ |
| Evolutionary analysis | Clustal |
| Protein-protein docking | MJ |
| Prefiltering for MJ analysis | Clustal |

## Combining Clustal and MJ

A common workflow uses Clustal for initial filtering:

```python
from mj_align import (
    best_ungapped_window_pair_clustal,
    score_aligned, anchors_by_threshold
)

# Phase 1: Find similar regions using Clustal
score, i, j, ident = best_ungapped_window_pair_clustal(
    seq1, seq2, window=10, prefilter_min_identity=3
)

# Phase 2: Score that region with MJ for complementarity
window_seq1 = seq1[i:i+10]
window_seq2 = seq2[j:j+10]
mj_total, per_pos = score_aligned(window_seq1, window_seq2, mj)
anchors = anchors_by_threshold(per_pos, thr=-25.0)
```

## Dependencies

- `amino_acid_properties`: AA20, CLUSTAL_STRONG_GROUPS, CLUSTAL_WEAK_GROUPS
