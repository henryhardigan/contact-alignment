# scoring.py - MJ Matrix Handling and Alignment Scoring

## Purpose

This module provides the core scoring functionality for protein sequence alignments using Miyazawa-Jernigan (MJ) contact potentials. It handles loading MJ matrices, computing alignment scores, and identifying anchor positions.

## Key Functions

### `load_mj_csv(path: str) -> dict[tuple[str, str], float]`

Loads an MJ matrix from a CSV file.

**Expected CSV format:**
- First row: header with blank cell followed by amino acid codes
- First column: amino acid codes
- Body: numeric MJ values

**Example:**
```python
mj = load_mj_csv("mj_matrix.csv")
print(mj[("A", "V")])  # Get Ala-Val interaction score
```

### `get_mj_scorer(mj) -> Callable[[str, str], Optional[float]]`

Creates a scorer function from an MJ matrix dictionary. The returned callable:
- Handles key order automatically (tries both (a,b) and (b,a))
- Applies MJ overrides for special residue pairs

**Example:**
```python
scorer = get_mj_scorer(mj)
score = scorer("W", "A")  # Returns -8.0 (aromatic + small neutral override)
```

### `score_aligned(seq1_aln, seq2_aln, mj, ...) -> tuple[float, list[Optional[float]]]`

Scores two aligned sequences position-by-position.

**Parameters:**
- `seq1_aln`, `seq2_aln`: Aligned sequences (same length, may contain gaps)
- `mj`: MJ matrix dictionary
- `gap_char`: Gap character (default: '-')
- `gap_penalty`: Penalty per gap position (default: None = ignore gaps)
- `unknown_policy`: How to handle unknown residues ('error', 'skip', 'zero')
- `context_bonus`: Apply bonuses for neighboring residue interactions

**Returns:**
- `total_score`: Sum of all position scores
- `per_pos`: List of per-position scores (None for skipped positions)

**Example:**
```python
total, per_pos = score_aligned("ACDEF", "ACDEK", mj)
print(f"Total: {total}, Positions: {per_pos}")
```

### `score_aligned_with_gaps(aln1, aln2, mj, *, gap_open, gap_ext, ...) -> float`

Scores aligned sequences using affine gap penalties (separate open/extend costs).

**Parameters:**
- `gap_open`: Penalty for opening a new gap
- `gap_ext`: Penalty for extending an existing gap

This better models biological gap formation where starting a gap is more costly than extending one.

### `context_bonus_aligned(seq1_aln, seq2_aln, mj, ...) -> list[float]`

Computes context bonuses for neighboring residue interactions.

**Bonus rules:**
- Charge opposites (K/R vs D/E) at ±1: +0.25 × MJ score
- Proline with aromatic at ±1: +0.5 × MJ score
- Proline with proline at ±1: +0.5 × MJ score
- Hydrophobe with hydrophobe at ±1: +0.6 × MJ score

These bonuses reflect structural tendencies for certain residue combinations to co-occur at interfaces.

### `anchors_by_threshold(per_pos, thr=-25.0) -> list[int]`

Finds anchor positions where MJ score indicates strong complement.

**Parameters:**
- `per_pos`: Per-position scores from `score_aligned()`
- `thr`: Score threshold (default: -25.0)

**Returns:** List of 1-based position indices qualifying as anchors

**Example:**
```python
per_pos = [-30.0, -10.0, None, -28.0, -5.0]
anchors = anchors_by_threshold(per_pos, thr=-25.0)
# Returns [1, 4] - positions 1 and 4 have scores <= -25
```

### `anchor_score(per_pos, anchors) -> tuple[float, list[tuple[int, float]]]`

Sums MJ scores over specified anchor positions.

**Returns:**
- Total anchor score
- List of (position, score) pairs for positions used

### `top_contributors(seq1_aln, seq2_aln, per_pos, n=10) -> list[tuple[float, int, str, str]]`

Finds the n most favorable (most negative) aligned positions.

**Returns:** List of (score, 1-based_position, residue1, residue2) tuples, sorted by score (most negative first).

## Implementation Notes

1. **MJ Overrides**: The `apply_mj_overrides()` function is called on all lookups. Currently implements:
   - Aromatic (W/F/Y) + small neutral (A/G): returns -8.0

2. **Unknown Residue Handling**: Three policies available:
   - `error`: Raise ValueError on unknown residues
   - `skip`: Return None for that position
   - `zero`: Treat unknown pairs as score 0.0

3. **Gap Handling**: Gaps can be:
   - Ignored (gap_penalty=None)
   - Penalized uniformly (gap_penalty=float)
   - Penalized with affine model (gap_open + gap_ext)

## Dependencies

- `amino_acid_properties`: Constants and override logic
