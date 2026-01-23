# dynamic_programming.py - Gapped Alignment Extension (Phase 2)

## Purpose

This module implements the second phase of alignment: extending ungapped seeds with gaps using dynamic programming. It takes seed windows from Phase 1 and produces optimal gapped alignments.

## Key Concepts

### Band-Constrained DP

Rather than filling the entire DP matrix, the algorithm restricts computation to a diagonal band. This:
- Reduces time complexity from O(n²) to O(n × band_width)
- Reflects the biological expectation that good alignments stay near the diagonal

### Affine Gap Penalties

Uses separate penalties for gap opening and extension:
- `gap_open`: Cost to start a new gap
- `gap_ext`: Cost to extend an existing gap by one position

This better models biology where insertions/deletions tend to cluster.

### Proline Run Handling

Proline (P) has unique conformational properties. The algorithm can:
- Force gaps to cover proline runs in one sequence
- Insert gaps post-hoc to ensure proline run coverage

## Proline Utilities

### `proline_run_mask(seq: str) -> list[bool]`

Creates a mask for positions in proline runs (PP+).

**Example:**
```python
mask = proline_run_mask("ACPPDEF")
# Returns [False, False, True, True, False, False, False]
# Positions 2-3 (0-indexed) are in a PP run
```

### `proline_run_ids(seq: str) -> list[int]`

Assigns unique run IDs to positions in proline runs.

**Example:**
```python
ids = proline_run_ids("APPBPPC")
# Returns [-1, 0, 0, -1, 1, 1, -1]
# Two separate PP runs get IDs 0 and 1
```

### `alignment_gaps_cover_all_proline_runs_seq2(aln1, aln2, ...) -> tuple[bool, int]`

Checks if alignment gaps cover all proline runs in seq2.

**Returns:** `(all_covered, gap_count)` where:
- `all_covered`: True if every PP+ run has at least one gap
- `gap_count`: Number of gaps that hit proline runs

### `force_gap_per_proline_run_seq2(aln1, aln2, ...) -> tuple[str, str, bool]`

Modifies alignment to insert gaps in aln1 at proline runs in seq2.

**Returns:** `(modified_aln1, modified_aln2, was_changed)`

## Main DP Functions

### `stage2_extend_fixed_core(...)`

Extends a fixed seed using bidirectional DP.

**Parameters:**
- `s1`, `s2`: Full sequences
- `mj`: MJ matrix dictionary
- `anchor_i`, `anchor_j`: Start of seed in each sequence (0-based)
- `seed_len`: Length of fixed seed core
- `min_len`, `max_len`: Alignment length constraints
- `band`: Band width for diagonal constraint
- `switch_pen`: Penalty for changing alignment trajectory
- `gap_open`, `gap_ext`: Gap penalties
- `gap_proline_force_runs_seq2`: Require gaps at proline runs
- `gap_proline_force_post`: Insert gaps post-hoc
- `max_gaps`: Maximum number of gap openings
- `max_gap_len`: Maximum length of any single gap
- `unknown_policy`: How to handle unknown residues
- `context_bonus`: Apply context bonuses

**Returns:** `(score, aln1, aln2, start_positions)` or all None if failed

**Algorithm:**
1. Score the fixed seed core
2. Build DP tables for left extension (reversed sequences)
3. Build DP tables for right extension
4. Find best combination of left/right extensions
5. Reconstruct alignment via backtracing
6. Optionally enforce/check proline constraints

### `stage2_best_from_seed(...)`

Wrapper that extracts flanked subsequences and runs Stage 2 DP.

**Additional Parameters:**
- `seed_i`, `seed_j`: Seed position in full sequences
- `flank`: Extra positions to include on each side
- `reanchor`: If True, try all anchor positions in flanked region

**Returns:** `(score, aln1, aln2, start_positions, anchor_positions)`

### `stage2_null_scores(...)`

Generates null distribution for Stage 2 alignment scores.

**Parameters:**
- `n`: Number of null samples
- `seed`: Random seed
- `reanchor`: Try all anchor positions for each shuffle

**Algorithm:**
1. Extract flanked subsequences
2. For each null sample:
   - Shuffle both subsequences
   - Run Stage 2 DP
   - Record best score

**Returns:** List of scores from null alignments

## DP State Machine

The internal DP uses states:
- `S`: Start state
- `M`: Match state (both sequences advance)
- `G1`: Gap in sequence 1 (sequence 2 advances)
- `G2`: Gap in sequence 2 (sequence 1 advances)

Transitions track:
- Position in both sequences
- Number of gaps opened
- Current state
- Current gap length

## Example Usage

```python
from mj_align import stage2_best_from_seed, load_mj_csv

mj = load_mj_csv("mj_matrix.csv")

score, aln1, aln2, start, anchor = stage2_best_from_seed(
    seq1, seq2, mj,
    seed_i=50, seed_j=30,      # Seed from Phase 1
    seed_len=8,                 # Core window length
    flank=10,                   # Extend 10 residues each side
    min_len=15, max_len=25,     # Alignment length bounds
    band=3,                     # Diagonal band width
    switch_pen=2.0,             # Trajectory change penalty
    gap_open=5.0, gap_ext=1.0,  # Affine gap costs
    max_gaps=2, max_gap_len=3,  # Gap constraints
    unknown_policy="skip",
)

if score is not None:
    print(f"Score: {score}")
    print(f"Alignment:\n{aln1}\n{aln2}")
```

## Dependencies

- `scoring`: _mj_pair_score, score_aligned_with_gaps
