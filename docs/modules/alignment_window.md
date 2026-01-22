# alignment_window.py - Ungapped Window-Based Alignment Search (Phase 1)

## Purpose

This module implements the first phase of alignment discovery: finding the best ungapped window pairs between two protein sequences. These "seeds" identify promising starting points for subsequent gapped extension.

## Key Functions

### `best_ungapped_window_pair(seq1, seq2, mj, window, ...)`

Finds the best ungapped window pair using MJ scoring.

**Parameters:**
- `seq1`, `seq2`: Input sequences
- `mj`: MJ matrix dictionary
- `window`: Window length
- `mode`: 'min' for most negative (strongest complement) or 'max'
- `max_evals`: If >0 and less than total pairs, use random sampling
- `rng_seed`: Random seed for sampling
- `unknown_policy`: How to handle unknown residues
- `context_bonus`: Apply context bonuses

**Returns:** `(score, start_i, start_j, per_pos_scores)`

**Algorithm:**
1. For each window position (i, j), sum MJ scores for all aligned positions
2. Track the best (most negative) scoring window
3. Optionally use random sampling for large sequence pairs

**Example:**
```python
score, i, j, per_pos = best_ungapped_window_pair(seq1, seq2, mj, window=10)
print(f"Best window at ({i}, {j}) with score {score}")
```

### `best_ungapped_window_pair_clustal(seq1, seq2, *, window, ...)`

Finds best ungapped window using Clustal similarity scoring instead of MJ.

**Additional Parameters:**
- `prefilter_min_strong`: Minimum strong similarities required
- `prefilter_min_identity`: Minimum identities required
- `kmer_len`: K-mer length for prefiltering (0 = disabled)
- `kmer_min`: Minimum k-mer matches required
- `require_positions2`: Position constraints as (offset, allowed_set) tuples
- `rank_by`: 'score' or 'identity' for ranking
- `filter_charge_runs`: Skip windows containing charge runs

**Returns:** `(score, start_i, start_j, identity_count)`

This is useful for finding regions with sequence similarity that might also have complementarity.

### `best_ungapped_window_pair_clustal_topk(..., topk=3)`

Returns top-K best windows instead of just the single best.

**Returns:** List of `(score, identity_count, start_i, start_j)` tuples, sorted by ranking criterion.

### `seed_windows(seq1, seq2, mj, *, window, score_max, kmax, kmin, ...)`

Generates a filtered list of seed windows for alignment.

**Parameters:**
- `window`: Window length for scoring
- `score_max`: Maximum score threshold (more negative = stricter filter)
- `kmax`: Maximum number of seeds to return (0 = unlimited)
- `kmin`: Minimum seeds to keep even if they don't pass threshold
- `prefilter_len`: Shorter window for prefiltering (0 = disabled)
- `prefilter_score_max`: Score threshold for prefiltering
- `prefilter_kmax`, `prefilter_kmin`: Prefilter candidate limits

**Returns:** List of `(score, start_i, start_j)` tuples sorted by score (most negative first)

**Algorithm:**
1. Optionally prefilter with shorter windows for efficiency
2. Score all (or filtered) window positions
3. Filter by score threshold
4. Return top candidates

**Example:**
```python
seeds = seed_windows(seq1, seq2, mj, window=10, score_max=-100, kmax=5, kmin=1)
for score, i, j in seeds:
    print(f"Seed at ({i}, {j}) with score {score}")
```

### `overlaps_fraction(a_start, a_end, b_start, b_end) -> float`

Computes overlap fraction between two intervals.

**Returns:** Overlap length divided by the length of the shorter interval.

**Example:**
```python
# Intervals [0,10] and [5,15] overlap at [5,10]
# Overlap = 6, shorter length = 11, fraction = 6/11 ≈ 0.545
frac = overlaps_fraction(0, 10, 5, 15)
```

## Performance Optimization

### Random Sampling

For large sequence pairs where exhaustive search is impractical, random sampling can be used:

```python
# Sample 10000 random window pairs instead of exhaustive search
score, i, j, _ = best_ungapped_window_pair(
    seq1, seq2, mj, window=10, max_evals=10000, rng_seed=42
)
```

### NumPy Acceleration

The `seed_windows` function automatically uses NumPy when available for faster computation. This provides significant speedup for large sequences.

### K-mer Prefiltering

For Clustal-based searches, k-mer prefiltering can dramatically reduce computation:

```python
# Only evaluate windows where at least 2 3-mers match
score, i, j, ident = best_ungapped_window_pair_clustal(
    seq1, seq2, window=10, kmer_len=3, kmer_min=2
)
```

## Null Distribution Generation

### `seed_null_best_scores(seq1, seq2, mj, *, window, n, ...)`

Generates a null distribution for best window scores by shuffling one sequence.

**Parameters:**
- `n`: Number of null samples
- `seed`: Random seed

**Returns:** List of best window scores from shuffled sequences

This is used to assess statistical significance of observed window scores.

## Dependencies

- `amino_acid_properties`: AA20, has_charge_run
- `clustering`: clustal_pair_score, _kmer_hits
- `scoring`: context_bonus_aligned, get_mj_scorer
