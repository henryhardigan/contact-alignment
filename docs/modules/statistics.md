# statistics.py - Statistical Analysis and Null Distribution Generation

## Purpose

This module provides functions for assessing the statistical significance of alignment scores. It implements both empirical (shuffling, circular shifts) and analytic (Poisson-binomial) approaches to null distribution generation.

## Empirical Null Distributions

### `shuffle_preserve_gaps(seq: str, gap_char='-') -> str`

Shuffles residues while keeping gap positions fixed.

**Use case:** Generate null alignments that preserve the gap structure of the original alignment.

**Example:**
```python
import random
random.seed(42)
shuffled = shuffle_preserve_gaps("A-BC--D")
# Returns something like "D-CB--A" with gaps at positions 2, 5, 6 preserved
```

### `null_distribution(fixed_seq, shuffle_seq, mj, n_iter=1000, ...) -> list[float]`

Generates null distribution by shuffling one sequence.

**Parameters:**
- `fixed_seq`: Sequence held constant
- `shuffle_seq`: Sequence shuffled each iteration
- `mj`: MJ matrix dictionary
- `n_iter`: Number of shuffle iterations (default: 1000)
- `gap_char`, `gap_penalty`, `unknown_policy`, `context_bonus`: Scoring parameters

**Returns:** List of total scores from each shuffled alignment

**Example:**
```python
null_scores = null_distribution(seq1, seq2, mj, n_iter=1000)
p_value = sum(1 for s in null_scores if s <= observed) / len(null_scores)
```

### `circular_shift(seq: str, k: int) -> str`

Circularly shifts a sequence by k positions.

**Example:**
```python
circular_shift("ABCD", 1)  # Returns "DABC"
circular_shift("ABCD", 3)  # Returns "BCDA"
```

### `circular_shift_null(fixed_seq, shift_seq, mj, ...) -> list[float]`

Generates null distribution using circular shifts.

**Advantages over shuffling:**
- Preserves sequence composition exactly
- Preserves local sequence structure (adjacent residue pairs)

**Parameters:**
- `n_samples`: If provided and < (L-1), randomly sample this many shifts
- `seed`: Random seed for sampling

The observed alignment (shift k=0) is excluded from the null.

## Quantile Calculation

### `quantile(values: list[float], q: float) -> float`

Computes quantile using linear interpolation.

**Example:**
```python
values = [1, 2, 3, 4, 5]
quantile(values, 0.5)   # Returns 3.0 (median)
quantile(values, 0.25)  # Returns 2.0
```

## Window Score Analysis

### `best_window_score(per_pos, window, *, mode='min', none_as=0.0) -> tuple[Optional[float], Optional[int]]`

Finds the best contiguous window score over per-position values.

**Parameters:**
- `per_pos`: Per-position scores (None for gaps)
- `window`: Window size
- `mode`: 'min' for most negative, 'max' for most positive
- `none_as`: Value to use for None positions (None = skip windows with gaps)

**Returns:** `(best_score, 1-based_start_position)`

**Example:**
```python
per_pos = [-5.0, -10.0, -15.0, -8.0, -3.0]
score, start = best_window_score(per_pos, 3, mode="min")
# Returns (-33.0, 2) - positions 2-4 sum to -33
```

### `scan_null_best_window(fixed_seq, other_seq, mj, *, window, null_method, n, shuffle_which, ...)`

Generates scan-aware null distribution over best window scores.

**Why "scan-aware"?**

When scanning for the best window, we're implicitly doing multiple tests. This function accounts for that by:
1. Generating a null alignment (shuffle or shift)
2. Finding the best window in that null
3. Repeating to build a distribution of "best-of" scores

## Set Overlap Statistics

### `jaccard(a, b) -> float`

Computes Jaccard index (intersection over union) for two sets.

**Example:**
```python
jaccard([1, 2, 3], [2, 3, 4])  # Returns 0.5 (intersection={2,3}, union={1,2,3,4})
```

### `hypergeom_p_at_least(k, N, K, n) -> float`

Computes P(X >= k) for hypergeometric distribution.

**Parameters:**
- `k`: Observed overlap (successes drawn)
- `N`: Population size (total eligible positions)
- `K`: Number of successes in population (anchors in set A)
- `n`: Number of draws (anchors in set B)

**Use case:** Testing if anchor overlap between two analyses exceeds chance.

**Example:**
```python
# 5 positions, set A has 2 anchors, set B has 2 anchors
# P(overlap >= 1) under random placement
p = hypergeom_p_at_least(k=1, N=5, K=2, n=2)
# Returns 0.7
```

## Analytic Null (Model C)

### `poisson_binomial_p_ge(ps: list[float], x: int) -> float`

Computes P(X >= x) for Poisson-binomial distribution via DP.

The Poisson-binomial distribution is the sum of independent Bernoulli random variables with potentially different success probabilities.

**Use case:** When each position has a different probability of being an anchor, compute the probability of observing at least x anchors.

**Example:**
```python
# Three positions with complement probabilities 0.3, 0.5, 0.7
p = poisson_binomial_p_ge([0.3, 0.5, 0.7], x=2)
# Returns ~0.59 (P(at least 2 successes))
```

### `per_position_complement_prob_uniform(fixed_seq_aln, mj, *, thr, ...)`

Computes per-position probability of complement under uniform amino acid distribution.

**Model:** For each position, calculate P(MJ(fixed[i], B) <= thr) where B is uniformly distributed over the 20 amino acids.

**Returns:** List of per-position probabilities (None for gaps/unknown)

### `modelc_uniform_anchor_stats(seq1_aln, seq2_aln, mj, *, thr, ...)`

Computes analytic anchor statistics under uniform partner model (Model C).

**Returns:** `(x_obs, ex_seq1_fixed, p_seq1_fixed, ex_seq2_fixed, p_seq2_fixed)`
- `x_obs`: Observed number of anchor positions
- `ex_seq1_fixed`: E[X] with seq1 fixed, partner uniform
- `p_seq1_fixed`: P(X >= x_obs | seq1 fixed)
- `ex_seq2_fixed`: E[X] with seq2 fixed, partner uniform
- `p_seq2_fixed`: P(X >= x_obs | seq2 fixed)

This provides p-values from both directions (conditioning on either sequence).

## Example Workflow

```python
from mj_align import (
    score_aligned, anchors_by_threshold, null_distribution,
    quantile, modelc_uniform_anchor_stats
)

# Score alignment
total, per_pos = score_aligned(seq1, seq2, mj)
anchors = anchors_by_threshold(per_pos, thr=-25.0)
print(f"Observed: {total}, {len(anchors)} anchors")

# Empirical null
null_scores = null_distribution(seq1, seq2, mj, n_iter=1000)
p_emp = sum(1 for s in null_scores if s <= total) / len(null_scores)
print(f"Empirical p-value: {p_emp}")
print(f"5th percentile null: {quantile(null_scores, 0.05)}")

# Analytic null (Model C)
x_obs, ex1, p1, ex2, p2 = modelc_uniform_anchor_stats(seq1, seq2, mj, thr=-25.0)
print(f"Analytic: {x_obs} anchors, E[X]={ex1:.1f}, p={p1:.4f}")
```

## Dependencies

- `amino_acid_properties`: AA20
- `scoring`: score_aligned
