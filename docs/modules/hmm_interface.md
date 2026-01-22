# hmm_interface.py - Hidden Markov Model for Interface Footprinting

## Purpose

This module implements a three-state Hidden Markov Model (HMM) for identifying protein-protein interface regions based on MJ interaction scores. It segments alignments into background, peripheral, and core interface regions.

## The Three-State Model

| State | Symbol | Description | Expected MJ Score |
|-------|--------|-------------|-------------------|
| Background | B | Non-interface regions | Any (weak penalty) |
| Peripheral | P | Moderate complementarity | score <= tp |
| Core | C | Strong complementarity | score <= tc |

Where `tc` (core threshold) < `tp` (peripheral threshold) < 0.

### Biological Interpretation

- **Core (C)**: Interface hotspots with the strongest complementary interactions. These positions likely contribute most to binding affinity.
- **Peripheral (P)**: Interface edges with moderate interactions. These positions may contribute to specificity or secondary contacts.
- **Background (B)**: Non-interacting regions or positions with no particular complementarity.

## Transition Model

The fixed transition probabilities reflect biological expectations:

```
From B: B→B: 0.985, B→P: 0.014, B→C: 0.001
From P: P→B: 0.12,  P→P: 0.85,  P→C: 0.03
From C: C→B: 0.01,  C→P: 0.07,  C→C: 0.92
```

**Interpretation:**
- Background tends to stay in background
- Once in peripheral/core, tends to persist (interface regions are contiguous)
- Core rarely jumps directly to background
- Peripheral can transition to either background or core

## Emission Model

### `hmm_emission(state: str, s: float, tc: float, tp: float) -> float`

Computes emission score for a given state and MJ score.

**Logic:**
```python
# Penalize very positive (unfavorable) MJ scores
if s >= 30: penalty = -6.0
elif s >= 20: penalty = -3.0
else: penalty = 0.0

# State-specific scoring
if state == "C":  # Core
    if s <= tc:
        score = 4.0 + 0.1 * (tc - s)  # Bonus for strong complement
    else:
        score = -8.0  # Heavy penalty for weak scores in core

elif state == "P":  # Peripheral
    if s <= tp:
        score = 2.0 + 0.05 * (tp - s)  # Moderate bonus
    else:
        score = -2.0  # Light penalty

else:  # Background
    if s <= tp:
        score = 0.5  # Mild bonus for any reasonable score
```

## Key Functions

### `hmm_interface_path(per_pos, *, tc, tp) -> list[Optional[str]]`

Finds most likely B/P/C state path using Viterbi algorithm.

**Parameters:**
- `per_pos`: Per-position MJ scores (None for gaps)
- `tc`: Core threshold (e.g., -25.0)
- `tp`: Peripheral threshold (e.g., -15.0)

**Returns:** List of state labels ('B', 'P', 'C') with None for gaps

**Example:**
```python
per_pos = [-30.0, -15.0, -5.0, None, -28.0, -10.0]
path = hmm_interface_path(per_pos, tc=-25.0, tp=-15.0)
# Returns ['C', 'P', 'B', None, 'C', 'P']
```

### `hmm_segments(path) -> list[tuple[str, int, int]]`

Extracts contiguous state segments from an HMM path.

**Returns:** List of `(state, start_pos, end_pos)` tuples using 1-based indices

**Example:**
```python
path = ['B', 'B', 'P', 'P', 'C', 'C', 'C', None, 'B']
segments = hmm_segments(path)
# Returns [('B', 1, 2), ('P', 3, 4), ('C', 5, 7), ('B', 9, 9)]
```

## Viterbi Algorithm

The implementation uses standard Viterbi decoding:

1. **Initialization**: Start in B state with probability 1.0
2. **Forward pass**: For each position, compute:
   - Best score to reach each state from any previous state
   - Store backpointers for traceback
3. **Termination**: Select state with highest final score
4. **Traceback**: Follow backpointers to recover optimal path

## Example Workflow

```python
from mj_align import (
    score_aligned, hmm_interface_path, hmm_segments
)

# Score alignment
total, per_pos = score_aligned(seq1, seq2, mj)

# Run HMM
path = hmm_interface_path(per_pos, tc=-25.0, tp=-15.0)

# Extract segments
segments = hmm_segments(path)

# Report interface regions
print("Interface analysis:")
for state, start, end in segments:
    if state == 'C':
        print(f"  CORE: positions {start}-{end}")
    elif state == 'P':
        print(f"  Peripheral: positions {start}-{end}")
```

## Choosing Thresholds

**Core threshold (tc):** Should capture strong, unambiguous complementarity
- Default: -25.0
- More negative = stricter (fewer core positions)

**Peripheral threshold (tp):** Should capture moderate complementarity
- Default: -15.0
- Should be less negative than tc

**Guidelines:**
- Based on MJ matrix statistics, -25 roughly corresponds to top 5% of favorable scores
- Adjust based on your specific application and expected interface size

## Limitations

1. **Fixed transition model**: The transition probabilities are hard-coded and may not be optimal for all applications.

2. **Position independence**: Emission scores don't consider neighboring positions (though transitions do capture some dependency).

3. **Two-sequence only**: Designed for pairwise alignments, not multiple sequence alignments.

4. **Linear segmentation**: Assumes interface regions are contiguous; doesn't handle complex topologies.

## Dependencies

None (uses only Python standard library)
