# formatting.py - Output Formatting Utilities

## Purpose

This module provides helper functions for formatting floating-point numbers, percentages, and probabilities in a clean, readable format suitable for command-line output and reports.

## Key Functions

### `fmt_float(x: float, nd: int = 2) -> str`

Formats a float with rounding, removing unnecessary trailing zeros.

**Parameters:**
- `x`: The number to format
- `nd`: Number of decimal places (default: 2)

**Examples:**
```python
fmt_float(3.14159, 2)  # "3.14"
fmt_float(5.00, 2)     # "5"
fmt_float(2.50, 2)     # "2.5"
fmt_float(0.001, 4)    # "0.001"
```

### `fmt_pct(x: float, nd: int = 4) -> str`

Formats a proportion in [0,1] with trimmed trailing zeros.

**Parameters:**
- `x`: Proportion value (typically 0 to 1)
- `nd`: Number of decimal places (default: 4)

**Examples:**
```python
fmt_pct(0.5000, 4)   # "0.5"
fmt_pct(0.12345, 4)  # "0.1235"
fmt_pct(1.0, 4)      # "1"
fmt_pct(0.0001, 4)   # "0.0001"
```

### `fmt_prob(p: float) -> str`

Formats a probability or p-value in a compact, readable form.

**Special handling:**
- Very small values (< 1e-4): Scientific notation
- Zero: Returns "0"
- NaN: Returns "nan"

**Examples:**
```python
fmt_prob(0.05)     # "0.05"
fmt_prob(0.00001)  # "1.000e-05"
fmt_prob(0.0)      # "0"
fmt_prob(float('nan'))  # "nan"
```

## Design Rationale

### Why strip trailing zeros?

Cleaner output that's easier to read:
- `5` is cleaner than `5.00`
- `2.5` is cleaner than `2.50`

### Why scientific notation for small p-values?

- `1.000e-05` is more readable than `0.00001`
- Clearly communicates order of magnitude
- Standard practice in statistical reporting

### Why separate functions?

Different contexts call for different precision:
- `fmt_float`: General numbers (scores, coordinates)
- `fmt_pct`: Proportions/percentages (similarity scores)
- `fmt_prob`: Statistical p-values (need to distinguish very small values)

## Usage in Reports

```python
from mj_align import fmt_float, fmt_pct, fmt_prob

# Alignment score report
print(f"Total MJ score: {fmt_float(total, 2)}")
print(f"Mean per-position: {fmt_float(total/length, 3)}")

# Similarity report
print(f"Identity: {fmt_pct(identity_ratio, 4)}")
print(f"Normalized similarity: {fmt_pct(norm_score, 4)}")

# Statistical report
print(f"Empirical p-value: {fmt_prob(p_emp)}")
print(f"Analytic p-value: {fmt_prob(p_analytic)}")
```

## Dependencies

None (uses only Python string formatting)
