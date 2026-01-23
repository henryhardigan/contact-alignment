# cli.py - Command-Line Interface

## Purpose

This module provides the CLI entry point for the mj-align package. It offers a simplified interface for basic scoring operations. For full functionality with all options, the original `mj_score.py` can be used directly.

## Basic Usage

```bash
# Score two sequences directly
mj-score --mj mj_matrix.csv --seq1 ACDEFGHIK --seq2 FGHIKACDE

# From FASTA files
mj-score --mj mj_matrix.csv --fasta1 protein1.fasta --fasta2 protein2.fasta

# With specific FASTA entries
mj-score --mj mj_matrix.csv \
    --fasta1 proteins.fasta --fasta1-entry P12345 \
    --fasta2 proteins.fasta --fasta2-entry Q67890
```

## Command-Line Options

### Required/Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mj` | Path to MJ matrix CSV | `mj_matrix.csv` |
| `--seq1` | First aligned sequence | - |
| `--seq2` | Second aligned sequence | - |

### FASTA Input Options

| Option | Description |
|--------|-------------|
| `--fasta1` | FASTA file for sequence 1 |
| `--fasta2` | FASTA file for sequence 2 |
| `--fasta1-entry` | Entry filter for fasta1 |
| `--fasta2-entry` | Entry filter for fasta2 |
| `--name1` | Display name for sequence 1 |
| `--name2` | Display name for sequence 2 |

### Analysis Options

| Option | Description | Default |
|--------|-------------|---------|
| `--thr` | Anchor threshold (MJ <= thr) | `-25.0` |
| `--top` | Number of top contributors to show | `10` |
| `--null` | Number of null distribution samples | `0` (disabled) |
| `--clustal` | Show Clustal-style similarity | `false` |
| `--unknown` | Unknown residue handling | `error` |

### Unknown Residue Policies

- `error`: Raise error on unknown residues
- `skip`: Skip positions with unknown residues
- `zero`: Treat unknown pairs as score 0

## Output Format

```
=== MJ Score Results ===

Sequence 1: protein1
Sequence 2: protein2
Alignment length: 100

Total MJ score: -250.5
Anchors (MJ <= -25): 15 positions
Anchor positions: [3, 7, 12, 15, ...]

Top 10 contributing positions:
  Position 7: W-A = -35.2
  Position 12: K-E = -32.1
  ...

Clustal similarity:
  Score: 45.5 / 100 = 0.455
  ACDEF...
  *:.. ...
  ACDKF...

Null distribution (1000 shuffles)...
  Mean: -150.2
  5th percentile: -180.5
  Observed: -250.5
  P-value: 0.002
```

## Programmatic Usage

The CLI can also be called programmatically:

```python
from mj_align.cli import main

# With argument list
exit_code = main(["--mj", "matrix.csv", "--seq1", "ACDEF", "--seq2", "ACDEK"])

# Returns 0 on success, 1 on error
```

## Full CLI (mj_score.py)

For advanced features not available in the simplified CLI, use `mj_score.py` directly:

```bash
python mj_score.py --help
```

Additional features in the full CLI include:
- Two-phase alignment (ungapped seeds + gapped extension)
- HMM interface footprinting
- ScanProsite motif generation
- Advanced statistical analysis
- Multiple null distribution methods
- Window scanning modes
- Proline constraint handling

## Implementation

The CLI module imports from the modularized package:

```python
from .scoring import load_mj_csv, score_aligned, anchors_by_threshold, top_contributors
from .fasta_io import read_fasta_all, read_fasta_entry
from .clustering import clustal_similarity
from .statistics import null_distribution, quantile
from .formatting import fmt_float, fmt_pct, fmt_prob
```

## Dependencies

- All other mj_align modules
- Python argparse (standard library)
