# amino_acid_properties.py - Amino Acid Constants and Classifications

## Purpose

This module provides fundamental constants and utility functions for working with protein sequences. It defines the standard amino acid set, chemical property classifications, and Clustal similarity groups used throughout the package.

## Constants

### Standard Amino Acids

```python
AA20_STR = "ACDEFGHIKLMNPQRSTVWY"  # String form, canonical order
AA20 = set(AA20_STR)               # Set form for membership testing
```

### Chemical Property Classifications

```python
AROMATICS = {"W", "F", "Y"}       # Tryptophan, Phenylalanine, Tyrosine
SMALL_NEUTRAL = {"A", "G"}        # Alanine, Glycine
POS_CHARGES = {"K", "R"}          # Lysine, Arginine (positive at pH 7)
NEG_CHARGES = {"D", "E"}          # Aspartate, Glutamate (negative at pH 7)
HYDROPHOBES = {"A", "I", "L", "M", "V", "F", "W", "Y"}  # Hydrophobic residues
```

### Scoring Weights

```python
HYDROPHOBE_OFFSET_WEIGHT = 0.6    # Weight for hydrophobe context bonuses
```

### Clustal Similarity Groups

**Strong groups** (marked ':' in Clustal alignments):
```python
CLUSTAL_STRONG_GROUPS = [
    set("STA"),    # Small hydroxyl/sulfhydryl
    set("NEQK"),   # Amide/charged
    set("NHQK"),   # Amide/basic
    set("NDEQ"),   # Acidic/amide
    set("QHRK"),   # Basic/amide
    set("MILV"),   # Aliphatic
    set("MILF"),   # Aliphatic/aromatic
    set("HY"),     # Aromatic/basic
    set("FYW"),    # Aromatic
]
```

**Weak groups** (marked '.' in Clustal alignments):
```python
CLUSTAL_WEAK_GROUPS = [
    set("CSA"),    # Small
    set("ATV"),    # Small aliphatic
    set("SAG"),    # Small
    set("STNK"),   # Polar/basic
    set("STPA"),   # Small/polar
    set("SGND"),   # Small/acidic
    set("SNDEQK"), # Polar
    set("NDEQHK"), # Polar/charged
    set("NEQHRK"), # Polar/basic
    set("FVLIM"),  # Hydrophobic
    set("HFY"),    # Aromatic
]
```

## Functions

### `apply_mj_overrides(a: str, b: str, val: Optional[float]) -> Optional[float]`

Applies custom MJ score overrides for special residue pairs.

**Current rules:**
- Aromatic (W/F/Y) + small neutral (A/G): returns -8.0 (favorable interaction)

This reflects the empirical observation that aromatic residues often interact favorably with small neutral residues at interfaces.

**Example:**
```python
apply_mj_overrides("W", "A", -5.0)   # Returns -8.0 (override applied)
apply_mj_overrides("K", "E", -10.0)  # Returns -10.0 (no override)
```

### `has_charge_run(seq: str, run_len: int = 3) -> bool`

Detects runs of charged residues in a protein sequence.

**Parameters:**
- `seq`: Protein sequence
- `run_len`: Minimum consecutive charged residues to detect (default: 3)

**Returns:** True if sequence contains a run of positive (K/R) or negative (D/E) charges of at least `run_len` consecutive residues.

**Example:**
```python
has_charge_run("ACKKKLM", run_len=3)  # True (has KKK)
has_charge_run("ACKKLM", run_len=3)   # False (KK is only 2)
has_charge_run("ACDDDLM", run_len=3)  # True (has DDD)
```

## Biological Rationale

### Why These Classifications?

1. **Aromatics**: Large planar rings enable pi-stacking and cation-pi interactions
2. **Charges**: Electrostatic interactions are long-range and strong
3. **Hydrophobes**: Drive protein folding and interface formation via the hydrophobic effect
4. **Small neutrals**: Provide conformational flexibility at interfaces

### Clustal Groups

The Clustal similarity groups are derived from:
- Physicochemical properties (size, charge, hydrophobicity)
- Observed substitution patterns in protein evolution
- Structural interchangeability in folded proteins

Strong groups have higher evolutionary and structural interchangeability than weak groups.

## Usage

```python
from mj_align import AA20, AROMATICS, POS_CHARGES, has_charge_run

# Check if residue is standard
if residue in AA20:
    print("Standard amino acid")

# Check chemical properties
if residue in AROMATICS:
    print("Aromatic residue")

# Detect charge clusters
if has_charge_run(sequence):
    print("Contains charge run - may be a low complexity region")
```
