# MJ-Align: Linear Complement Alignment for Protein Interaction Assessment

## Thesis

MJ-Align is a computational tool for assessing protein-protein interaction potential by scoring the **residue-residue complementarity** between aligned short linear sequences. Rather than looking for sequence *similarity* (as traditional alignment tools do), this software looks for sequence *complementarity* - identifying residue pairs that are energetically favorable for physical contact.

The core insight is that protein-protein interactions often occur through short linear motifs (SLiMs) where specific residue types on one protein surface complement residue types on the partner protein. By quantifying this complementarity using **Miyazawa-Jernigan (MJ) contact potentials**, we can predict whether two sequence regions might form a favorable interaction interface.

### Key Concept: Complementarity vs. Similarity

Traditional sequence alignment asks: "How similar are these sequences?"

MJ-Align asks: "If these sequences were aligned face-to-face at an interaction interface, how energetically favorable would the residue-residue contacts be?"

This is analogous to asking whether two puzzle pieces fit together, rather than whether they look alike.

## The Miyazawa-Jernigan Contact Potential

The MJ matrix is a 20x20 matrix of statistical contact potentials derived from observed amino acid contacts in known protein structures (Miyazawa & Jernigan, 1996). Each entry MJ(A,B) represents the free energy contribution of a contact between residue types A and B.

- **More negative values** indicate favorable (attractive) interactions
- **Positive values** indicate unfavorable (repulsive) interactions
- The matrix is symmetric: MJ(A,B) = MJ(B,A)

For example:
- Hydrophobic residues (W, F, Y, L, I, V) tend to have favorable contacts with each other
- Oppositely charged residues (K/R with D/E) have favorable electrostatic interactions
- Like charges repel

## How the Algorithm Works

### Phase 1: Ungapped Window Search (Seed Finding)

The algorithm first identifies promising "seed" regions using an ungapped sliding window approach:

1. **Window Scanning**: For each possible window position in both sequences, compute the total MJ score for the aligned residue pairs
2. **Ranking**: Windows are ranked by score (most negative = best complementarity)
3. **Filtering**: Optional filters include:
   - Clustal-style similarity prefiltering (k-mer matching)
   - Minimum identity/similarity requirements
   - Charge run avoidance

This phase efficiently identifies candidate regions that may represent interaction hotspots.

### Phase 2: Gapped Alignment Extension (Dynamic Programming)

Promising seeds from Phase 1 are extended using dynamic programming with gaps:

1. **Bidirectional Extension**: From the fixed seed core, extend left and right using band-constrained DP
2. **Affine Gap Penalties**: Separate penalties for gap opening and extension
3. **Proline Constraints**: Optional enforcement that proline-rich regions in one sequence be gapped (reflecting structural rigidity)
4. **Trajectory Penalties**: Discourage zigzag alignments that would be structurally implausible

The result is an optimal gapped alignment that maximizes complementarity.

### Anchor Detection

Positions where MJ score <= threshold (default: -25) are marked as "anchors" - these represent the strongest complementary contacts and likely interaction hotspots.

### Statistical Significance

Several methods assess whether observed complementarity exceeds random expectation:

1. **Shuffling Null**: Repeatedly shuffle one sequence and re-score to build an empirical null distribution
2. **Circular Shift Null**: Circularly shift one sequence (preserves composition and local structure)
3. **Analytic Null (Model C)**: Compute p-value assuming uniform amino acid distribution using Poisson-binomial statistics

### HMM Interface Footprinting

A three-state Hidden Markov Model segments the alignment into:
- **B (Background)**: Non-interface regions
- **P (Peripheral)**: Moderate complementarity (interface edges)
- **C (Core)**: Strong complementarity (interface hotspots)

This provides a structural interpretation of which regions likely form the interaction core.

## Input/Output

### Input
- **MJ Matrix**: CSV file with 20x20 interaction scores
- **Sequences**: Either as direct strings or from FASTA files
- **Parameters**: Thresholds, window sizes, gap penalties, etc.

### Output
- **Total MJ Score**: Sum of all position scores
- **Per-Position Scores**: Score at each aligned position
- **Anchor Positions**: Positions meeting the complementarity threshold
- **Top Contributors**: Most favorable residue pairs
- **Statistical Significance**: P-values from null distributions
- **HMM Path**: Interface region segmentation
- **ScanProsite Motifs**: Pattern representation of anchor positions

## Algorithm Limitations

### Fundamental Limitations

1. **Linear Sequence Only**: The algorithm considers only primary sequence, not 3D structure. Two residues that are distant in sequence but close in space would not be detected.

2. **Statistical Potentials Are Approximate**: MJ values are derived statistics from known structures, not true free energies. They capture trends but not precise energetics.

3. **No Conformational Flexibility**: The algorithm assumes a fixed alignment. Real proteins are dynamic and may adopt different conformations upon binding.

4. **Context Independence**: Each position is scored independently (except for context bonuses). Cooperative effects spanning multiple residues are not fully captured.

### Practical Limitations

5. **Short Motifs Only**: The method works best for short linear motifs (5-20 residues). Longer interfaces with complex topology may not be well-represented.

6. **No Structural Validation**: High complementarity scores don't guarantee binding - structural compatibility, accessibility, and other factors matter.

7. **Gap Model Simplicity**: Affine gap penalties are a crude approximation of the structural cost of insertions/deletions.

8. **Proline Handling**: While proline runs receive special treatment, the structural implications of proline are complex and may not be fully captured.

### Interpretation Caveats

9. **Complementarity ≠ Binding**: A favorable MJ score is necessary but not sufficient for binding. Many complementary sequences may exist that don't interact.

10. **Threshold Sensitivity**: Anchor detection depends on threshold choice. The default (-25) is empirically reasonable but may need adjustment.

11. **Multiple Testing**: When scanning many positions, significant-looking results may arise by chance. Use proper statistical corrections.

## Verification Examples

The following examples from the test suite demonstrate correct behavior:

### Basic Functionality

```python
# Circular shift preserves sequence composition
from mj_align import circular_shift
assert circular_shift("ABCD", 1) == "DABC"
assert circular_shift("ABCD", 4) == "ABCD"

# Quantile calculation
from mj_align import quantile
values = [1.0, 2.0, 3.0, 4.0, 5.0]
assert quantile(values, 0.5) == 3.0  # Median

# Hypergeometric test for anchor overlap
from mj_align import hypergeom_p_at_least
# 5 positions, set A has 2 anchors, set B has 2 anchors
# P(overlap >= 1) = 0.7
p = hypergeom_p_at_least(k=1, N=5, K=2, n=2)
assert abs(p - 0.7) < 1e-10
```

### Clustal Similarity

```python
from mj_align import clustal_pair_score, clustal_similarity

# Identical residues score 1.0
assert clustal_pair_score("A", "A") == 1.0

# No similarity scores 0.0
assert clustal_pair_score("A", "K") == 0.0

# Full identity alignment
symbols, score, norm, n = clustal_similarity("ACDEF", "ACDEF")
assert symbols == "*****"
assert score == 5.0
assert norm == 1.0
```

### Proline Run Detection

```python
from mj_align import proline_run_mask, has_charge_run

# Consecutive prolines (PP+) are flagged
mask = proline_run_mask("ACPPDEF")
assert mask == [False, False, True, True, False, False, False]

# Charge runs are detected
assert has_charge_run("ACKKKLM", run_len=3) == True  # KKK
assert has_charge_run("ACKKLM", run_len=3) == False  # Only KK
```

### Pattern Parsing

```python
from mj_align import parse_scanprosite_pattern

# x(n) notation expands to n wildcards
tokens = parse_scanprosite_pattern("R-x(2)-K")
assert tokens == ["R", "X", "X", "K"]
```

## Usage Example

```bash
# Basic scoring of two aligned sequences
mj-score --mj mj_matrix.csv --seq1 ACDEFGHIK --seq2 FGHIKACDE

# With null distribution for significance
mj-score --mj mj_matrix.csv --seq1 ACDEFGHIK --seq2 FGHIKACDE --null 1000

# From FASTA files with Clustal similarity display
mj-score --mj mj_matrix.csv --fasta1 protein1.fasta --fasta2 protein2.fasta --clustal
```

## References

- Miyazawa S, Jernigan RL (1996). Residue-residue potentials with a favorable contact pair term and an unfavorable high packing density term. *J Mol Biol* 256:623-644.
- Thompson JD, Higgins DG, Gibson TJ (1994). CLUSTAL W: improving the sensitivity of progressive multiple sequence alignment. *Nucleic Acids Res* 22:4673-4680.
