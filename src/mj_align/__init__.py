"""MJ-Align: Linear complement alignment scoring for protein interaction assessment.

This package provides tools for scoring protein sequence alignments using
Miyazawa-Jernigan (MJ) contact potentials, with support for:

- MJ matrix-based alignment scoring
- Clustal-style similarity analysis
- Ungapped window search (Phase 1)
- Gapped alignment extension via DP (Phase 2)
- Statistical significance testing
- HMM-based interface detection
- ScanProsite motif generation

Quick Start:
    >>> from mj_align import load_mj_csv, score_aligned, anchors_by_threshold
    >>> mj = load_mj_csv("mj_matrix.csv")
    >>> total, per_pos = score_aligned("ACDEF", "ACDEK", mj)
    >>> anchors = anchors_by_threshold(per_pos, thr=-25.0)

Modules:
    - amino_acid_properties: Constants and AA classifications
    - scoring: MJ matrix loading and alignment scoring
    - clustering: Clustal-style similarity scoring
    - motifs: ScanProsite pattern generation
    - alignment_window: Ungapped window search
    - dynamic_programming: Gapped alignment extension
    - statistics: Null distributions and hypothesis testing
    - hmm_interface: HMM-based interface detection
    - fasta_io: FASTA file operations
    - formatting: Output formatting utilities
    - cli: Command-line interface

CLI Usage:
    mj-score --mj matrix.csv --seq1 ACDEF --seq2 FGHIK
"""

__version__ = "0.1.0"

# Core scoring functions
# Window search
from .alignment_window import (
    best_ungapped_window_pair,
    best_ungapped_window_pair_clustal,
    best_ungapped_window_pair_clustal_topk,
    overlaps_fraction,
    seed_windows,
)

# Amino acid properties
from .amino_acid_properties import (
    AA20,
    AA20_STR,
    AROMATICS,
    CLUSTAL_STRONG_GROUPS,
    CLUSTAL_WEAK_GROUPS,
    HYDROPHOBES,
    NEG_CHARGES,
    POS_CHARGES,
    apply_mj_overrides,
    has_charge_run,
)

# Clustal similarity
from .clustering import (
    clustal_anchor_positions,
    clustal_entry_key,
    clustal_pair_score,
    clustal_similarity,
)

# Dynamic programming
from .dynamic_programming import (
    alignment_gaps_cover_all_proline_runs_seq2,
    force_gap_per_proline_run_seq2,
    proline_run_ids,
    proline_run_mask,
    stage2_best_from_seed,
    stage2_extend_fixed_core,
    stage2_null_scores,
)

# FASTA I/O
from .fasta_io import (
    read_fasta_all,
    read_fasta_entry,
)

# Formatting utilities
from .formatting import (
    fmt_float,
    fmt_pct,
    fmt_prob,
)

# HMM interface detection
from .hmm_interface import (
    hmm_emission,
    hmm_interface_path,
    hmm_segments,
)

# Motif generation
from .motifs import (
    avg_mj_score_pattern_to_seq,
    combined_aligned_regex,
    combined_aligned_strong_regex,
    parse_scanprosite_pattern,
    parse_scanprosite_pattern_with_sets,
    scanprosite_complement_motif,
    scanprosite_expected_forms,
    scanprosite_forms_in_fasta,
    scanprosite_motif_from_anchors,
)
from .scoring import (
    anchor_score,
    anchors_by_threshold,
    context_bonus_aligned,
    get_mj_scorer,
    load_mj_csv,
    score_aligned,
    score_aligned_with_gaps,
    top_contributors,
)

# Statistics
from .statistics import (
    best_window_score,
    circular_shift,
    circular_shift_null,
    hypergeom_p_at_least,
    jaccard,
    modelc_uniform_anchor_stats,
    null_distribution,
    poisson_binomial_p_ge,
    quantile,
    shuffle_preserve_gaps,
)

__all__ = [
    # Version
    "__version__",
    # Scoring
    "load_mj_csv",
    "get_mj_scorer",
    "score_aligned",
    "score_aligned_with_gaps",
    "context_bonus_aligned",
    "anchors_by_threshold",
    "anchor_score",
    "top_contributors",
    # Amino acid properties
    "AA20",
    "AA20_STR",
    "AROMATICS",
    "HYDROPHOBES",
    "POS_CHARGES",
    "NEG_CHARGES",
    "CLUSTAL_STRONG_GROUPS",
    "CLUSTAL_WEAK_GROUPS",
    "apply_mj_overrides",
    "has_charge_run",
    # Clustering
    "clustal_similarity",
    "clustal_pair_score",
    "clustal_anchor_positions",
    "clustal_entry_key",
    # FASTA
    "read_fasta_all",
    "read_fasta_entry",
    # Formatting
    "fmt_float",
    "fmt_pct",
    "fmt_prob",
    # Statistics
    "shuffle_preserve_gaps",
    "null_distribution",
    "circular_shift",
    "circular_shift_null",
    "quantile",
    "best_window_score",
    "jaccard",
    "hypergeom_p_at_least",
    "poisson_binomial_p_ge",
    "modelc_uniform_anchor_stats",
    # Window search
    "best_ungapped_window_pair",
    "best_ungapped_window_pair_clustal",
    "best_ungapped_window_pair_clustal_topk",
    "seed_windows",
    "overlaps_fraction",
    # Dynamic programming
    "proline_run_mask",
    "proline_run_ids",
    "alignment_gaps_cover_all_proline_runs_seq2",
    "force_gap_per_proline_run_seq2",
    "stage2_extend_fixed_core",
    "stage2_best_from_seed",
    "stage2_null_scores",
    # HMM
    "hmm_emission",
    "hmm_interface_path",
    "hmm_segments",
    # Motifs
    "scanprosite_motif_from_anchors",
    "combined_aligned_regex",
    "combined_aligned_strong_regex",
    "parse_scanprosite_pattern",
    "parse_scanprosite_pattern_with_sets",
    "scanprosite_forms_in_fasta",
    "scanprosite_expected_forms",
    "scanprosite_complement_motif",
    "avg_mj_score_pattern_to_seq",
]
