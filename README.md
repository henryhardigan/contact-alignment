# contact-alignment
A method for assessing protein interaction potential by aligning short linear sequences for residue–residue complementarity.

## DB200K Workflow

The repo also contains a DB200K-based motif scanning workflow for cases where a
query peptide or loop should be scored against a FASTA using DB200K-derived
per-position profiles.

Main entry points:
- `scripts/scan_db200k.py`
- `scripts/scan_db200k_accessibility.py`
- `scripts/report_db200k_window_ranks.py`
- `scripts/rerank_db200k_contact_weighted.py`

The recommended starting point is:
- `scripts/scan_db200k.py` for raw DB200K scanning
- `scripts/scan_db200k_accessibility.py` when exposed/flexible windows should be favored over buried rigid ones

Worked examples, inputs, and output notes are documented in:
- [docs/db200k.md](docs/db200k.md)

Release-facing notes for packaging and citation:
- [docs/zenodo_release.md](docs/zenodo_release.md)
- [requirements-db200k.txt](requirements-db200k.txt)
- [CITATION.cff](CITATION.cff)

## Scope

This repository is a DB200K-focused export intended for citation and archival:
- packaged code lives under `contact_alignment/`
- DB200K command-line entry points are in `scripts/`
- examples and docs live under `examples/` and `docs/`
- non-DB200K tooling has been removed to keep the release lightweight

## Key References

- J. Holland and G. Grigoryan. *Structure-conditioned amino-acid couplings: How contact geometry affects pairwise sequence preferences.* Protein Science, 31(4), 2022. doi:10.1002/pro.4280.
- R. Kurusu, Y. Fujimoto, H. Morishita, D. Noshiro, S. Takada, K. Yamano, H. Tanaka, R. Arai, S. Kageyama, T. Funakoshi, et al. *Integrated proteomics identifies p62-dependent selective autophagy of the supramolecular vault complex.* Developmental Cell, 58(13):1189–1205.e11, 2023. doi:10.1016/j.devcel.2023.04.015.
