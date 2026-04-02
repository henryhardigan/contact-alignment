# contact-alignment

Tools for scoring short linear sequence windows against DB200K contact-conditioned profiles. The repository is scoped to the DB200K workflow and small, reproducible examples.

## What's Included
- Python package: `contact_alignment/` (DB200K profile loading and scanning utilities)
- CLI scripts under `scripts/` for scanning, accessibility-aware reranking, reporting exact window ranks, and reciprocal rescoring
- Documentation: `docs/db200k.md`, `docs/zenodo_release.md`
- Minimal example inputs under `examples/db200k/` (NBR1 UBA domain FASTA, structure map, AlphaFold PDB)
- Citation metadata: `CITATION.cff`

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[db200k]
```
Dependencies are listed in `requirements-db200k.txt` and the `db200k` extra in `pyproject.toml`.

## Quickstart (local example)
Scan the MVP loop queries against the bundled NBR1 UBA FASTA:
```bash
python scripts/scan_db200k.py \
  --query-seq ETSEAKGPDGMALPRPR \
  --db-root /path/to/pisces-cache-scaac \
  --fasta examples/db200k/nbr1_human_uba_913_959.fasta \
  --alignment-mode trim_query_one_target_gap \
  --target-flank 1 \
  --peripheral-flank-weight 0.5 \
  --top-k 20
```
For accessibility/disorder-aware reranking:
```bash
python scripts/scan_db200k_accessibility.py \
  --query-seq FGFETSEAKGPDGMALPRPRDQA \
  --db-root /path/to/pisces-cache-scaac \
  --fasta examples/db200k/nbr1_human_uba_913_959.fasta \
  --structure-map examples/db200k/nbr1_human_uba_structure_map.tsv \
  --alignment-mode trim_query_one_target_gap \
  --target-flank 1 \
  --peripheral-flank-weight 0.5 \
  --top-k 5
```
More examples and guidance are in `docs/db200k.md`.

## Scripts at a Glance
- `scan_db200k.py` — raw DB200K scan with alignment breakdowns
- `scan_db200k_accessibility.py` — adds heuristic prefiltering, RSA/pLDDT rerank
- `report_db200k_window_ranks.py` — report exact rank of specified target windows
- `rerank_db200k_contact_weighted.py` / `rescore_db200k_reciprocal_3x3.py` — alternate scoring and reciprocal modes

## Release Notes
See `docs/zenodo_release.md` for the checklist used when tagging a Zenodo release (clean tree, docs verified, tests run).

## Key References
- J. Holland and G. Grigoryan. *Structure-conditioned amino-acid couplings: How contact geometry affects pairwise sequence preferences.* Protein Science, 31(4), 2022. doi:10.1002/pro.4280.
- R. Kurusu et al. *Integrated proteomics identifies p62-dependent selective autophagy of the supramolecular vault complex.* Developmental Cell, 58(13):1189–1205.e11, 2023. doi:10.1016/j.devcel.2023.04.015.
