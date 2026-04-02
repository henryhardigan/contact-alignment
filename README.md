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
Align the MVP shoulder loop segment (607-623) against the NBR1 UBA domain (913-966):
```bash
python scripts/scan_db200k.py \
  --query-seq ETSEAKGPDGMALPRPR \
  --db-root /path/to/pisces-cache-scaac \
  --fasta examples/db200k/nbr1_human_uba_913_966.fasta \
  --top-k 10
```
More examples and options are in `docs/db200k.md`.

## Scripts at a Glance
- `scan_db200k.py` — raw DB200K scan with alignment breakdowns
- `scan_db200k_accessibility.py` — adds heuristic prefiltering, RSA/pLDDT rerank
- `report_db200k_window_ranks.py` — report exact rank of specified target windows
- `rerank_db200k_contact_weighted.py` / `rescore_db200k_reciprocal_3x3.py` — alternate scoring and reciprocal modes

## Release Notes
See `docs/zenodo_release.md` for the checklist used when tagging a Zenodo release (clean tree, docs verified, tests run).

## Minimal FASTA Samples
- `examples/sprot_min/MVP_HUMAN.fasta` (UniProt Q14764)
- `examples/sprot_min/NBR1_HUMAN.fasta` (UniProt Q14596)

To test against the full Swiss-Prot set, download directly from UniProt:
```bash
# Reviewed (Swiss-Prot) only
curl -L -o swissprot.fasta.gz \"https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28reviewed:true%29\"
gunzip swissprot.fasta.gz
```
Then point `--fasta` or `--fasta-glob` at the downloaded file.

## DB200K Download

DB200K fragment PDBs and energy tables are **not** bundled. Download `DB200K.tar.gz` from the supplemental files of the Protein Science paper “Structure-conditioned amino-acid couplings” (PMC8927866), extract it, and point `--db-root` at the extracted directory, e.g.:

```bash
tar -xzf DB200K.tar.gz -C /path/to
db_root=/path/to/DB200K
python scripts/scan_db200k.py --db-root "$db_root" ...
```

Expected layout under `$db_root`:
```
frags-1x1/  frags-3x3/  frags-5x5/
en-1x1/     en-3x3/     en-5x5/
```
Each motif directory holds fragment `.pdb` files and `.etab` energy tables ordered by the DB200K residue alphabet `MGKTRADEYVLQWFSHNPCI`.

## Key References
- J. Holland and G. Grigoryan. *Structure-conditioned amino-acid couplings: How contact geometry affects pairwise sequence preferences.* Protein Science, 31(4), 2022. doi:10.1002/pro.4280.
- R. Kurusu et al. *Integrated proteomics identifies p62-dependent selective autophagy of the supramolecular vault complex.* Developmental Cell, 58(13):1189–1205.e11, 2023. doi:10.1016/j.devcel.2023.04.015.
