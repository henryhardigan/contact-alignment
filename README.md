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

## Running Pocket Queries

This repo expects two inputs:
- `ecoli_batch/` (structure/residue data)
- `pulldown.tsv` (UniProt IDs)

Default structure input is expected under `src/`:
- `src/ecoli_batch/`

No default pulldown file is versioned in the repo.
Pass `--pulldown` explicitly to a TSV containing a `uniprot_id` column.

### Quickstart
Run from the repo root:
```bash
python3 src/pocket_pipeline/pocket_pipeline.py \
  --ecoli-batch src/ecoli_batch \
  --pulldown /path/to/pulldown.tsv \
  --mode metrics
```

### Common Query Examples
```bash
# Metrics for a short motif
python3 src/pocket_pipeline/pocket_pipeline.py \
  --ecoli-batch src/ecoli_batch \
  --pulldown /path/to/pulldown.tsv \
  --pattern "H [RK] [LIV] [DE] Y" \
  --mode metrics

# Rank pulldown proteins by similarity
python3 src/pocket_pipeline/pocket_pipeline.py \
  --ecoli-batch src/ecoli_batch \
  --pulldown /path/to/pulldown.tsv \
  --pattern "H [RK] [LIVM] [DE] Y" \
  --mode rank-pulldown

# Variant search with subset expansion (default)
python3 src/pocket_pipeline/pocket_pipeline.py \
  --ecoli-batch src/ecoli_batch \
  --pulldown /path/to/pulldown.tsv \
  --pattern "H [RK] [LIVM] [DE] Y" \
  --mode variant-search

# Pocket search using greedy cluster-seed strategy
python3 src/pocket_pipeline/pocket_pipeline.py \
  --ecoli-batch src/ecoli_batch \
  --pulldown /path/to/pulldown.tsv \
  --mode pocket-search \
  --cluster-seed \
  --no-mj-norm

# Fixed-register motif reuse search:
# require at least 8 exact identities and 2 conservative substitutions in a 15 aa window
# `--motif-len` can be omitted to use the full query length.
python3 mj_score.py \
  --seq1 MDRFLVAGQAAAALR \
  --fasta2 /path/to/proteome.fasta \
  --motif-search \
  --motif-len 15 \
  --motif-min-identity 8 \
  --motif-min-conservative 2 \
  --motif-rank total \
  --motif-topk 10

# Fast exhaustive motif reuse search over a proteome
python3 scripts/fast_motif_search.py \
  --seq1 VRLVGLHVTLLDPQMERQLVLGL \
  --fasta2 /path/to/proteome.fasta \
  --topk 10 \
  --rank-by total

# Grantham-distance fixed-window scan across Swiss-Prot chunks
python3 scripts/grantham_window_scan.py \
  --query AKGPDGMALP \
  --fasta-glob 'sprot_chunks/*.fasta' \
  --topk 20 \
  --out tmp/akgpdgmalp_grantham_top20.tsv
```

### Notes
- Outputs are TSVs written to the current working directory.
- If you want to change inputs, pass `--ecoli-batch` and `--pulldown` explicitly.
