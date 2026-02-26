# linear-complement-alignment
A method for assessing protein interaction potential by aligning short linear sequences for residue–residue complementarity.

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
```

### Notes
- Outputs are TSVs written to the current working directory.
- If you want to change inputs, pass `--ecoli-batch` and `--pulldown` explicitly.
