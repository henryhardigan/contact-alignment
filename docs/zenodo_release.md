# Zenodo Release Notes

This repo is used as an active analysis workspace. Before creating a Zenodo
snapshot, treat the release as a curated software export rather than a dump of
the working tree.

## Recommended Minimum for Release

- release from a clean commit
- confirm [README.md](../README.md) and [docs/db200k.md](db200k.md) match the intended release state
- include [CITATION.cff](../CITATION.cff)
- include [requirements-db200k.txt](../requirements-db200k.txt) or install the `db200k` extra from [pyproject.toml](../pyproject.toml)

## External Inputs Not Bundled

The DB200K examples require an extracted DB200K root passed as `--db-root`.
That dataset is not versioned in this repo.

The documented examples in [docs/db200k.md](db200k.md) use small repo-local files under [examples/db200k](../examples/db200k). The structure map points at a local NBR1 AlphaFold model path; if your layout differs, update that TSV before running the example.

## Suggested Release Scope

Good release content:
- `contact_alignment/`
- `scripts/` needed for DB200K scanning and reranking
- `tests/`
- `docs/`
- small example inputs used in the documentation

Avoid releasing transient analysis outputs unless they are part of the intended
record:
- large `tmp/` outputs
- one-off result TSVs
- ad hoc exploratory notes

## Reproducibility Checklist

1. Create a clean release commit.
2. Verify the documented commands in [docs/db200k.md](db200k.md).
3. Run `pytest -q tests/test_db200k_scan.py`.
4. Tag the release.
5. Let Zenodo archive the tagged GitHub release.
