"""Helpers for working with the DB200K structure-conditioned energy dataset."""

from pathlib import Path

import numpy as onp

from contact_alignment import residues


THREE_TO_ONE = {three.upper(): one for one, three in residues.RES_CODE.items()}


def get_motif_stem(pdb_id: str, left_id: str, right_id: str) -> str:
    """Builds the DB200K motif stem from identifiers like `1A1X`, `A13`, `A23`."""
    return f"{pdb_id}_{left_id}_{right_id}"


def get_motif_paths(
    db_root: str | Path,
    motif_size: str,
    pdb_id: str,
    left_id: str,
    right_id: str,
) -> tuple[Path, Path]:
    """Returns the fragment PDB path and energy-table path for a DB200K motif."""
    if motif_size not in {"1x1", "3x3", "5x5"}:
        raise ValueError("motif_size must be one of: 1x1, 3x3, 5x5.")

    pdb_id = pdb_id.upper()
    shard = pdb_id[:2]
    motif_stem = get_motif_stem(pdb_id, left_id, right_id)
    db_root = Path(db_root)

    frag_path = db_root / f"frags-{motif_size}" / shard / pdb_id / f"{motif_stem}.pdb"
    energy_path = db_root / f"en-{motif_size}" / shard / pdb_id / f"{motif_stem}.etab"
    return frag_path, energy_path


def load_etab_matrix(etab_path: str | Path) -> onp.ndarray:
    """Loads a DB200K `.etab` file into a `(20, 20)` matrix ordered by `RES_ALPHA`."""
    etab_path = Path(etab_path)
    matrix = onp.full((residues.NUM_RESIDUES, residues.NUM_RESIDUES), onp.nan, dtype=onp.float64)

    for line in etab_path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed .etab line in {etab_path}: {line}")

        _, _, left_three, right_three, energy = fields
        left_one = THREE_TO_ONE[left_three.upper()]
        right_one = THREE_TO_ONE[right_three.upper()]
        left_idx = residues.RES_ALPHA.index(left_one)
        right_idx = residues.RES_ALPHA.index(right_one)
        matrix[left_idx, right_idx] = float(energy)

    if onp.isnan(matrix).any():
        raise ValueError(f"Incomplete .etab matrix in {etab_path}.")

    return matrix
