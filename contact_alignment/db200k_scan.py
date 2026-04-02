"""Query-specific scanning utilities built on the DB200K motif-energy dataset."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import hashlib
import heapq
import os
from pathlib import Path

import numpy as onp

from contact_alignment import db200k, residues


THREE_TO_ONE = {three.upper(): one for one, three in residues.RES_CODE.items()}
CENTER_ALPHABET = residues.RES_ALPHA
CENTER_ALPHABET_SET = set(CENTER_ALPHABET)
CENTER_ALPHABET_SIZE = residues.NUM_RESIDUES
CACHE_VERSION = 4
DEGENERATE_5X5_BLEND_WEIGHT = 0.25
LATENT_GEOMETRY_TEMPERATURE = 1.0
PROFILE_STRATEGIES = (
    "hierarchical_5x5_3x3_1x1",
    "shrinkage_5x5_3x3_1x1",
    "blend_3x3_1x1",
    "hierarchical_5x5degen1_3x3_1x1",
)
ONE_BY_ONE_MODES = ("directed", "reciprocal_sum", "reciprocal_mean")
ALIGNMENT_MODES = ("rigid", "trim_query_one_target_gap")
SCORE_MODES = ("raw", "centered", "confidence_adjusted")
AROMATIC_PROLINE_DIPEPTIDE_RESCUE_RESIDUES = frozenset({"F", "W", "Y"})
OFFSET_CHARGE_RESCUE_PAIRS = {
    "D": frozenset({"R", "K"}),
    "E": frozenset({"R", "K"}),
    "R": frozenset({"D", "E"}),
    "K": frozenset({"D", "E"}),
}
SECONDARY_OFFSET_CHARGE_RESCUE_WEIGHT = 0.5
GRANTHAM_DIST = {
    ("A","C"):195,("A","D"):126,("A","E"):107,("A","F"):113,("A","G"):60,("A","H"):86,("A","I"):94,("A","K"):106,("A","L"):96,("A","M"):84,("A","N"):111,("A","P"):27,("A","Q"):91,("A","R"):112,("A","S"):99,("A","T"):58,("A","V"):64,("A","W"):148,("A","Y"):112,
    ("C","D"):154,("C","E"):170,("C","F"):205,("C","G"):159,("C","H"):174,("C","I"):198,("C","K"):202,("C","L"):198,("C","M"):196,("C","N"):139,("C","P"):169,("C","Q"):154,("C","R"):180,("C","S"):112,("C","T"):149,("C","V"):192,("C","W"):215,("C","Y"):194,
    ("D","E"):45,("D","F"):177,("D","G"):94,("D","H"):81,("D","I"):168,("D","K"):101,("D","L"):172,("D","M"):160,("D","N"):23,("D","P"):108,("D","Q"):61,("D","R"):96,("D","S"):65,("D","T"):85,("D","V"):152,("D","W"):181,("D","Y"):160,
    ("E","F"):140,("E","G"):98,("E","H"):40,("E","I"):134,("E","K"):56,("E","L"):138,("E","M"):126,("E","N"):42,("E","P"):93,("E","Q"):29,("E","R"):54,("E","S"):80,("E","T"):65,("E","V"):121,("E","W"):152,("E","Y"):122,
    ("F","G"):153,("F","H"):100,("F","I"):21,("F","K"):102,("F","L"):22,("F","M"):28,("F","N"):158,("F","P"):114,("F","Q"):116,("F","R"):97,("F","S"):155,("F","T"):103,("F","V"):50,("F","W"):40,("F","Y"):22,
    ("G","H"):98,("G","I"):135,("G","K"):127,("G","L"):138,("G","M"):127,("G","N"):80,("G","P"):42,("G","Q"):87,("G","R"):125,("G","S"):56,("G","T"):59,("G","V"):109,("G","W"):184,("G","Y"):147,
    ("H","I"):94,("H","K"):32,("H","L"):99,("H","M"):87,("H","N"):68,("H","P"):77,("H","Q"):24,("H","R"):29,("H","S"):89,("H","T"):47,("H","V"):84,("H","W"):115,("H","Y"):83,
    ("I","K"):102,("I","L"):5,("I","M"):10,("I","N"):149,("I","P"):95,("I","Q"):109,("I","R"):97,("I","S"):142,("I","T"):89,("I","V"):29,("I","W"):61,("I","Y"):33,
    ("K","L"):107,("K","M"):95,("K","N"):94,("K","P"):103,("K","Q"):53,("K","R"):26,("K","S"):121,("K","T"):78,("K","V"):97,("K","W"):110,("K","Y"):85,
    ("L","M"):15,("L","N"):153,("L","P"):98,("L","Q"):113,("L","R"):102,("L","S"):145,("L","T"):92,("L","V"):32,("L","W"):61,("L","Y"):36,
    ("M","N"):142,("M","P"):87,("M","Q"):101,("M","R"):91,("M","S"):135,("M","T"):81,("M","V"):21,("M","W"):67,("M","Y"):36,
    ("N","P"):91,("N","Q"):46,("N","R"):86,("N","S"):46,("N","T"):65,("N","V"):133,("N","W"):174,("N","Y"):143,
    ("P","Q"):76,("P","R"):103,("P","S"):74,("P","T"):38,("P","V"):68,("P","W"):147,("P","Y"):110,
    ("Q","R"):43,("Q","S"):68,("Q","T"):42,("Q","V"):96,("Q","W"):130,("Q","Y"):99,
    ("R","S"):110,("R","T"):71,("R","V"):96,("R","W"):101,("R","Y"):77,
    ("S","T"):58,("S","V"):124,("S","W"):177,("S","Y"):144,
    ("T","V"):69,("T","W"):128,("T","Y"):92,
    ("V","W"):88,("V","Y"):55,
    ("W","Y"):37,
}

def _grantham_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if (a, b) in GRANTHAM_DIST:
        return GRANTHAM_DIST[(a, b)]
    if (b, a) in GRANTHAM_DIST:
        return GRANTHAM_DIST[(b, a)]
    return 999


@dataclass(frozen=True)
class SequenceIndexEntry:
    count: int
    mean: onp.ndarray
    std: onp.ndarray
    geometry_bucket_count: int


@dataclass(frozen=True)
class PositionProfile:
    position: int
    center_residue: str
    query_context: str
    energies: onp.ndarray
    energy_stds: onp.ndarray
    count_5x5: int
    count_3x3: int
    count_1x1: int
    geometry_buckets_5x5: int
    geometry_buckets_3x3: int
    geometry_buckets_1x1: int
    weight_5x5: float
    weight_3x3: float
    weight_1x1: float
    support_mode: str
    effective_support: float
    rescue_index_3x3: dict[str, SequenceIndexEntry] | None = None


@dataclass(frozen=True)
class AlignmentResult:
    score: float
    breakdown: list[tuple[int, str, float]]
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    skipped_target_index: int | None
    aligned_window: str
    peripheral_breakdown: tuple[tuple[int, int, str, float], ...] = ()


def _expected_seq_length(motif_size: str) -> int:
    if motif_size == "1x1":
        return 1
    if motif_size == "3x3":
        return 3
    if motif_size == "5x5":
        return 5
    raise ValueError(f"Unsupported motif_size: {motif_size}")


def _load_fragment_chain_data(pdb_path: str | Path) -> dict[str, tuple[str, onp.ndarray]]:
    residues_by_chain: dict[str, list[str]] = {}
    coords_by_chain: dict[str, list[tuple[float, float, float]]] = {}
    seen = set()
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip().upper()
        if atom_name != "CA":
            continue
        chain = line[21].strip() or " "
        resnum = int(line[22:26])
        icode = line[26].strip()
        key = (chain, resnum, icode)
        if key in seen:
            continue
        seen.add(key)
        resname = line[17:20].strip().upper()
        if resname not in THREE_TO_ONE:
            raise ValueError(f"Unsupported residue in DB200K fragment: {resname}")
        residues_by_chain.setdefault(chain, []).append(THREE_TO_ONE[resname])
        coords_by_chain.setdefault(chain, []).append(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    return {
        chain: (
            "".join(residues_by_chain[chain]),
            onp.asarray(coords_by_chain[chain], dtype=onp.float64),
        )
        for chain in residues_by_chain
    }


def _geometry_signature(ca_coords: onp.ndarray) -> tuple[float, ...]:
    if len(ca_coords) <= 1:
        return ()
    deltas = ca_coords[:, None, :] - ca_coords[None, :, :]
    distances = onp.sqrt(onp.sum(deltas * deltas, axis=-1))
    upper = distances[onp.triu_indices(len(ca_coords), k=1)]
    return tuple(onp.round(upper, 1).tolist())


def _get_cache_dir(db_root: str | Path, cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(db_root) / ".db200k_cache"


def _canonical_db_root(db_root: str | Path) -> str:
    return str(Path(db_root).resolve())


def _db_root_cache_token(db_root: str | Path) -> str:
    return hashlib.sha256(_canonical_db_root(db_root).encode("utf-8")).hexdigest()[:12]


def _get_cache_path(
    db_root: str | Path,
    motif_size: str,
    cache_dir: str | Path | None = None,
) -> Path:
    cache_root = _get_cache_dir(db_root, cache_dir)
    token = _db_root_cache_token(db_root)
    return cache_root / f"sequence_index_v{CACHE_VERSION}_{token}_{motif_size}.npz"


def _iter_center_fixed_grantham50_degenerate_pentapeptides(pentapeptide: str) -> list[str]:
    candidates = [""]
    for idx, residue in enumerate(pentapeptide):
        if idx == 2:
            options = (residue,)
        else:
            options = tuple(
                alt for alt in CENTER_ALPHABET if _grantham_distance(residue, alt) <= 50
            )
            options = options if options else (residue,)
        candidates = [prefix + option for prefix in candidates for option in options]
    return list(dict.fromkeys(candidates))


def _write_cache_npz(cache_path: Path, **arrays) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f"{cache_path.stem}.tmp-{os.getpid()}.npz"
    onp.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(cache_path)


def _aggregate_geometry_bucket_rows(
    bucket_means: list[onp.ndarray],
    bucket_stds: list[onp.ndarray],
    bucket_counts: list[int],
) -> tuple[onp.ndarray, onp.ndarray]:
    if not bucket_means:
        raise ValueError("At least one geometry bucket is required.")
    if len(bucket_means) == 1:
        return bucket_means[0].copy(), bucket_stds[0].copy()

    weights = onp.asarray(bucket_counts, dtype=onp.float64)
    weights /= onp.sum(weights)
    means = onp.stack(bucket_means, axis=0)
    stds = onp.stack(bucket_stds, axis=0)

    # Treat contact geometry as a latent state and aggregate in Boltzmann space
    # rather than taking an arithmetic mean of incompatible geometry-specific rows.
    scaled = onp.log(weights)[:, None] - means / LATENT_GEOMETRY_TEMPERATURE
    max_scaled = onp.max(scaled, axis=0)
    log_partition = max_scaled + onp.log(onp.sum(onp.exp(scaled - max_scaled), axis=0))
    effective_mean = -LATENT_GEOMETRY_TEMPERATURE * log_partition

    variances = onp.sum(
        weights[:, None] * (onp.square(stds) + onp.square(means - effective_mean)),
        axis=0,
    )
    effective_std = onp.sqrt(onp.maximum(variances, 0.0))
    return effective_mean, effective_std


def _build_sequence_index(
    db_root: str | Path,
    motif_size: str,
) -> dict[str, SequenceIndexEntry]:
    db_root = Path(db_root)
    frag_root = db_root / f"frags-{motif_size}"
    expected_len = _expected_seq_length(motif_size)
    bucket_sums: dict[str, dict[tuple[float, ...], onp.ndarray]] = defaultdict(dict)
    bucket_sumsq: dict[str, dict[tuple[float, ...], onp.ndarray]] = defaultdict(dict)
    bucket_counts: dict[str, dict[tuple[float, ...], int]] = defaultdict(dict)

    for frag_path in frag_root.rglob("*.pdb"):
        try:
            chain_data = _load_fragment_chain_data(frag_path)
        except ValueError:
            continue

        valid_chains = [
            (seq, ca_coords)
            for seq, ca_coords in chain_data.values()
            if len(seq) == expected_len and len(ca_coords) == expected_len
        ]
        if not valid_chains:
            continue

        pdb_id, left_id, right_id = frag_path.stem.split("_")
        _, etab_path = db200k.get_motif_paths(db_root, motif_size, pdb_id, left_id, right_id)
        matrix = onp.array(db200k.load_etab_matrix(etab_path))

        for seq, ca_coords in valid_chains:
            center_residue = seq[0] if motif_size == "1x1" else seq[1]
            center_idx = CENTER_ALPHABET.index(center_residue)
            geometry_signature = _geometry_signature(ca_coords)
            if geometry_signature not in bucket_sums[seq]:
                bucket_sums[seq][geometry_signature] = onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64)
                bucket_sumsq[seq][geometry_signature] = onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64)
                bucket_counts[seq][geometry_signature] = 0
            row = matrix[center_idx]
            bucket_sums[seq][geometry_signature] += row
            bucket_sumsq[seq][geometry_signature] += row * row
            bucket_counts[seq][geometry_signature] += 1

    stats: dict[str, SequenceIndexEntry] = {}
    for seq in sorted(bucket_counts):
        signatures = sorted(bucket_counts[seq], key=lambda signature: (-bucket_counts[seq][signature], signature))
        per_bucket_counts = [bucket_counts[seq][signature] for signature in signatures]
        per_bucket_means = [
            bucket_sums[seq][signature] / bucket_counts[seq][signature]
            for signature in signatures
        ]
        per_bucket_stds = [
            onp.sqrt(
                onp.maximum(
                    bucket_sumsq[seq][signature] / bucket_counts[seq][signature]
                    - onp.square(bucket_sums[seq][signature] / bucket_counts[seq][signature]),
                    0.0,
                )
            )
            for signature in signatures
        ]
        effective_mean, effective_std = _aggregate_geometry_bucket_rows(
            per_bucket_means,
            per_bucket_stds,
            per_bucket_counts,
        )
        stats[seq] = SequenceIndexEntry(
            count=sum(per_bucket_counts),
            mean=effective_mean,
            std=effective_std,
            geometry_bucket_count=len(signatures),
        )
    return stats


def _save_sequence_index_cache(
    cache_path: str | Path,
    db_root: str | Path,
    stats: dict[str, SequenceIndexEntry],
) -> None:
    cache_path = Path(cache_path)
    seqs = onp.array(sorted(stats), dtype="<U8")
    counts = onp.array([stats[seq].count for seq in seqs], dtype=onp.int64)
    means = onp.stack([stats[seq].mean for seq in seqs], axis=0) if len(seqs) else onp.empty(
        (0, CENTER_ALPHABET_SIZE),
        dtype=onp.float64,
    )
    stds = onp.stack([stats[seq].std for seq in seqs], axis=0) if len(seqs) else onp.empty(
        (0, CENTER_ALPHABET_SIZE),
        dtype=onp.float64,
    )
    geometry_bucket_counts = onp.array([stats[seq].geometry_bucket_count for seq in seqs], dtype=onp.int64)
    _write_cache_npz(
        cache_path,
        version=onp.array([CACHE_VERSION], dtype=onp.int64),
        db_root=onp.array([_canonical_db_root(db_root)], dtype="<U4096"),
        seqs=seqs,
        counts=counts,
        means=means,
        stds=stds,
        geometry_bucket_counts=geometry_bucket_counts,
    )


def _load_sequence_index_cache(
    cache_path: str | Path,
    db_root: str | Path | None = None,
) -> dict[str, SequenceIndexEntry]:
    cache_path = Path(cache_path)
    with onp.load(cache_path, allow_pickle=False) as payload:
        version = int(payload["version"][0])
        if version != CACHE_VERSION:
            raise ValueError(
                f"Incompatible DB200K cache version in {cache_path}: {version} != {CACHE_VERSION}"
            )
        cache_db_root = str(payload["db_root"][0]) if "db_root" in payload else None
        expected_db_root = _canonical_db_root(db_root) if db_root is not None else None
        if expected_db_root is not None and cache_db_root != expected_db_root:
            raise ValueError(
                f"Incompatible DB200K cache source in {cache_path}: "
                f"{cache_db_root!r} != {expected_db_root!r}"
            )
        seqs = payload["seqs"]
        counts = payload["counts"]
        means = payload["means"]
        stds = payload["stds"] if "stds" in payload else onp.zeros_like(means)
        geometry_bucket_counts = (
            payload["geometry_bucket_counts"]
            if "geometry_bucket_counts" in payload
            else onp.ones(len(seqs), dtype=onp.int64)
        )

    return {
        str(seq): SequenceIndexEntry(
            count=int(count),
            mean=means[idx].astype(onp.float64, copy=True),
            std=stds[idx].astype(onp.float64, copy=True),
            geometry_bucket_count=int(geometry_bucket_counts[idx]),
        )
        for idx, (seq, count) in enumerate(zip(seqs, counts))
    }


def load_or_build_sequence_index(
    db_root: str | Path,
    motif_size: str,
    *,
    cache_dir: str | Path | None = None,
    rebuild: bool = False,
) -> dict[str, SequenceIndexEntry]:
    """Loads a cached sequence index or builds it once from the raw DB200K files."""
    cache_path = _get_cache_path(db_root, motif_size, cache_dir)
    if cache_path.exists() and not rebuild:
        return _load_sequence_index_cache(cache_path, db_root)

    stats = _build_sequence_index(db_root, motif_size)
    _save_sequence_index_cache(cache_path, db_root, stats)
    return stats


def _load_reciprocal_1x1_matrix_tsv(
    matrix_tsv: str | Path,
    directed_index_1x1: dict[str, SequenceIndexEntry],
) -> dict[str, SequenceIndexEntry]:
    matrix_tsv = Path(matrix_tsv)
    lines = [line.rstrip("\n\r") for line in matrix_tsv.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Reciprocal 1x1 matrix TSV is empty: {matrix_tsv}")

    header = lines[0].split("\t")
    if not header or header[0] != "":
        raise ValueError(f"Reciprocal 1x1 matrix TSV must start with a blank header cell: {matrix_tsv}")
    column_order = header[1:]
    if sorted(column_order) != sorted(CENTER_ALPHABET):
        raise ValueError(
            f"Reciprocal 1x1 matrix TSV columns do not match the DB200K residue alphabet: {matrix_tsv}"
        )
    column_index = {residue: idx for idx, residue in enumerate(column_order)}

    reciprocal_rows: dict[str, onp.ndarray] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(column_order) + 1:
            raise ValueError(f"Malformed reciprocal 1x1 matrix row in {matrix_tsv}: {line}")
        residue = fields[0]
        if residue not in CENTER_ALPHABET_SET:
            raise ValueError(f"Unsupported residue {residue!r} in {matrix_tsv}")
        values = [float(value) for value in fields[1:]]
        reciprocal_rows[residue] = onp.array(
            [values[column_index[target]] for target in CENTER_ALPHABET],
            dtype=onp.float64,
        )

    missing = sorted(set(CENTER_ALPHABET) - set(reciprocal_rows))
    if missing:
        raise ValueError(f"Reciprocal 1x1 matrix TSV is missing rows for: {''.join(missing)}")

    return {
        residue: SequenceIndexEntry(
            count=directed_index_1x1[residue].count,
            mean=reciprocal_rows[residue].copy(),
            std=onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64),
            geometry_bucket_count=directed_index_1x1[residue].geometry_bucket_count,
        )
        for residue in CENTER_ALPHABET
    }


def _build_reciprocal_1x1_index(
    directed_index_1x1: dict[str, SequenceIndexEntry],
    *,
    metric: str,
) -> dict[str, SequenceIndexEntry]:
    if metric not in {"sum", "mean"}:
        raise ValueError(f"Unsupported reciprocal 1x1 metric: {metric}")

    reciprocal_index = {}
    for source in CENTER_ALPHABET:
        source_entry = directed_index_1x1[source]
        count = source_entry.count
        source_row = source_entry.mean
        source_var = onp.square(source_entry.std)
        reciprocal_row = onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64)
        reciprocal_std = onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64)
        for target_idx, target in enumerate(CENTER_ALPHABET):
            reverse_entry = directed_index_1x1[target]
            reverse_row = reverse_entry.mean
            reverse_var = onp.square(reverse_entry.std)
            reverse_idx = CENTER_ALPHABET.index(source)
            total = float(source_row[target_idx]) + float(reverse_row[reverse_idx])
            total_var = float(source_var[target_idx]) + float(reverse_var[reverse_idx])
            if metric == "sum":
                reciprocal_row[target_idx] = total
                reciprocal_std[target_idx] = total_var ** 0.5
            else:
                reciprocal_row[target_idx] = total / 2.0
                reciprocal_std[target_idx] = (total_var ** 0.5) / 2.0
        reciprocal_index[source] = SequenceIndexEntry(
            count=count,
            mean=reciprocal_row,
            std=reciprocal_std,
            geometry_bucket_count=source_entry.geometry_bucket_count,
        )
    return reciprocal_index


def load_one_by_one_index(
    db_root: str | Path,
    *,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
    one_by_one_mode: str = "directed",
    one_by_one_matrix_tsv: str | Path | None = None,
) -> dict[str, SequenceIndexEntry]:
    if one_by_one_mode not in ONE_BY_ONE_MODES:
        raise ValueError(f"Unsupported one_by_one_mode: {one_by_one_mode}")

    directed_index_1x1 = load_or_build_sequence_index(
        db_root,
        "1x1",
        cache_dir=cache_dir,
        rebuild=rebuild_cache,
    )
    if one_by_one_matrix_tsv is not None:
        return _load_reciprocal_1x1_matrix_tsv(one_by_one_matrix_tsv, directed_index_1x1)
    if one_by_one_mode == "directed":
        return directed_index_1x1
    metric = "sum" if one_by_one_mode == "reciprocal_sum" else "mean"
    return _build_reciprocal_1x1_index(directed_index_1x1, metric=metric)


def _blend_profile_row(
    row_3x3: onp.ndarray | None,
    count_3x3: int,
    row_1x1: onp.ndarray,
    count_1x1: int,
    strong_threshold: int,
    strong_weight_3x3: float,
    weak_weight_3x3: float,
) -> tuple[onp.ndarray, float, float, str]:
    if row_3x3 is None or count_3x3 == 0:
        return row_1x1.copy(), 0.0, 1.0, "1x1_only"

    if count_3x3 >= strong_threshold:
        weight_3x3 = strong_weight_3x3
        support_mode = "strong"
    else:
        weight_3x3 = weak_weight_3x3
        support_mode = "weak"

    weight_1x1 = 1.0 - weight_3x3
    energies = weight_3x3 * row_3x3 + weight_1x1 * row_1x1
    return energies, weight_3x3, weight_1x1, support_mode


def _shrinkage_weight(count: int, prior_count: float) -> float:
    if count <= 0:
        return 0.0
    return float(count) / float(count + prior_count)


def _blend_entries(
    primary_entry: SequenceIndexEntry | None,
    fallback_mean: onp.ndarray,
    fallback_std: onp.ndarray,
    fallback_effective_support: float,
    *,
    prior_count: float,
    primary_name: str,
    fallback_name: str,
) -> tuple[onp.ndarray, onp.ndarray, float, float, str, float, int]:
    if primary_entry is None or primary_entry.count == 0:
        return (
            fallback_mean.copy(),
            fallback_std.copy(),
            0.0,
            1.0,
            fallback_name,
            fallback_effective_support,
            0,
        )

    primary_weight = _shrinkage_weight(primary_entry.count, prior_count)
    fallback_weight = 1.0 - primary_weight
    energies = primary_weight * primary_entry.mean + fallback_weight * fallback_mean
    variances = (
        (primary_weight ** 2) * onp.square(primary_entry.std)
        + (fallback_weight ** 2) * onp.square(fallback_std)
    )
    return (
        energies,
        onp.sqrt(onp.maximum(variances, 0.0)),
        primary_weight,
        fallback_weight,
        f"{primary_name}_shrinkage",
        primary_weight * primary_entry.count + fallback_weight * fallback_effective_support,
        primary_entry.geometry_bucket_count,
    )


def _select_hierarchical_profile_row(
    row_5x5: onp.ndarray | None,
    count_5x5: int,
    row_3x3: onp.ndarray | None,
    count_3x3: int,
    row_1x1: onp.ndarray | None,
) -> tuple[onp.ndarray, float, float, float, str]:
    if row_5x5 is not None and count_5x5 > 0:
        return row_5x5.copy(), 1.0, 0.0, 0.0, "5x5_primary"
    if row_3x3 is not None and count_3x3 > 0:
        return row_3x3.copy(), 0.0, 1.0, 0.0, "3x3_primary"
    if row_1x1 is None:
        raise ValueError("No DB200K support found for query position under hierarchical profile selection.")
    return row_1x1.copy(), 0.0, 0.0, 1.0, "1x1_only"


def _blend_degenerate_5x5_with_3x3(
    row_5x5: onp.ndarray,
    row_3x3: onp.ndarray | None,
    row_1x1: onp.ndarray | None,
) -> tuple[onp.ndarray, float, float, float, str]:
    if row_3x3 is not None:
        weight_5x5 = DEGENERATE_5X5_BLEND_WEIGHT
        weight_3x3 = 1.0 - weight_5x5
        energies = weight_5x5 * row_5x5 + weight_3x3 * row_3x3
        return energies, weight_5x5, weight_3x3, 0.0, "5x5_degen_3x3_blend"
    return row_5x5.copy(), 1.0, 0.0, 0.0, "5x5_degen_only"


def _make_terminal_1x1_profile(
    position: int,
    residue: str,
    index_1x1: dict[str, SequenceIndexEntry],
    *,
    rescue_index_3x3: dict[str, SequenceIndexEntry] | None = None,
) -> PositionProfile:
    if residue not in index_1x1:
        raise ValueError(f"No 1x1 DB200K support found for residue {residue}.")
    entry = index_1x1[residue]
    return PositionProfile(
        position=position,
        center_residue=residue,
        query_context=residue,
        energies=entry.mean.copy(),
        energy_stds=entry.std.copy(),
        count_5x5=0,
        count_3x3=0,
        count_1x1=entry.count,
        geometry_buckets_5x5=0,
        geometry_buckets_3x3=0,
        geometry_buckets_1x1=entry.geometry_bucket_count,
        weight_5x5=0.0,
        weight_3x3=0.0,
        weight_1x1=1.0,
        support_mode="1x1_terminal",
        effective_support=float(entry.count),
        rescue_index_3x3=rescue_index_3x3,
    )


def build_query_profiles(
    query_seq: str,
    db_root: str | Path,
    *,
    profile_strategy: str = "hierarchical_5x5_3x3_1x1",
    one_by_one_mode: str = "directed",
    one_by_one_matrix_tsv: str | Path | None = None,
    strong_threshold: int = 100,
    strong_weight_3x3: float = 0.8,
    weak_weight_3x3: float = 0.5,
    shrinkage_prior_3x3: float = 25.0,
    shrinkage_prior_5x5: float = 10.0,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
) -> list[PositionProfile]:
    """Builds DB200K partner profiles across the full query length.

    Terminal query residues are always represented by 1x1-only profiles, while
    interior positions use the selected 5x5/3x3/1x1 strategy.
    """
    query_seq = query_seq.upper()
    if not query_seq:
        raise ValueError("query_seq must be non-empty.")
    invalid = sorted(set(query_seq) - CENTER_ALPHABET_SET)
    if invalid:
        raise ValueError(f"Unsupported residues in query_seq: {''.join(invalid)}")
    if profile_strategy not in PROFILE_STRATEGIES:
        raise ValueError(f"Unsupported profile_strategy: {profile_strategy}")
    if one_by_one_mode not in ONE_BY_ONE_MODES:
        raise ValueError(f"Unsupported one_by_one_mode: {one_by_one_mode}")

    triplets = [query_seq[i : i + 3] for i in range(len(query_seq) - 2)]
    pentapeptides = [
        query_seq[i - 2 : i + 3] if 2 <= i <= len(query_seq) - 3 else None
        for i in range(1, len(query_seq) - 1)
    ]
    center_residues = sorted({triplet[1] for triplet in triplets})

    index_5x5 = {}
    if profile_strategy in {
        "shrinkage_5x5_3x3_1x1",
        "hierarchical_5x5_3x3_1x1",
        "hierarchical_5x5degen1_3x3_1x1",
    }:
        index_5x5 = load_or_build_sequence_index(
            db_root,
            "5x5",
            cache_dir=cache_dir,
            rebuild=rebuild_cache,
        )
    index_3x3 = load_or_build_sequence_index(
        db_root,
        "3x3",
        cache_dir=cache_dir,
        rebuild=rebuild_cache,
    )
    index_1x1 = load_one_by_one_index(
        db_root,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        one_by_one_mode=one_by_one_mode,
        one_by_one_matrix_tsv=one_by_one_matrix_tsv,
    )
    if len(query_seq) == 1:
        return [_make_terminal_1x1_profile(1, query_seq[0], index_1x1, rescue_index_3x3=index_3x3)]
    stats_5x5 = {
        pentapeptide: index_5x5[pentapeptide]
        for pentapeptide in set(p for p in pentapeptides if p is not None)
        if pentapeptide in index_5x5
    }
    stats_3x3 = {triplet: index_3x3[triplet] for triplet in set(triplets) if triplet in index_3x3}
    stats_1x1 = {
        residue: index_1x1[residue]
        for residue in center_residues
        if residue in index_1x1
    }

    profiles = [_make_terminal_1x1_profile(1, query_seq[0], index_1x1, rescue_index_3x3=index_3x3)]
    for i, triplet in enumerate(triplets, start=2):
        center_residue = triplet[1]
        if profile_strategy in {"blend_3x3_1x1", "shrinkage_5x5_3x3_1x1"} and center_residue not in stats_1x1:
            raise ValueError(f"No 1x1 DB200K support found for residue {center_residue}.")

        count_1x1 = 0
        row_1x1 = None
        std_1x1 = onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64)
        geometry_buckets_1x1 = 0
        if center_residue in stats_1x1:
            entry_1x1 = stats_1x1[center_residue]
            count_1x1 = entry_1x1.count
            row_1x1 = entry_1x1.mean
            std_1x1 = entry_1x1.std
            geometry_buckets_1x1 = entry_1x1.geometry_bucket_count
        count_5x5 = 0
        row_5x5 = None
        std_5x5 = onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64)
        geometry_buckets_5x5 = 0
        pentapeptide = pentapeptides[i - 2]
        if pentapeptide is not None:
            if pentapeptide in stats_5x5:
                entry_5x5 = stats_5x5[pentapeptide]
                count_5x5 = entry_5x5.count
                row_5x5 = entry_5x5.mean
                std_5x5 = entry_5x5.std
                geometry_buckets_5x5 = entry_5x5.geometry_bucket_count
            elif profile_strategy == "hierarchical_5x5degen1_3x3_1x1":
                matched_5x5 = [
                    candidate
                    for candidate in _iter_center_fixed_grantham50_degenerate_pentapeptides(pentapeptide)
                    if candidate in index_5x5
                ]
                if matched_5x5:
                    count_5x5 = sum(index_5x5[candidate].count for candidate in matched_5x5)
                    row_5x5 = sum(
                        index_5x5[candidate].count * index_5x5[candidate].mean for candidate in matched_5x5
                    ) / count_5x5
                    std_5x5 = onp.sqrt(
                        onp.maximum(
                            sum(
                                index_5x5[candidate].count * (
                                    onp.square(index_5x5[candidate].std) + onp.square(index_5x5[candidate].mean)
                                )
                                for candidate in matched_5x5
                            ) / count_5x5
                            - onp.square(row_5x5),
                            0.0,
                        )
                    )
                    geometry_buckets_5x5 = sum(index_5x5[candidate].geometry_bucket_count for candidate in matched_5x5)
        count_3x3 = 0
        row_3x3 = None
        std_3x3 = onp.zeros(CENTER_ALPHABET_SIZE, dtype=onp.float64)
        geometry_buckets_3x3 = 0
        if triplet in stats_3x3:
            entry_3x3 = stats_3x3[triplet]
            count_3x3 = entry_3x3.count
            row_3x3 = entry_3x3.mean
            std_3x3 = entry_3x3.std
            geometry_buckets_3x3 = entry_3x3.geometry_bucket_count

        effective_support = 0.0
        if profile_strategy == "shrinkage_5x5_3x3_1x1":
            if row_1x1 is None:
                raise ValueError(f"No 1x1 DB200K support found for residue {center_residue}.")
            energies = row_1x1.copy()
            energy_stds = std_1x1.copy()
            weight_5x5 = 0.0
            weight_3x3 = 0.0
            weight_1x1 = 1.0
            support_mode = "1x1_only"
            effective_support = float(count_1x1)
            geometry_buckets_from_fallback = geometry_buckets_1x1
            if row_3x3 is not None and count_3x3 > 0:
                energies, energy_stds, weight_3x3, weight_1x1, support_mode, effective_support, geometry_buckets_from_fallback = _blend_entries(
                    stats_3x3[triplet],
                    energies,
                    energy_stds,
                    effective_support,
                    prior_count=shrinkage_prior_3x3,
                    primary_name="3x3",
                    fallback_name=support_mode,
                )
            if row_5x5 is not None and count_5x5 > 0:
                five_entry = SequenceIndexEntry(
                    count=count_5x5,
                    mean=row_5x5,
                    std=std_5x5,
                    geometry_bucket_count=geometry_buckets_5x5,
                )
                energies, energy_stds, weight_5x5, fallback_weight, support_mode, effective_support, geometry_buckets_from_fallback = _blend_entries(
                    five_entry,
                    energies,
                    energy_stds,
                    effective_support,
                    prior_count=shrinkage_prior_5x5,
                    primary_name="5x5",
                    fallback_name=support_mode,
                )
                weight_3x3 *= fallback_weight
                weight_1x1 *= fallback_weight
            if geometry_buckets_5x5 == 0:
                geometry_buckets_5x5 = geometry_buckets_from_fallback if weight_5x5 > 0.0 else 0
        elif profile_strategy in {"hierarchical_5x5_3x3_1x1", "hierarchical_5x5degen1_3x3_1x1"}:
            energies, weight_5x5, weight_3x3, weight_1x1, support_mode = _select_hierarchical_profile_row(
                row_5x5,
                count_5x5,
                row_3x3,
                count_3x3,
                row_1x1,
            )
            if (
                profile_strategy == "hierarchical_5x5degen1_3x3_1x1"
                and support_mode == "5x5_primary"
                and pentapeptide is not None
                and pentapeptide not in stats_5x5
            ):
                energies, weight_5x5, weight_3x3, weight_1x1, support_mode = _blend_degenerate_5x5_with_3x3(
                    row_5x5,
                    row_3x3,
                    row_1x1,
                )
            if support_mode == "5x5_degen_3x3_blend":
                variances = (weight_5x5 ** 2) * onp.square(std_5x5) + (weight_3x3 ** 2) * onp.square(std_3x3)
                energy_stds = onp.sqrt(onp.maximum(variances, 0.0))
                effective_support = weight_5x5 * count_5x5 + weight_3x3 * count_3x3
            elif support_mode.startswith("5x5"):
                energy_stds = std_5x5.copy()
                effective_support = float(count_5x5)
            elif support_mode.startswith("3x3"):
                energy_stds = std_3x3.copy()
                effective_support = float(count_3x3)
            else:
                energy_stds = std_1x1.copy()
                effective_support = float(count_1x1)
        else:
            if row_1x1 is None:
                raise ValueError(f"No 1x1 DB200K support found for residue {center_residue}.")
            energies, weight_3x3, weight_1x1, support_mode = _blend_profile_row(
                row_3x3,
                count_3x3,
                row_1x1,
                count_1x1,
                strong_threshold,
                strong_weight_3x3,
                weak_weight_3x3,
            )
            weight_5x5 = 0.0
            variances = (weight_3x3 ** 2) * onp.square(std_3x3) + (weight_1x1 ** 2) * onp.square(std_1x1)
            energy_stds = onp.sqrt(onp.maximum(variances, 0.0))
            effective_support = weight_3x3 * count_3x3 + weight_1x1 * count_1x1
        profiles.append(
            PositionProfile(
                position=i,
                center_residue=center_residue,
                query_context=triplet,
                energies=energies,
                energy_stds=energy_stds,
                count_5x5=count_5x5,
                count_3x3=count_3x3,
                count_1x1=count_1x1,
                geometry_buckets_5x5=geometry_buckets_5x5,
                geometry_buckets_3x3=geometry_buckets_3x3,
                geometry_buckets_1x1=geometry_buckets_1x1,
                weight_5x5=weight_5x5,
                weight_3x3=weight_3x3,
                weight_1x1=weight_1x1,
                support_mode=support_mode,
                effective_support=effective_support,
                rescue_index_3x3=index_3x3,
            )
        )

    profiles.append(
        _make_terminal_1x1_profile(
            len(query_seq),
            query_seq[-1],
            index_1x1,
            rescue_index_3x3=index_3x3,
        )
    )
    return profiles


def score_window(
    window_seq: str,
    profiles: list[PositionProfile],
    *,
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
) -> tuple[float, list[tuple[int, str, float]]]:
    """Scores a candidate window against a query profile using additive DB200K energies."""
    if len(window_seq) != len(profiles):
        raise ValueError("window_seq length must match the number of profiles.")
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    breakdown = []
    for residue, profile in zip(window_seq, profiles):
        energy = _score_profile_residue(profile, residue, score_mode=score_mode, uncertainty_floor=uncertainty_floor)
        breakdown.append((profile.position, residue, energy))
    total, adjusted, _ = _apply_offset_neighbor_rescue_with_trace(
        window_seq,
        profiles,
        breakdown,
        score_mode=score_mode,
        uncertainty_floor=uncertainty_floor,
    )
    return total, adjusted


def score_window_with_donor_trace(
    window_seq: str,
    profiles: list[PositionProfile],
    *,
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
) -> tuple[float, list[tuple[int, str, float]], list[int]]:
    """Like score_window(), but also returns the target donor index for each position."""
    if len(window_seq) != len(profiles):
        raise ValueError("window_seq length must match the number of profiles.")
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    breakdown = []
    for residue, profile in zip(window_seq, profiles):
        energy = _score_profile_residue(profile, residue, score_mode=score_mode, uncertainty_floor=uncertainty_floor)
        breakdown.append((profile.position, residue, energy))
    return _apply_offset_neighbor_rescue_with_trace(
        window_seq,
        profiles,
        breakdown,
        score_mode=score_mode,
        uncertainty_floor=uncertainty_floor,
    )


def _score_profile_residue(
    profile: PositionProfile,
    residue: str,
    *,
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
) -> float:
    if residue not in CENTER_ALPHABET_SET:
        raise ValueError(f"Unsupported residue in window_seq: {residue}")
    residue_idx = CENTER_ALPHABET.index(residue)
    energy = float(profile.energies[residue_idx])
    if score_mode == "raw":
        return energy
    centered = energy - float(onp.mean(profile.energies))
    if score_mode == "centered":
        return centered
    if score_mode == "confidence_adjusted":
        scale = max(float(profile.energy_stds[residue_idx]), float(uncertainty_floor))
        return centered / scale
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def _neighbor_rescue_allowed(source_center_residue: str, neighbor_residue: str) -> bool:
    if (
        source_center_residue in AROMATIC_PROLINE_DIPEPTIDE_RESCUE_RESIDUES
        and neighbor_residue == "P"
    ):
        return True
    return neighbor_residue in OFFSET_CHARGE_RESCUE_PAIRS.get(source_center_residue, ())


def _format_rescued_breakdown_label(center_residue: str, neighbor_residue: str) -> str:
    return f"{center_residue}<-{neighbor_residue}"


def _format_multi_rescued_breakdown_label(center_residue: str, neighbor_residues: Iterable[str]) -> str:
    return f"{center_residue}<-{'+'.join(neighbor_residues)}"


def _score_charge_rescue_with_target_centered_3x3(
    profile: PositionProfile,
    window_seq: str,
    neighbor_idx: int,
) -> float | None:
    index_3x3 = profile.rescue_index_3x3
    if index_3x3 is None:
        return None
    if neighbor_idx <= 0 or neighbor_idx >= len(window_seq) - 1:
        return None
    target_triplet = window_seq[neighbor_idx - 1 : neighbor_idx + 2]
    if len(target_triplet) != 3:
        return None
    target_entry = index_3x3.get(target_triplet)
    if target_entry is None:
        return None
    return float(target_entry.mean[CENTER_ALPHABET.index(profile.center_residue)])


def _apply_offset_neighbor_rescue(
    window_seq: str,
    profiles: list[PositionProfile],
    breakdown: list[tuple[int, str, float]],
    *,
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
) -> tuple[float, list[tuple[int, str, float]]]:
    total, adjusted, _ = _apply_offset_neighbor_rescue_with_trace(
        window_seq,
        profiles,
        breakdown,
        score_mode=score_mode,
        uncertainty_floor=uncertainty_floor,
    )
    return total, adjusted


def _apply_offset_neighbor_rescue_with_trace(
    window_seq: str,
    profiles: list[PositionProfile],
    breakdown: list[tuple[int, str, float]],
    *,
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
) -> tuple[float, list[tuple[int, str, float]], list[int]]:
    adjusted = list(breakdown)
    donor_indices = list(range(len(window_seq)))
    if len(window_seq) < 2:
        return sum(entry[2] for entry in adjusted), adjusted, donor_indices

    consumed_donor_indices: set[int] = set()
    for idx, profile in enumerate(profiles):
        if idx in consumed_donor_indices:
            continue
        direct_energy = adjusted[idx][2]
        direct_label = adjusted[idx][1]
        rescue_candidates: list[tuple[float, float, int, float]] = []
        for neighbor_idx in (idx - 1, idx + 1):
            if neighbor_idx < 0 or neighbor_idx >= len(window_seq):
                continue
            if neighbor_idx in consumed_donor_indices:
                continue
            if not _neighbor_rescue_allowed(profile.center_residue, window_seq[neighbor_idx]):
                continue
            contextual_rescue = None
            if profile.center_residue in OFFSET_CHARGE_RESCUE_PAIRS:
                contextual_rescue = _score_charge_rescue_with_target_centered_3x3(
                    profile,
                    window_seq,
                    neighbor_idx,
                )
            rescued_energy = min(
                direct_energy,
                contextual_rescue
                if contextual_rescue is not None
                else _score_profile_residue(
                    profile,
                    window_seq[neighbor_idx],
                    score_mode=score_mode,
                    uncertainty_floor=uncertainty_floor,
                ),
            )
            donor_energy = adjusted[neighbor_idx][2]
            improvement = (direct_energy + donor_energy) - rescued_energy
            if improvement > 0.0:
                delta = rescued_energy - (direct_energy + donor_energy)
                rescue_candidates.append((delta, improvement, neighbor_idx, rescued_energy))

        if not rescue_candidates:
            continue

        rescue_candidates.sort(key=lambda item: (item[0], item[3], item[2]))
        _, primary_improvement, neighbor_idx, rescued_energy = rescue_candidates[0]
        rescue_label = direct_label
        donors_to_consume = [neighbor_idx]
        center_energy = rescued_energy

        if profile.center_residue in OFFSET_CHARGE_RESCUE_PAIRS and len(rescue_candidates) > 1:
            _, secondary_improvement, secondary_idx, secondary_rescued_energy = rescue_candidates[1]
            secondary_delta = SECONDARY_OFFSET_CHARGE_RESCUE_WEIGHT * secondary_improvement
            combined_energy = (
                direct_energy
                + adjusted[neighbor_idx][2]
                + adjusted[secondary_idx][2]
                - (primary_improvement + secondary_delta)
            )
            if combined_energy < rescued_energy + adjusted[secondary_idx][2]:
                donors_to_consume.append(secondary_idx)
                center_energy = min(center_energy, combined_energy, secondary_rescued_energy)

        if window_seq[neighbor_idx] != direct_label or rescued_energy < direct_energy:
            if len(donors_to_consume) == 1:
                rescue_label = f"{_format_rescued_breakdown_label(direct_label, window_seq[neighbor_idx])}@{neighbor_idx+1}"
            else:
                donor_labels = ",".join(f"{window_seq[donor]}@{donor+1}" for donor in donors_to_consume)
                rescue_label = f"{direct_label}<-{donor_labels}"
        adjusted[idx] = (adjusted[idx][0], rescue_label, center_energy)
        donor_indices[idx] = neighbor_idx
        for donor_idx in donors_to_consume:
            consumed_donor_indices.add(donor_idx)
            adjusted[donor_idx] = (
                adjusted[donor_idx][0],
                adjusted[donor_idx][1],
                0.0,
            )

    return sum(entry[2] for entry in adjusted), adjusted, donor_indices


def _score_triplet_center_with_offset_neighbor_rescue(
    source_center_residue: str,
    target_triplet: str,
    source_row: onp.ndarray,
) -> float:
    target_center = target_triplet[1]
    direct_energy = float(source_row[CENTER_ALPHABET.index(target_center)])

    rescued_energy = direct_energy
    for neighbor_residue in (target_triplet[0], target_triplet[2]):
        if not _neighbor_rescue_allowed(source_center_residue, neighbor_residue):
            continue
        rescued_energy = min(
            rescued_energy,
            float(source_row[CENTER_ALPHABET.index(neighbor_residue)]),
        )
    return rescued_energy


def score_window_reciprocal_3x3(
    query_seq: str,
    target_seq: str,
    index_3x3: dict[str, SequenceIndexEntry],
    *,
    metric: str = "sum",
    require_full: bool = False,
) -> tuple[float, list[tuple[int, str, str, float, float, float]], int]:
    """Dynamically rescores an aligned window using reciprocal 3x3 triplet terms.

    For each interior aligned position, the score combines:
    - forward: query triplet -> target center residue
    - reverse: target triplet -> query center residue

    Positions whose query or target triplet is missing from the 3x3 index are
    skipped unless `require_full` is set.
    """
    query_seq = query_seq.upper()
    target_seq = target_seq.upper()
    if len(query_seq) != len(target_seq):
        raise ValueError("query_seq and target_seq must have equal length.")
    if len(query_seq) < 3:
        return 0.0, [], 0
    if metric not in {"sum", "mean"}:
        raise ValueError(f"Unsupported reciprocal 3x3 metric: {metric}")
    invalid_query = sorted(set(query_seq) - CENTER_ALPHABET_SET)
    if invalid_query:
        raise ValueError(f"Unsupported residues in query_seq: {''.join(invalid_query)}")
    invalid_target = sorted(set(target_seq) - CENTER_ALPHABET_SET)
    if invalid_target:
        raise ValueError(f"Unsupported residues in target_seq: {''.join(invalid_target)}")

    total = 0.0
    breakdown: list[tuple[int, str, str, float, float, float]] = []
    used_positions = 0
    for center_idx in range(1, len(query_seq) - 1):
        query_triplet = query_seq[center_idx - 1 : center_idx + 2]
        target_triplet = target_seq[center_idx - 1 : center_idx + 2]
        if query_triplet not in index_3x3 or target_triplet not in index_3x3:
            if require_full:
                missing = query_triplet if query_triplet not in index_3x3 else target_triplet
                raise ValueError(f"Missing 3x3 DB200K support for triplet: {missing}")
            continue

        query_center = query_triplet[1]
        target_center = target_triplet[1]
        forward = _score_triplet_center_with_offset_neighbor_rescue(
            query_center,
            target_triplet,
            index_3x3[query_triplet].mean,
        )
        reverse = _score_triplet_center_with_offset_neighbor_rescue(
            target_center,
            query_triplet,
            index_3x3[target_triplet].mean,
        )
        combined = forward + reverse
        if metric == "mean":
            combined /= 2.0
        used_positions += 1
        total += combined
        breakdown.append(
            (
                center_idx + 1,
                query_triplet,
                target_triplet,
                forward,
                reverse,
                combined,
            )
        )
    return total, breakdown, used_positions


def _interpolate_peripheral_rescue(
    direct_energy: float,
    flank_energy: float,
    peripheral_flank_weight: float,
) -> float:
    if flank_energy >= direct_energy:
        return direct_energy
    return direct_energy + peripheral_flank_weight * (flank_energy - direct_energy)


def _apply_peripheral_rescue(
    window_seq: str,
    sub_profiles: list[PositionProfile],
    breakdown: list[tuple[int, str, float]],
    aligned_target_indices: list[int],
    *,
    target_offset: int,
    peripheral_flank_weight: float,
    score_mode: str,
    uncertainty_floor: float,
) -> tuple[float, list[tuple[int, str, float]], tuple[tuple[int, int, str, float], ...]]:
    if peripheral_flank_weight <= 0.0 or not aligned_target_indices:
        return sum(entry[2] for entry in breakdown), breakdown, ()

    adjusted = list(breakdown)
    rescue_events: list[tuple[int, int, str, float]] = []

    def maybe_apply(profile_idx: int, flank_rel_index: int) -> None:
        if flank_rel_index < 1 or flank_rel_index > len(window_seq):
            return
        profile = sub_profiles[profile_idx]
        direct_energy = adjusted[profile_idx][2]
        flank_residue = window_seq[flank_rel_index - 1]
        flank_energy = _score_profile_residue(
            profile,
            flank_residue,
            score_mode=score_mode,
            uncertainty_floor=uncertainty_floor,
        )
        rescued_energy = _interpolate_peripheral_rescue(
            direct_energy,
            flank_energy,
            peripheral_flank_weight,
        )
        if rescued_energy >= direct_energy:
            return
        adjusted[profile_idx] = (adjusted[profile_idx][0], adjusted[profile_idx][1], rescued_energy)
        rescue_events.append(
            (
                profile.position,
                target_offset + flank_rel_index,
                flank_residue,
                rescued_energy,
            )
        )

    if len(aligned_target_indices) == 1:
        profile = sub_profiles[0]
        direct_energy = adjusted[0][2]
        best_event: tuple[int, int, str, float] | None = None
        best_energy = direct_energy
        for flank_rel_index in (aligned_target_indices[0] - 1, aligned_target_indices[0] + 1):
            if flank_rel_index < 1 or flank_rel_index > len(window_seq):
                continue
            flank_residue = window_seq[flank_rel_index - 1]
            flank_energy = _score_profile_residue(
                profile,
                flank_residue,
                score_mode=score_mode,
                uncertainty_floor=uncertainty_floor,
            )
            rescued_energy = _interpolate_peripheral_rescue(
                direct_energy,
                flank_energy,
                peripheral_flank_weight,
            )
            if rescued_energy < best_energy:
                best_energy = rescued_energy
                best_event = (
                    profile.position,
                    target_offset + flank_rel_index,
                    flank_residue,
                    rescued_energy,
                )
        if best_event is not None:
            adjusted[0] = (adjusted[0][0], adjusted[0][1], best_energy)
            rescue_events.append(best_event)
    else:
        maybe_apply(0, aligned_target_indices[0] - 1)
        maybe_apply(len(adjusted) - 1, aligned_target_indices[-1] + 1)

    return sum(entry[2] for entry in adjusted), adjusted, tuple(rescue_events)


def score_window_semiglobal(
    window_seq: str,
    profiles: list[PositionProfile],
    *,
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
    max_target_gaps: int = 1,
    min_aligned_positions: int | None = None,
    target_gap_penalty: float = 0.0,
    target_offset: int = 0,
    peripheral_flank_weight: float = 0.5,
) -> AlignmentResult:
    """Scores a candidate window allowing query-end trimming and limited target skips."""
    if any(res not in CENTER_ALPHABET_SET for res in window_seq):
        raise ValueError(f"Unsupported residue in window_seq: {window_seq}")

    n_profiles = len(profiles)
    n_window = len(window_seq)
    if min_aligned_positions is None:
        min_aligned_positions = max(1, n_profiles - 1)
    if min_aligned_positions > n_profiles:
        raise ValueError("min_aligned_positions must be <= number of profiles.")
    if max_target_gaps not in {0, 1}:
        raise ValueError("Only 0 or 1 target gaps are currently supported.")
    if target_offset < 0:
        raise ValueError("target_offset must be >= 0.")
    if not 0.0 <= peripheral_flank_weight <= 1.0:
        raise ValueError("peripheral_flank_weight must be between 0.0 and 1.0.")
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    best: AlignmentResult | None = None
    for q_start in range(n_profiles):
        for q_end in range(q_start + min_aligned_positions, n_profiles + 1):
            sub_profiles = profiles[q_start:q_end]
            aligned_len = len(sub_profiles)

            for t_start in range(n_window - aligned_len + 1):
                target = window_seq[t_start : t_start + aligned_len]
                score, breakdown = score_window(
                    target,
                    sub_profiles,
                    score_mode=score_mode,
                    uncertainty_floor=uncertainty_floor,
                )
                aligned_target_indices = list(range(t_start + 1, t_start + aligned_len + 1))
                score, breakdown, peripheral_breakdown = _apply_peripheral_rescue(
                    window_seq,
                    sub_profiles,
                    breakdown,
                    aligned_target_indices,
                    target_offset=target_offset,
                    peripheral_flank_weight=peripheral_flank_weight,
                    score_mode=score_mode,
                    uncertainty_floor=uncertainty_floor,
                )
                result = AlignmentResult(
                    score=score,
                    breakdown=breakdown,
                    query_start=q_start + 1,
                    query_end=q_end,
                    target_start=target_offset + t_start + 1,
                    target_end=target_offset + t_start + aligned_len,
                    skipped_target_index=None,
                    aligned_window=target,
                    peripheral_breakdown=peripheral_breakdown,
                )
                if best is None or result.score < best.score:
                    best = result

            if max_target_gaps == 0 or aligned_len + 1 > n_window:
                continue

            for t_start in range(n_window - (aligned_len + 1) + 1):
                target_block = window_seq[t_start : t_start + aligned_len + 1]
                for gap_pos in range(aligned_len + 1):
                    target = target_block[:gap_pos] + target_block[gap_pos + 1 :]
                    score, breakdown = score_window(
                        target,
                        sub_profiles,
                        score_mode=score_mode,
                        uncertainty_floor=uncertainty_floor,
                    )
                    aligned_target_indices = [
                        t_start + offset + 1 for offset in range(aligned_len + 1) if offset != gap_pos
                    ]
                    score, breakdown, peripheral_breakdown = _apply_peripheral_rescue(
                        window_seq,
                        sub_profiles,
                        breakdown,
                        aligned_target_indices,
                        target_offset=target_offset,
                        peripheral_flank_weight=peripheral_flank_weight,
                        score_mode=score_mode,
                        uncertainty_floor=uncertainty_floor,
                    )
                    score += target_gap_penalty
                    result = AlignmentResult(
                        score=score,
                        breakdown=breakdown,
                        query_start=q_start + 1,
                        query_end=q_end,
                        target_start=target_offset + t_start + 1,
                        target_end=target_offset + t_start + aligned_len + 1,
                        skipped_target_index=target_offset + t_start + gap_pos + 1,
                        aligned_window=target,
                        peripheral_breakdown=peripheral_breakdown,
                    )
                    if best is None or result.score < best.score:
                        best = result

    if best is None:
        raise ValueError("No valid semiglobal alignment found.")
    return best


def iter_fasta_records(fasta_path: str | Path) -> Iterator[tuple[str, str]]:
    """Yields `(header, sequence)` records from a FASTA file."""
    header = None
    seq_chunks = []
    for line in Path(fasta_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_chunks)
            header = line[1:]
            seq_chunks = []
        else:
            seq_chunks.append(line.upper())
    if header is not None:
        yield header, "".join(seq_chunks)


def read_fasta_records(fasta_path: str | Path) -> list[tuple[str, str]]:
    """Reads a FASTA file into a list of `(header, sequence)` records."""
    return list(iter_fasta_records(fasta_path))


def scan_records(
    records_iter: Iterable[tuple[str, str]],
    profiles: list[PositionProfile],
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    prefilter_score_threshold: float | None = None,
    prefilter_score_mode: str = "raw",
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
    alignment_mode: str = "rigid",
    max_target_gaps: int = 1,
    min_aligned_positions: int | None = None,
    target_gap_penalty: float = 0.0,
    target_flank: int = 1,
    peripheral_flank_weight: float = 0.5,
    stats: dict[str, int] | None = None,
    progress_every_windows: int | None = None,
    progress_label: str | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Scores every sliding window in an iterable of FASTA records."""
    if alignment_mode not in ALIGNMENT_MODES:
        raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    if prefilter_score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported prefilter_score_mode: {prefilter_score_mode}")
    if target_flank < 0:
        raise ValueError("target_flank must be >= 0.")
    records: list[dict[str, object]] | list[tuple[float, int, dict[str, object]]] = []
    entry_idx = 0
    windows_scanned = 0
    prefilter_passed = 0
    scored_windows = 0
    threshold_passed = 0
    window_len = len(profiles)
    for header, sequence in records_iter:
        if len(sequence) < window_len:
            continue
        for start in range(len(sequence) - window_len + 1):
            window = sequence[start : start + window_len]
            if any(res not in CENTER_ALPHABET_SET for res in window):
                continue
            windows_scanned += 1
            prefilter_result: tuple[float, list[tuple[int, str, float]]] | None = None
            if prefilter_score_threshold is not None:
                prefilter_result = score_window(
                    window,
                    profiles,
                    score_mode=prefilter_score_mode,
                    uncertainty_floor=uncertainty_floor,
                )
                if prefilter_result[0] > prefilter_score_threshold:
                    if (
                        progress_every_windows is not None
                        and windows_scanned % progress_every_windows == 0
                    ):
                        label = progress_label or "scan"
                        print(
                            f"[{label}] windows={windows_scanned} "
                            f"prefilter_passed={prefilter_passed} "
                            f"scored={scored_windows} "
                            f"hits={threshold_passed}",
                            flush=True,
                        )
                    continue
                prefilter_passed += 1
            if alignment_mode == "rigid":
                if prefilter_result is not None and prefilter_score_mode == score_mode:
                    score, breakdown = prefilter_result
                else:
                    score, breakdown = score_window(
                        window,
                        profiles,
                        score_mode=score_mode,
                        uncertainty_floor=uncertainty_floor,
                    )
                alignment = None
            else:
                region_start = max(0, start - target_flank)
                region_end = min(len(sequence), start + window_len + target_flank)
                region_seq = sequence[region_start:region_end]
                if any(res not in CENTER_ALPHABET_SET for res in region_seq):
                    continue
                alignment = score_window_semiglobal(
                    region_seq,
                    profiles,
                    score_mode=score_mode,
                    uncertainty_floor=uncertainty_floor,
                    max_target_gaps=max_target_gaps,
                    min_aligned_positions=min_aligned_positions,
                    target_gap_penalty=target_gap_penalty,
                    target_offset=region_start,
                    peripheral_flank_weight=peripheral_flank_weight,
                )
                score = alignment.score
                breakdown = alignment.breakdown
            scored_windows += 1
            record = {
                "header": header,
                "start": start + 1,
                "end": start + window_len,
                "window": window,
                "score": score,
                "breakdown": breakdown,
            }
            if alignment is not None:
                record["alignment"] = alignment
            if score_threshold is not None and score > score_threshold:
                if (
                    progress_every_windows is not None
                    and windows_scanned % progress_every_windows == 0
                ):
                    label = progress_label or "scan"
                    print(
                        f"[{label}] windows={windows_scanned} "
                        f"prefilter_passed={prefilter_passed} "
                        f"scored={scored_windows} "
                        f"hits={threshold_passed}",
                        flush=True,
                    )
                continue
            threshold_passed += 1
            if top_k is None:
                records.append(record)
                if (
                    progress_every_windows is not None
                    and windows_scanned % progress_every_windows == 0
                ):
                    label = progress_label or "scan"
                    print(
                        f"[{label}] windows={windows_scanned} "
                        f"prefilter_passed={prefilter_passed} "
                        f"scored={scored_windows} "
                        f"hits={threshold_passed}",
                        flush=True,
                    )
                continue

            heap_entry = (-score, entry_idx, record)
            entry_idx += 1
            if len(records) < top_k:
                heapq.heappush(records, heap_entry)
            elif heap_entry[0] > records[0][0]:
                heapq.heapreplace(records, heap_entry)
            if (
                progress_every_windows is not None
                and windows_scanned % progress_every_windows == 0
            ):
                label = progress_label or "scan"
                print(
                    f"[{label}] windows={windows_scanned} "
                    f"prefilter_passed={prefilter_passed} "
                    f"scored={scored_windows} "
                    f"hits={threshold_passed}",
                    flush=True,
                )

    if stats is not None:
        stats.clear()
        stats["windows_scanned"] = windows_scanned
        stats["prefilter_passed"] = prefilter_passed
        stats["scored_windows"] = scored_windows
        stats["threshold_passed"] = threshold_passed

    if top_k is None:
        records.sort(key=lambda rec: rec["score"])
        return records, windows_scanned

    top_hits = [entry[2] for entry in records]
    top_hits.sort(key=lambda rec: rec["score"])
    return top_hits, windows_scanned


def scan_fasta(
    fasta_path: str | Path,
    profiles: list[PositionProfile],
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    prefilter_score_threshold: float | None = None,
    prefilter_score_mode: str = "raw",
    score_mode: str = "raw",
    uncertainty_floor: float = 1.0,
    alignment_mode: str = "rigid",
    max_target_gaps: int = 1,
    min_aligned_positions: int | None = None,
    target_gap_penalty: float = 0.0,
    target_flank: int = 1,
    peripheral_flank_weight: float = 0.5,
    stats: dict[str, int] | None = None,
    progress_every_windows: int | None = None,
    progress_label: str | None = None,
) -> list[dict[str, object]]:
    """Scores every sliding window in a FASTA file against a query profile."""
    records, _ = scan_records(
        iter_fasta_records(fasta_path),
        profiles,
        top_k=top_k,
        score_threshold=score_threshold,
        prefilter_score_threshold=prefilter_score_threshold,
        prefilter_score_mode=prefilter_score_mode,
        score_mode=score_mode,
        uncertainty_floor=uncertainty_floor,
        alignment_mode=alignment_mode,
        max_target_gaps=max_target_gaps,
        min_aligned_positions=min_aligned_positions,
        target_gap_penalty=target_gap_penalty,
        target_flank=target_flank,
        peripheral_flank_weight=peripheral_flank_weight,
        stats=stats,
        progress_every_windows=progress_every_windows,
        progress_label=progress_label,
    )
    return records
