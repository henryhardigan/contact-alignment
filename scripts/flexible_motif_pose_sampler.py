#!/usr/bin/env python3
"""Sample flexible peptide poses near a target motif with fixed sequences.

This repurposes the local idp-design loop-binder/JAX-MD utilities for a
targeted, fixed-sequence workflow:
1. Extract a target CA window from a PDB structure.
2. Build a coarse-grained peptide initialized near a chosen target motif.
3. Run short Langevin simulations with the target window restrained in place.
4. Rank sampled poses by motif contact, distance, and steric plausibility.

Defaults are set up for the TonB/BTP system discussed in this workspace.
"""

from __future__ import annotations

import argparse
import csv
import math
import math as pymath
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDP_DESIGN_ROOT = Path("/Users/henryhardigan/Desktop/src/idp-design")
if str(DEFAULT_IDP_DESIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_IDP_DESIGN_ROOT))

try:
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jax import jit, lax, random, vmap
    from jax_md import simulate, space
except ImportError as exc:  # pragma: no cover - import path/runtime guard
    raise SystemExit(
        "This script requires the JAX/idp-design environment. Run it with "
        "`conda run -n idp-design-py310 python "
        "scripts/flexible_motif_pose_sampler.py ...`."
    ) from exc

try:
    import contact_alignment.loop_binder as loop_binder
    import contact_alignment.utils as utils
    from contact_alignment.energy_prob import get_energy_fn
except ImportError as exc:  # pragma: no cover - import path/runtime guard
    raise SystemExit(
        "Failed to import local idp-design modules. Expected checkout at "
        f"{DEFAULT_IDP_DESIGN_ROOT}."
    ) from exc


AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


@dataclass(frozen=True)
class PoseRecord:
    orientation: str
    replica: int
    frame_index: int
    total_score: float
    motif_contact: float
    region_contact: float
    motif_centroid_dist: float
    binder_self_contact: float
    min_full_protein_ca_dist: float
    pdb_path: str
    coords: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-pdb",
        default="/Users/henryhardigan/structure_cache/alphafold/P02929.pdb",
        help="Path to the target PDB/AlphaFold model.",
    )
    parser.add_argument(
        "--target-chain",
        default=None,
        help="Optional chain identifier. Defaults to the first chain with CA atoms.",
    )
    parser.add_argument(
        "--target-window-start",
        type=int,
        default=55,
        help="First target residue in the anchored sampling window (1-based, inclusive).",
    )
    parser.add_argument(
        "--target-window-end",
        type=int,
        default=81,
        help="Last target residue in the anchored sampling window (1-based, inclusive).",
    )
    parser.add_argument(
        "--target-motif-start",
        type=int,
        default=66,
        help="First residue of the target motif inside the target structure (1-based).",
    )
    parser.add_argument(
        "--target-motif-end",
        type=int,
        default=70,
        help="Last residue of the target motif inside the target structure (1-based).",
    )
    parser.add_argument(
        "--binder-seq",
        default="MDRWLVKGILQWRKIRRRRRRRRRRR",
        help="Peptide sequence to sample.",
    )
    parser.add_argument(
        "--binder-motif-start",
        type=int,
        default=3,
        help="First residue of the peptide motif (1-based).",
    )
    parser.add_argument(
        "--binder-motif-end",
        type=int,
        default=7,
        help="Last residue of the peptide motif (1-based).",
    )
    parser.add_argument(
        "--orientation",
        choices=("parallel", "antiparallel", "both"),
        default="both",
        help="Initial motif alignment orientation.",
    )
    parser.add_argument(
        "--replicas-per-orientation",
        type=int,
        default=8,
        help="Independent replicas per starting orientation.",
    )
    parser.add_argument(
        "--eq-steps",
        type=int,
        default=1500,
        help="Equilibration steps per replica.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=5000,
        help="Sampling steps per replica after equilibration.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=100,
        help="Record one frame every N MD steps.",
    )
    parser.add_argument(
        "--kt",
        type=float,
        default=0.593,
        help="Thermal energy in kcal/mol-equivalent units.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Langevin timestep.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Langevin friction coefficient.",
    )
    parser.add_argument(
        "--binder-offset",
        type=float,
        default=8.0,
        help="Initial displacement of the peptide motif away from the target motif.",
    )
    parser.add_argument(
        "--jitter-sd-axis",
        type=float,
        default=1.0,
        help="Replica-specific translation jitter along the target motif axis.",
    )
    parser.add_argument(
        "--jitter-sd-binormal",
        type=float,
        default=1.0,
        help="Replica-specific translation jitter orthogonal to the motif axis/surface normal.",
    )
    parser.add_argument(
        "--jitter-sd-normal",
        type=float,
        default=0.5,
        help="Replica-specific translation jitter along the surface normal.",
    )
    parser.add_argument(
        "--rotation-jitter-deg",
        type=float,
        default=12.0,
        help="Replica-specific random rigid-body rotation magnitude.",
    )
    parser.add_argument(
        "--max-motif-dist",
        type=float,
        default=12.0,
        help="Distance threshold before the harmonic motif restraint turns on.",
    )
    parser.add_argument(
        "--spring-k",
        type=float,
        default=20.0,
        help="Strength of the harmonic motif-distance restraint.",
    )
    parser.add_argument(
        "--contact-r0",
        type=float,
        default=10.0,
        help="Soft-contact midpoint distance.",
    )
    parser.add_argument(
        "--contact-width",
        type=float,
        default=1.0,
        help="Soft-contact logistic width.",
    )
    parser.add_argument(
        "--motif-contact-bias",
        type=float,
        default=8.0,
        help="Weight of the attractive motif-contact term in the simulation energy.",
    )
    parser.add_argument(
        "--anchor-spring-k",
        type=float,
        default=60.0,
        help="Harmonic restraint strength keeping the target window near the structure.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top sampled poses to save as PDB files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    parser.add_argument(
        "--outdir",
        default="results/tonb_btp_flexible_sampler",
        help="Output directory.",
    )
    return parser.parse_args()


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.copy()
    return vec / norm


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    h = src.T @ dst
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    return rot


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def build_extended_ca_trace(seq: str, motif_start: int, motif_end: int) -> np.ndarray:
    step = 3.8
    amp = 1.15
    coords = []
    for i, _aa in enumerate(seq):
        coords.append(
            np.array([i * step, amp if i % 2 == 0 else -amp, 0.0], dtype=float)
        )
    arr = np.vstack(coords)
    motif_slice = slice(motif_start - 1, motif_end)
    return arr - arr[motif_slice].mean(axis=0)


def local_frame(motif_ca: np.ndarray, protein_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = unit(motif_ca[-1] - motif_ca[0])
    motif_cent = motif_ca.mean(axis=0)
    protein_cent = protein_ca.mean(axis=0)
    outward = unit(motif_cent - protein_cent)
    normal = outward - np.dot(outward, axis) * axis
    if np.linalg.norm(normal) < 1e-6:
        trial = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = trial - np.dot(trial, axis) * axis
    normal = unit(normal)
    binormal = unit(np.cross(axis, normal))
    return axis, normal, binormal


def extract_target_window(
    pdb_path: Path,
    start_residue: int,
    end_residue: int,
    chain_id: str | None,
) -> tuple[str, np.ndarray, np.ndarray, str]:
    residue_names, residue_coords, resolved_chain = loop_binder.load_pdb_ca_coords(
        str(pdb_path), chain_id=chain_id
    )
    residue_numbers = list(range(start_residue, end_residue + 1))
    missing = [resid for resid in residue_numbers if resid not in residue_coords]
    if missing:
        raise ValueError(f"Missing CA coordinates for residues: {missing}")
    sequence = "".join(residue_names[resid] for resid in residue_numbers)
    window_coords = np.vstack([np.asarray(residue_coords[resid]) for resid in residue_numbers])
    full_coords = np.vstack(
        [np.asarray(residue_coords[resid]) for resid in sorted(residue_coords)]
    )
    return sequence, window_coords, full_coords, resolved_chain


def place_binder_near_target(
    binder_template: np.ndarray,
    binder_motif_start: int,
    binder_motif_end: int,
    target_motif_ca: np.ndarray,
    protein_ca: np.ndarray,
    offset: float,
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    motif_slice = slice(binder_motif_start - 1, binder_motif_end)
    binder_motif = binder_template[motif_slice].copy()
    if reverse:
        binder_motif = binder_motif[::-1]
    binder_motif -= binder_motif.mean(axis=0)
    target_center = target_motif_ca.mean(axis=0)
    target_shifted = target_motif_ca - target_center
    rotation = kabsch(binder_motif, target_shifted)
    coords = binder_template @ rotation.T
    axis, normal, binormal = local_frame(target_motif_ca, protein_ca)
    coords += target_center + normal * offset
    return coords, axis, normal, binormal


def parse_orientations(orientation: str) -> list[str]:
    if orientation == "both":
        return ["parallel", "antiparallel"]
    return [orientation]


def make_initial_states(
    args: argparse.Namespace,
    binder_template: np.ndarray,
    target_motif_ca: np.ndarray,
    protein_ca: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(args.seed)
    orientations = parse_orientations(args.orientation)
    states = []
    labels = []
    for orientation in orientations:
        reverse = orientation == "antiparallel"
        base_coords, axis, normal, binormal = place_binder_near_target(
            binder_template,
            args.binder_motif_start,
            args.binder_motif_end,
            target_motif_ca,
            protein_ca,
            args.binder_offset,
            reverse,
        )
        target_center = target_motif_ca.mean(axis=0)
        for replica in range(args.replicas_per_orientation):
            shift = (
                rng.normal(0.0, args.jitter_sd_axis) * axis
                + rng.normal(0.0, args.jitter_sd_binormal) * binormal
                + rng.normal(0.0, args.jitter_sd_normal) * normal
            )
            random_axis = unit(
                rng.normal(size=3)
                + 0.6 * axis
                + 0.3 * binormal
                + 0.2 * normal
            )
            rot = axis_angle_rotation(
                random_axis, math.radians(rng.normal(0.0, args.rotation_jitter_deg))
            )
            coords = (base_coords - target_center) @ rot.T + target_center + shift
            states.append(coords)
            labels.append(f"{orientation}:{replica}")
    return labels, np.stack(states)


def residue_range_to_relative(
    start_residue: int,
    end_residue: int,
    window_start: int,
    window_end: int,
) -> tuple[int, int]:
    if start_residue < window_start or end_residue > window_end or end_residue < start_residue:
        raise ValueError(
            "Motif residues must lie inside the requested target window: "
            f"motif={start_residue}-{end_residue}, window={window_start}-{window_end}"
        )
    return start_residue - window_start, end_residue - window_start + 1


def soft_contact_strength_np(
    dists: np.ndarray, contact_r0: float, contact_width: float
) -> np.ndarray:
    scaled = (contact_r0 - dists) / contact_width
    return 1.0 / (1.0 + np.exp(-scaled))


def segment_contact_score_np(
    left: np.ndarray,
    right: np.ndarray,
    contact_r0: float,
    contact_width: float,
) -> float:
    dists = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(np.mean(soft_contact_strength_np(dists, contact_r0, contact_width)))


def binder_self_contact_score_np(
    binder_coords: np.ndarray,
    contact_r0: float,
    contact_width: float,
    min_seq_sep: int = 2,
) -> float:
    left_idx, right_idx = np.triu_indices(binder_coords.shape[0], k=min_seq_sep)
    if left_idx.size == 0:
        return 0.0
    dists = np.linalg.norm(
        binder_coords[left_idx] - binder_coords[right_idx], axis=1
    )
    return float(
        np.mean(soft_contact_strength_np(dists, contact_r0, contact_width))
    )


def write_pose_pdb(
    out_path: Path,
    target_pdb: Path,
    binder_seq: str,
    binder_coords: np.ndarray,
    summary_line: str,
) -> None:
    lines = []
    trailing = []
    atom_serial = 1
    with target_pdb.open() as handle:
        for raw in handle:
            if raw.startswith(("ATOM", "HETATM", "TER", "END")):
                if raw.startswith("ATOM"):
                    line = raw.rstrip("\n")
                    lines.append(f"{line[:6]}{atom_serial:5d}{line[11:]}")
                    atom_serial += 1
                else:
                    trailing.append(raw.rstrip("\n"))
    for chunk in reversed(textwrap.wrap(summary_line, width=68)):
        lines.insert(0, f"REMARK   1 {chunk}")
    for idx, (aa, coord) in enumerate(zip(binder_seq, binder_coords), start=1):
        x, y, z = coord
        lines.append(
            f"ATOM  {atom_serial:5d}  CA  {AA3[aa]:>3s} B{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        atom_serial += 1
    lines.extend(trailing)
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.sample_steps % args.sample_every != 0:
        raise ValueError("--sample-steps must be divisible by --sample-every.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target_pdb = Path(args.target_pdb)
    target_seq, target_coords, full_protein_ca, resolved_chain = extract_target_window(
        target_pdb,
        args.target_window_start,
        args.target_window_end,
        args.target_chain,
    )
    target_motif_start_idx, target_motif_end_idx = residue_range_to_relative(
        args.target_motif_start,
        args.target_motif_end,
        args.target_window_start,
        args.target_window_end,
    )
    binder_len = len(args.binder_seq)
    binder_template = build_extended_ca_trace(
        args.binder_seq, args.binder_motif_start, args.binder_motif_end
    )
    target_motif_ca = target_coords[target_motif_start_idx:target_motif_end_idx]
    replica_labels, binder_initial_states = make_initial_states(
        args, binder_template, target_motif_ca, full_protein_ca
    )

    target_len = len(target_seq)
    binder_start = target_len
    motif_abs_start = binder_start + args.binder_motif_start - 1
    motif_abs_end = binder_start + args.binder_motif_end

    bonded_nbrs = loop_binder.build_bonded_nbrs(target_len, binder_len)
    unbonded_nbrs = loop_binder.build_unbonded_nbrs(target_len + binder_len, bonded_nbrs)
    displacement_fn, shift_fn = space.free()
    _, base_energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn)
    pseq = jnp.array(
        np.concatenate(
            [utils.seq_to_one_hot(target_seq), utils.seq_to_one_hot(args.binder_seq)],
            axis=0,
        ),
        dtype=jnp.float64,
    )
    mass = utils.get_pseq_mass(pseq)
    target_positions = jnp.array(target_coords, dtype=jnp.float64)
    initial_states = jnp.concatenate(
        [jnp.repeat(target_positions[None, :, :], binder_initial_states.shape[0], axis=0),
         jnp.array(binder_initial_states, dtype=jnp.float64)],
        axis=1,
    )
    anchor_indices = jnp.arange(target_len, dtype=jnp.int32)
    anchor_targets = target_positions

    def segment_centroid_distance(R: jnp.ndarray) -> jnp.ndarray:
        left = R[target_motif_start_idx:target_motif_end_idx]
        right = R[motif_abs_start:motif_abs_end]
        left_center = jnp.mean(left, axis=0)
        right_center = jnp.mean(right, axis=0)
        return space.distance(displacement_fn(left_center, right_center))

    def motif_contact_score(R: jnp.ndarray) -> jnp.ndarray:
        left = R[target_motif_start_idx:target_motif_end_idx]
        right = R[motif_abs_start:motif_abs_end]
        dists = vmap(
            lambda lp: vmap(
                lambda rp: space.distance(displacement_fn(lp, rp))
            )(right)
        )(left)
        scaled = (args.contact_r0 - dists) / args.contact_width
        return jnp.mean(1.0 / (1.0 + jnp.exp(-scaled)))

    def biased_energy_fn(R: jnp.ndarray, pseq: jnp.ndarray) -> jnp.ndarray:
        base = base_energy_fn(R, pseq=pseq)
        restraint = loop_binder.anchor_restraint_energy(
            R,
            anchor_indices,
            anchor_targets,
            args.anchor_spring_k,
            displacement_fn,
        )
        dist = segment_centroid_distance(R)
        dist_penalty = jnp.where(
            dist > args.max_motif_dist,
            args.spring_k * (dist - args.max_motif_dist) ** 2,
            0.0,
        )
        return base + restraint + dist_penalty - args.motif_contact_bias * motif_contact_score(R)

    @jit
    def integrate(eq_key: jnp.ndarray, sample_key: jnp.ndarray, R0: jnp.ndarray) -> jnp.ndarray:
        init_fn, step_fn = simulate.nvt_langevin(
            biased_energy_fn, shift_fn, args.dt, args.kt, args.gamma
        )
        state = init_fn(eq_key, R0, pseq=pseq, mass=mass)

        def step_body(_idx: int, curr_state):
            return step_fn(curr_state, pseq=pseq)

        state = lax.fori_loop(0, args.eq_steps, step_body, state)
        state = init_fn(sample_key, state.position, pseq=pseq, mass=mass)

        def scan_body(curr_state, _):
            curr_state = lax.fori_loop(0, args.sample_every, step_body, curr_state)
            return curr_state, curr_state.position

        _, traj = lax.scan(
            scan_body,
            state,
            xs=None,
            length=args.sample_steps // args.sample_every,
        )
        return traj

    keys = random.split(random.PRNGKey(args.seed), initial_states.shape[0] * 2)
    eq_keys = keys[: initial_states.shape[0]]
    sample_keys = keys[initial_states.shape[0] :]
    trajectories = vmap(integrate)(eq_keys, sample_keys, initial_states)
    trajectories_np = np.asarray(trajectories)
    initial_states_np = np.asarray(initial_states)

    target_window_np = initial_states_np[0, :target_len]
    results: list[PoseRecord] = []
    invalid_frame_count = 0
    contact_r0 = args.contact_r0
    contact_width = args.contact_width

    for replica_idx, label in enumerate(replica_labels):
        orientation, replica_str = label.split(":")
        replica_num = int(replica_str)
        for frame_index, frame in enumerate(trajectories_np[replica_idx], start=1):
            binder_coords = frame[binder_start:]
            binder_motif = frame[motif_abs_start:motif_abs_end]
            target_motif = target_window_np[target_motif_start_idx:target_motif_end_idx]
            motif_centroid_dist = float(
                np.linalg.norm(
                    binder_motif.mean(axis=0) - target_motif.mean(axis=0)
                )
            )
            motif_contact = segment_contact_score_np(
                target_motif, binder_motif, contact_r0, contact_width
            )
            region_contact = segment_contact_score_np(
                target_motif, binder_coords, contact_r0, contact_width
            )
            binder_self_contact = binder_self_contact_score_np(
                binder_coords, contact_r0, contact_width
            )
            dists = np.linalg.norm(
                binder_coords[:, None, :] - full_protein_ca[None, :, :], axis=2
            )
            min_full_protein_ca_dist = float(np.min(dists))
            clash_penalty = 0.0
            if min_full_protein_ca_dist < 4.0:
                clash_penalty = (4.0 - min_full_protein_ca_dist) ** 2
            total_score = (
                4.0 * motif_contact
                + 1.0 * region_contact
                - 0.10 * motif_centroid_dist
                - 0.30 * binder_self_contact
                - 2.0 * clash_penalty
            )
            scalar_values = (
                total_score,
                motif_contact,
                region_contact,
                motif_centroid_dist,
                binder_self_contact,
                min_full_protein_ca_dist,
            )
            if not all(pymath.isfinite(float(v)) for v in scalar_values):
                invalid_frame_count += 1
                continue
            results.append(
                PoseRecord(
                    orientation=orientation,
                    replica=replica_num,
                    frame_index=frame_index,
                    total_score=total_score,
                    motif_contact=motif_contact,
                    region_contact=region_contact,
                    motif_centroid_dist=motif_centroid_dist,
                    binder_self_contact=binder_self_contact,
                    min_full_protein_ca_dist=min_full_protein_ca_dist,
                    pdb_path="",
                    coords=binder_coords.copy(),
                )
            )

    if not results:
        raise SystemExit(
            f"No finite sampled poses remained after filtering invalid frames "
            f"({invalid_frame_count} invalid frames). Try milder sampling parameters."
        )

    results.sort(
        key=lambda rec: (
            -rec.total_score,
            -rec.motif_contact,
            rec.motif_centroid_dist,
            -rec.min_full_protein_ca_dist,
        )
    )

    params_lines = [
        f"target_pdb={target_pdb}",
        f"resolved_chain={resolved_chain}",
        f"target_window={args.target_window_start}-{args.target_window_end}",
        f"target_window_seq={target_seq}",
        f"target_motif={args.target_motif_start}-{args.target_motif_end}",
        f"binder_seq={args.binder_seq}",
        f"binder_motif={args.binder_motif_start}-{args.binder_motif_end}",
        f"replicas={len(replica_labels)}",
        f"frames_per_replica={args.sample_steps // args.sample_every}",
        f"invalid_frames_filtered={invalid_frame_count}",
        f"finite_frames_retained={len(results)}",
    ]
    (outdir / "params.txt").write_text("\n".join(params_lines) + "\n")

    summary_path = outdir / "pose_summary.tsv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "orientation",
                "replica",
                "frame_index",
                "total_score",
                "motif_contact",
                "region_contact",
                "motif_centroid_dist",
                "binder_self_contact",
                "min_full_protein_ca_dist",
                "pdb_path",
            ]
        )
        for rank, record in enumerate(results[: args.top_k], start=1):
            pdb_name = f"pose_{rank:02d}_{record.orientation}_r{record.replica}_f{record.frame_index}.pdb"
            pdb_path = outdir / pdb_name
            summary_line = (
                f"score={record.total_score:.3f} motif_contact={record.motif_contact:.3f} "
                f"region_contact={record.region_contact:.3f} motif_centroid_dist={record.motif_centroid_dist:.3f} "
                f"min_full_protein_ca_dist={record.min_full_protein_ca_dist:.3f}"
            )
            write_pose_pdb(
                pdb_path,
                target_pdb,
                args.binder_seq,
                record.coords,
                summary_line,
            )
            writer.writerow(
                [
                    rank,
                    record.orientation,
                    record.replica,
                    record.frame_index,
                    f"{record.total_score:.4f}",
                    f"{record.motif_contact:.4f}",
                    f"{record.region_contact:.4f}",
                    f"{record.motif_centroid_dist:.4f}",
                    f"{record.binder_self_contact:.4f}",
                    f"{record.min_full_protein_ca_dist:.4f}",
                    str(pdb_path),
                ]
            )

    all_frames_path = outdir / "all_frames.tsv"
    with all_frames_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "orientation",
                "replica",
                "frame_index",
                "total_score",
                "motif_contact",
                "region_contact",
                "motif_centroid_dist",
                "binder_self_contact",
                "min_full_protein_ca_dist",
            ]
        )
        for record in results:
            writer.writerow(
                [
                    record.orientation,
                    record.replica,
                    record.frame_index,
                    f"{record.total_score:.4f}",
                    f"{record.motif_contact:.4f}",
                    f"{record.region_contact:.4f}",
                    f"{record.motif_centroid_dist:.4f}",
                    f"{record.binder_self_contact:.4f}",
                    f"{record.min_full_protein_ca_dist:.4f}",
                ]
            )

    best = results[0]
    report_lines = [
        f"wrote {min(args.top_k, len(results))} ranked poses to {outdir}",
        f"invalid_frames_filtered={invalid_frame_count}",
        f"finite_frames_retained={len(results)}",
        f"best_orientation={best.orientation}",
        f"best_replica={best.replica}",
        f"best_frame_index={best.frame_index}",
        f"best_total_score={best.total_score:.4f}",
        f"best_motif_contact={best.motif_contact:.4f}",
        f"best_motif_centroid_dist={best.motif_centroid_dist:.4f}",
        f"best_min_full_protein_ca_dist={best.min_full_protein_ca_dist:.4f}",
    ]
    (outdir / "README.txt").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
