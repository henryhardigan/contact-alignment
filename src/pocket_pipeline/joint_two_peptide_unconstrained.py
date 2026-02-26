#!/usr/bin/env python3
"""
Joint unconstrained two-peptide pocket simulation with shared pocket exclusivity.

Both peptides are sampled in the same pocket. Each pocket residue can be assigned
to at most one peptide contact in a given pose.
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np

import importlib.util


def load_pp():
    spec = importlib.util.spec_from_file_location("pp", "src/pocket_pipeline/pocket_pipeline.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    return pp


POCKET19 = [
    "R152",
    "L155",
    "T172",
    "D173",
    "G174",
    "H175",
    "R176",
    "L177",
    "P242",
    "R246",
    "V247",
    "N320",
    "Y323",
    "V344",
    "V360",
    "M362",
    "P363",
    "M364",
    "R365",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecoli-batch", default="src/ecoli_batch")
    ap.add_argument("--entry", default="DPO3B_ECOLI")
    ap.add_argument("--seq-a", default="MDRWLVKGILQWRKI")
    ap.add_argument("--seq-b", default="MDRWLVKWKKKRKI")
    ap.add_argument("--trials", type=int, default=50000)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--min-contacts", type=int, default=7)
    ap.add_argument(
        "--inter-peptide-clash-radius",
        type=float,
        default=2.8,
        help="Hard minimum allowed distance (A) between peptide pseudoatoms across peptides.",
    )
    ap.add_argument(
        "--assign-by",
        choices=["mj_then_dist", "dist_then_mj"],
        default="mj_then_dist",
        help="How to choose the winning peptide contact per pocket residue.",
    )
    ap.add_argument("--seed-a", type=int, default=101)
    ap.add_argument("--seed-b", type=int, default=103)
    ap.add_argument("--out", default="joint_two_peptide_best.tsv")
    return ap.parse_args()


def random_axis(rng):
    z = rng.uniform(-1.0, 1.0)
    t = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(t), r * math.sin(t), z)


def random_center(rng, base, rad):
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        s = x * x + y * y + z * z
        if 0.0 < s <= 1.0:
            break
    rr = rad * (rng.random() ** (1.0 / 3.0))
    return (base[0] + rr * x, base[1] + rr * y, base[2] + rr * z)


def load_residues(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def find_coord(rows, label):
    aa, resseq = label[0], label[1:]
    for r in rows:
        if r["aa"] == aa and r["resseq"] == resseq:
            return np.array([float(r["sc_x"]), float(r["sc_y"]), float(r["sc_z"])], dtype=float)
    raise RuntimeError(f"Residue not found: {label}")


def assign_shared_contacts(pocket, sc_a, seq_a, sc_b, seq_b, cutoff, mj, assign_by):
    contacts = []
    for lab, p, aa in pocket:
        cand = []
        for i, pt in enumerate(sc_a):
            d = float(np.linalg.norm(pt - p))
            if d <= cutoff:
                s = mj.get(seq_a[i], {}).get(aa, mj.get(aa, {}).get(seq_a[i], 0.0))
                cand.append(("A", i + 1, seq_a[i], d, s))
        for i, pt in enumerate(sc_b):
            d = float(np.linalg.norm(pt - p))
            if d <= cutoff:
                s = mj.get(seq_b[i], {}).get(aa, mj.get(aa, {}).get(seq_b[i], 0.0))
                cand.append(("B", i + 1, seq_b[i], d, s))
        if not cand:
            continue
        if assign_by == "mj_then_dist":
            winner = min(cand, key=lambda x: (x[4], x[3]))
        else:
            winner = min(cand, key=lambda x: (x[3], x[4]))
        pep_id, pep_pos, pep_aa, d, s = winner
        contacts.append((lab, aa, pep_id, pep_pos, pep_aa, d, s))
    return contacts


def evaluate(contacts):
    n = len(contacts)
    if n == 0:
        return None
    total = sum(c[6] for c in contacts)
    mean_d = sum(c[5] for c in contacts) / n
    mpc = total / n
    n_a = sum(1 for c in contacts if c[2] == "A")
    n_b = sum(1 for c in contacts if c[2] == "B")
    return {"n": n, "n_a": n_a, "n_b": n_b, "total": total, "mpc": mpc, "mean_d": mean_d}


def has_inter_peptide_clash(sc_a, sc_b, clash_radius):
    r2 = clash_radius * clash_radius
    for pa in sc_a:
        d2 = np.sum((sc_b - pa) ** 2, axis=1)
        if np.any(d2 < r2):
            return True
    return False


def main():
    args = parse_args()
    pp = load_pp()

    rows = load_residues(f"{args.ecoli_batch}/{args.entry}/residues.tsv")
    pocket = [(lab, find_coord(rows, lab), lab[0]) for lab in POCKET19]
    pocket_pts = np.array([p for _, p, _ in pocket], dtype=float)
    cent = pocket_pts.mean(axis=0)
    rmax = max(float(np.linalg.norm(p - cent)) for p in pocket_pts)
    search = rmax + 8.0

    seq_a = list(args.seq_a)
    seq_b = list(args.seq_b)
    mj = pp.load_mj_matrix()

    rng_a = random.Random(args.seed_a)
    rng_b = random.Random(args.seed_b)

    best_pc = None
    best_total = None

    for _ in range(args.trials):
        axis_a = random_axis(rng_a)
        ctr_a = random_center(rng_a, cent, search)
        hel_a = pp.helix_positions(
            ctr_a,
            axis_a,
            len(seq_a),
            1.5 + rng_a.uniform(-0.35, 0.35),
            2.3 + rng_a.uniform(-0.40, 0.40),
            100.0 + rng_a.uniform(-25.0, 25.0),
        )
        sc_a = np.array(pp.peptide_sidechain_pseudoatoms(hel_a, seq_a), dtype=float)

        axis_b = random_axis(rng_b)
        ctr_b = random_center(rng_b, cent, search)
        hel_b = pp.helix_positions(
            ctr_b,
            axis_b,
            len(seq_b),
            1.5 + rng_b.uniform(-0.35, 0.35),
            2.3 + rng_b.uniform(-0.40, 0.40),
            100.0 + rng_b.uniform(-25.0, 25.0),
        )
        sc_b = np.array(pp.peptide_sidechain_pseudoatoms(hel_b, seq_b), dtype=float)

        if has_inter_peptide_clash(sc_a, sc_b, args.inter_peptide_clash_radius):
            continue

        contacts = assign_shared_contacts(
            pocket,
            sc_a,
            seq_a,
            sc_b,
            seq_b,
            args.cutoff,
            mj,
            args.assign_by,
        )
        ev = evaluate(contacts)
        if ev is None or ev["n"] < args.min_contacts:
            continue

        key_pc = (ev["mpc"], ev["mean_d"], -ev["n"])
        key_total = (ev["total"], ev["mean_d"], -ev["n"])

        if best_pc is None or key_pc < best_pc["key"]:
            best_pc = {"key": key_pc, "ev": ev, "contacts": sorted(contacts, key=lambda x: x[5])}
        if best_total is None or key_total < best_total["key"]:
            best_total = {"key": key_total, "ev": ev, "contacts": sorted(contacts, key=lambda x: x[5])}

    if best_pc is None or best_total is None:
        raise SystemExit("No accepted joint poses found.")

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "objective",
                "n_contacts",
                "n_contacts_pepA",
                "n_contacts_pepB",
                "total_mj",
                "mj_per_contact",
                "mean_distance",
                "contact",
            ]
        )
        for objective, best in [("best_mj_per_contact", best_pc), ("best_total_mj", best_total)]:
            ev = best["ev"]
            for c in best["contacts"]:
                lab, _paa, pep_id, pep_pos, pep_aa, d, s = c
                w.writerow(
                    [
                        objective,
                        ev["n"],
                        ev["n_a"],
                        ev["n_b"],
                        f"{ev['total']:.3f}",
                        f"{ev['mpc']:.3f}",
                        f"{ev['mean_d']:.3f}",
                        f"{lab}<->pep{pep_id}:{pep_pos}:{pep_aa};d={d:.3f};mj={s:.1f}",
                    ]
                )

    print(out)
    for name, best in [("best_mj_per_contact", best_pc), ("best_total_mj", best_total)]:
        ev = best["ev"]
        print(
            name,
            "n",
            ev["n"],
            "nA",
            ev["n_a"],
            "nB",
            ev["n_b"],
            "total",
            round(ev["total"], 3),
            "per",
            round(ev["mpc"], 3),
            "mean_d",
            round(ev["mean_d"], 3),
        )


if __name__ == "__main__":
    main()
