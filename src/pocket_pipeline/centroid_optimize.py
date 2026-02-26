#!/usr/bin/env python3
"""
Iterative centroid optimization for peptide helix contact scoring.

Samples candidate centroids within a radius around seed centers, scores them
against a pocket (residues within a radius from the combined motif centroid),
then keeps the top-K centers as seeds for the next iteration.

Note on pocket similarity:
- When comparing pockets across structures, prefer pairwise-distance RMSD
  (RMSD over all intra-pocket pair distances) rather than coordinate RMSD.
"""
import argparse
import csv
import math
import random
from types import SimpleNamespace
import importlib.util


def load_pp():
    spec = importlib.util.spec_from_file_location(
        "pp", "src/pocket_pipeline/pocket_pipeline.py"
    )
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    return pp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecoli-batch", default="src/ecoli_batch")
    ap.add_argument("--entry", default="DPO3B_ECOLI")
    ap.add_argument(
        "--motif-residues",
        default="H175,R176,L177,D173,Y323,V247,P242,P363,M362",
        help="Comma-separated motif residues AA+resseq in the entry.",
    )
    ap.add_argument("--peptide-seq", default="MDRWLVKGILQ")
    ap.add_argument("--peptide-cutoff", type=float, default=6.0)
    ap.add_argument("--peptide-rise", type=float, default=1.5)
    ap.add_argument("--peptide-radius", type=float, default=2.3)
    ap.add_argument("--peptide-rot", type=float, default=100.0)
    ap.add_argument("--min-contacts", type=int, default=7)
    ap.add_argument(
        "--require-residues",
        default="",
        help="Comma-separated AA+resseq that must be in contact (e.g., H175,Y323).",
    )
    ap.add_argument(
        "--mj-sum-max",
        type=float,
        default=None,
        help="Require mj_sum < this value (e.g., -200).",
    )
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--samples-per-iter", type=int, default=50)
    ap.add_argument(
        "--seed-radius",
        type=float,
        default=5.0,
        help="Sampling radius around each seed center (Å).",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--weight-dist", type=float, default=2.0)
    ap.add_argument("--weight-mj", type=float, default=0.5)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Contact reward weight in combined score (combined - alpha * n_contacts).",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--seed-center",
        default=None,
        help="Optional seed center as 'x,y,z' to start from (overrides motif centroid).",
    )
    ap.add_argument(
        "--require-pep-idx",
        default="",
        help="Comma-separated peptide indices (1-based) that must be used by at least one contact.",
    )
    ap.add_argument("--out", default="centroid_optimize.tsv")
    return ap.parse_args()


def main():
    args = parse_args()
    pp = load_pp()

    # Load residues for entry
    residues_path = f"{args.ecoli_batch}/{args.entry}/residues.tsv"
    res = []
    with open(residues_path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            res.append(row)

    def get_row(aa, resseq):
        for row in res:
            if row["aa"] == aa and row["resseq"] == str(resseq):
                return row
        return None

    motif_toks = [t.strip() for t in args.motif_residues.split(",") if t.strip()]
    motif_rows = []
    for t in motif_toks:
        aa = t[0]
        rs = t[1:]
        row = get_row(aa, rs)
        if row is None:
            raise SystemExit(f"Motif residue not found: {t}")
        motif_rows.append(row)

    motif_pts = [
        (float(r["sc_x"]), float(r["sc_y"]), float(r["sc_z"]))
        for r in motif_rows
    ]
    centroid = (
        sum(p[0] for p in motif_pts) / len(motif_pts),
        sum(p[1] for p in motif_pts) / len(motif_pts),
        sum(p[2] for p in motif_pts) / len(motif_pts),
    )
    pocket_radius = max(math.dist(centroid, p) for p in motif_pts)

    # Pocket residues within radius
    pocket = []
    for row in res:
        p = (float(row["sc_x"]), float(row["sc_y"]), float(row["sc_z"]))
        if math.dist(centroid, p) <= pocket_radius + 1e-9:
            pocket.append(row)

    pocket_points = [
        (float(r["sc_x"]), float(r["sc_y"]), float(r["sc_z"])) for r in pocket
    ]
    pocket_aas = [r["aa"] for r in pocket]
    pocket_labels = [f"{r['aa']}{r['resseq']}" for r in pocket]

    axis = pp.pca_axis(pocket_points)
    pep = list(args.peptide_seq)
    mj = pp.load_mj_matrix()

    def mj_score(a, b):
        return mj.get(a, {}).get(b, mj.get(b, {}).get(a, 0.0))

    def helix_positions(center):
        return pp.helix_positions(
            center, axis, len(pep), args.peptide_rise, args.peptide_radius, args.peptide_rot
        )

    req = [t.strip() for t in args.require_residues.split(",") if t.strip()]
    req_set = set(req)
    req_pep = [t.strip() for t in args.require_pep_idx.split(",") if t.strip()]
    req_pep_set = set(int(x) for x in req_pep) if req_pep else set()
    def contact_stats(center):
        helix = helix_positions(center)
        total_s = 0.0
        dists = []
        real_contact_set = set()
        used_pep = set()
        for label, p, aa in zip(pocket_labels, pocket_points, pocket_aas):
            best = None
            best_i = None
            for i, hp in enumerate(helix):
                d = math.dist(p, hp)
                if d <= args.peptide_cutoff:
                    s = mj_score(pep[i], aa)
                    key = (s, d)
                    if best is None or key < best:
                        best = key
                        best_i = i
            if best is not None:
                total_s += best[0]
                dists.append(best[1])
                real_contact_set.add(label)
                used_pep.add(best_i + 1)
        n_contacts = len(dists)
        if n_contacts < args.min_contacts:
            return None
        if req:
            # require specific residues to be within cutoff
            if not req_set.issubset(real_contact_set):
                return None
        if req_pep_set:
            if not req_pep_set.issubset(used_pep):
                return None
        if args.mj_sum_max is not None and total_s >= args.mj_sum_max:
            return None
        mean_d = sum(dists) / n_contacts
        return total_s, n_contacts, mean_d

    random.seed(args.seed)
    if args.seed_center:
        parts = [p.strip() for p in args.seed_center.split(",")]
        if len(parts) != 3:
            raise SystemExit("Invalid --seed-center (expected x,y,z)")
        seeds = [(float(parts[0]), float(parts[1]), float(parts[2]))]
    else:
        seeds = [centroid]

    all_rows = []
    with open(args.out, "w") as f:
        f.write(
            "iter\trank\tcx\tcy\tcz\tmean_d\tmj_sum\tn_contacts\tmj_metric\tmj_norm\tcombined\n"
        )
        for it in range(1, args.iterations + 1):
            candidates = []
            for s in seeds:
                for _ in range(args.samples_per_iter):
                    # sample in sphere around seed
                    while True:
                        x = random.uniform(-1, 1)
                        y = random.uniform(-1, 1)
                        z = random.uniform(-1, 1)
                        if x * x + y * y + z * z <= 1:
                            break
                    r = args.seed_radius * (random.random() ** (1 / 3))
                    c = (s[0] + r * x, s[1] + r * y, s[2] + r * z)
                    if math.dist(c, centroid) > pocket_radius + 1e-9:
                        continue
                    stats = contact_stats(c)
                    if stats is None:
                        continue
                    mj_sum, n_contacts, mean_d = stats
                    mj_metric = (-mj_sum / n_contacts) if n_contacts else 0.0
                    candidates.append((c, mean_d, mj_sum, n_contacts, mj_metric))

            if not candidates:
                raise SystemExit("No candidates met constraints in iteration %d" % it)

            # normalize MJ metric for this iteration
            vals = [c[4] for c in candidates]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var) if var > 0 else 1.0

            scored = []
            for c, mean_d, mj_sum, n_contacts, mj_metric in candidates:
                mj_norm = (mj_metric - mean) / std
                combined = (
                    args.weight_dist * mean_d
                    + args.weight_mj * mj_norm
                    - args.alpha * n_contacts
                )
                scored.append((combined, c, mean_d, mj_sum, n_contacts, mj_metric, mj_norm))

            scored.sort(key=lambda t: t[0])
            top = scored[: max(1, args.top_k)]
            for rank, (combined, c, mean_d, mj_sum, n_contacts, mj_metric, mj_norm) in enumerate(
                top, start=1
            ):
                row = {
                    "iter": it,
                    "rank": rank,
                    "cx": c[0],
                    "cy": c[1],
                    "cz": c[2],
                    "mean_d": mean_d,
                    "mj_sum": mj_sum,
                    "n_contacts": n_contacts,
                    "mj_metric": mj_metric,
                    "mj_norm": mj_norm,
                    "combined": combined,
                }
                all_rows.append(row)
                f.write(
                    f"{it}\t{rank}\t{c[0]:.3f}\t{c[1]:.3f}\t{c[2]:.3f}\t"
                    f"{mean_d:.3f}\t{mj_sum:.3f}\t{n_contacts}\t{mj_metric:.3f}\t"
                    f"{mj_norm:.3f}\t{combined:.3f}\n"
                )

            seeds = [t[1] for t in top]

    # Also write a globally ranked view (rank = global order by combined)
    ranked_out = args.out.replace(".tsv", "_ranked.tsv") if args.out.endswith(".tsv") else args.out + "_ranked.tsv"
    all_rows.sort(key=lambda r: r["combined"])
    with open(ranked_out, "w") as f:
        f.write(
            "rank\titer\tcx\tcy\tcz\tmean_d\tmj_sum\tn_contacts\tmj_metric\tmj_norm\tcombined\n"
        )
        for i, r in enumerate(all_rows, start=1):
            f.write(
                f"{i}\t{r['iter']}\t{r['cx']:.3f}\t{r['cy']:.3f}\t{r['cz']:.3f}\t"
                f"{r['mean_d']:.3f}\t{r['mj_sum']:.3f}\t{r['n_contacts']}\t"
                f"{r['mj_metric']:.3f}\t{r['mj_norm']:.3f}\t{r['combined']:.3f}\n"
            )

    print(args.out)
    print(ranked_out)


if __name__ == "__main__":
    main()
