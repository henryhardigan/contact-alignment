#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

DEFAULT_FULL19_TEMPLATE = (
    "R152,L155,T172,D173,G174,H175,R176,L177,P242,R246,V247,N320,Y323,"
    "V344,V360,M362,P363,M364,R365"
)


def dist(a, b):
    return math.dist(a, b)


def pair_count(k):
    return (k * (k - 1)) // 2


def load_blosum(path: str):
    rows = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            rows.append(s)
    if not rows:
        raise SystemExit(f"Empty BLOSUM file: {path}")
    header = rows[0].split()
    mat = {}
    for ln in rows[1:]:
        p = ln.split()
        aa = p[0]
        for bb, vv in zip(header, p[1:]):
            mat[(aa, bb)] = int(vv)
    return mat


def blosum_score(mat, a, b):
    return mat.get((a, b), mat.get((b, a), -99))


def parse_residue_token(tok: str):
    tok = tok.strip()
    if not tok:
        return None
    aa = tok[0]
    rs = tok[1:]
    if not rs.isdigit():
        raise ValueError(f"Invalid residue token: {tok}")
    return aa, rs


def parse_residue_list(s: str):
    out = []
    for tok in [x.strip() for x in s.split(",") if x.strip()]:
        out.append(parse_residue_token(tok))
    return out


def load_residue_rows(path: Path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append({
                "aa": row["aa"],
                "resseq": row["resseq"],
                "res_id": row["res_id"],
                "xyz": (float(row["sc_x"]), float(row["sc_y"]), float(row["sc_z"])),
            })
    return rows


def find_residue(rows, aa, rs):
    for r in rows:
        if r["aa"] == aa and r["resseq"] == rs:
            return r
    return None


@dataclass
class SolverResult:
    best_sse: float | None
    best_assign: list | None
    nodes: int
    complete: bool


class ExactSubsetSolver:
    def __init__(self, tmpl_points, tmpl_labels, pool_rows, domains, progress_every=0, max_nodes=0):
        self.tp = tmpl_points
        self.tl = tmpl_labels
        self.pool = pool_rows
        self.domains = domains
        self.k = len(tmpl_points)
        self.progress_every = int(progress_every or 0)
        self.max_nodes = int(max_nodes or 0)
        self._nodes = 0
        self._complete = True

        # pair reference distances in template subset
        self.tdist = {}
        for i in range(self.k - 1):
            for j in range(i + 1, self.k):
                self.tdist[(i, j)] = dist(self.tp[i], self.tp[j])

        # Precompute pairwise error matrices and lower-bound helpers.
        # key (i,j) with i<j:
        #  e2[ai][bj], row_min[ai], col_min[bj], pair_min
        self.pair = {}
        for i in range(self.k - 1):
            di = self.domains[i]
            for j in range(i + 1, self.k):
                dj = self.domains[j]
                ref = self.tdist[(i, j)]
                e2 = [[0.0] * len(dj) for _ in range(len(di))]
                row_min = [float("inf")] * len(di)
                col_min = [float("inf")] * len(dj)
                pmin = float("inf")
                for ai, ri in enumerate(di):
                    pi = self.pool[ri]["xyz"]
                    for bj, rj in enumerate(dj):
                        pj = self.pool[rj]["xyz"]
                        e = dist(pi, pj) - ref
                        vv = e * e
                        e2[ai][bj] = vv
                        if vv < row_min[ai]:
                            row_min[ai] = vv
                        if vv < col_min[bj]:
                            col_min[bj] = vv
                        if vv < pmin:
                            pmin = vv
                self.pair[(i, j)] = (e2, row_min, col_min, pmin)

        # fixed LB among unassigned-unassigned by pair minima
        self.uu_pair_min = {}
        for i in range(self.k - 1):
            for j in range(i + 1, self.k):
                self.uu_pair_min[(i, j)] = self.pair[(i, j)][3]

    def _pair_e2(self, i, ai, j, bj):
        if i < j:
            e2, _r, _c, _p = self.pair[(i, j)]
            return e2[ai][bj]
        e2, _r, _c, _p = self.pair[(j, i)]
        return e2[bj][ai]

    def _row_min(self, i, ai, j):
        if i < j:
            _e2, row_min, _c, _p = self.pair[(i, j)]
            return row_min[ai]
        _e2, _r, col_min, _p = self.pair[(j, i)]
        return col_min[ai]

    def solve(self):
        # position order: smallest domain first
        order = sorted(range(self.k), key=lambda x: len(self.domains[x]))
        pos_to_ord = {p: i for i, p in enumerate(order)}

        assigned = [None] * self.k  # stores domain-index for each position
        used_pool = set()
        best_sse = None
        best_assign = None

        def recurse(depth, cur_sse):
            nonlocal best_sse, best_assign
            self._nodes += 1
            if self.max_nodes > 0 and self._nodes > self.max_nodes:
                self._complete = False
                return
            if self.progress_every > 0 and self._nodes % self.progress_every == 0:
                print(f"progress nodes={self._nodes} depth={depth} best_sse={best_sse}", flush=True)

            if depth == self.k:
                if best_sse is None or cur_sse < best_sse:
                    best_sse = cur_sse
                    best_assign = assigned[:]
                return

            p = order[depth]
            dom_p = self.domains[p]

            # Precompute unassigned set for LB.
            unassigned = order[depth:]

            for ai, ridx in enumerate(dom_p):
                if ridx in used_pool:
                    continue

                # Incremental SSE vs already assigned positions.
                add = 0.0
                feasible = True
                for d2 in range(depth):
                    q = order[d2]
                    aq = assigned[q]
                    if aq is None:
                        continue
                    add += self._pair_e2(p, ai, q, aq)

                if not feasible:
                    continue
                nsse = cur_sse + add
                if best_sse is not None and nsse >= best_sse:
                    continue

                # Lower bound for remaining terms:
                # 1) assigned-vs-unassigned using row minima to current fixed candidate
                lb = 0.0
                for du in range(depth + 1, self.k):
                    u = order[du]
                    # assigned side includes prior + current p
                    b = self._row_min(u, 0, u) if False else 0.0
                    # against current p
                    b = self._row_min(p, ai, u)
                    # against previously assigned
                    for d2 in range(depth):
                        q = order[d2]
                        aq = assigned[q]
                        if aq is None:
                            continue
                        b += self._row_min(q, aq, u)
                    lb += b

                # 2) unassigned-vs-unassigned pair minima
                for a in range(depth + 1, self.k - 1):
                    u = order[a]
                    for b in range(a + 1, self.k):
                        v = order[b]
                        key = (u, v) if u < v else (v, u)
                        lb += self.uu_pair_min[key]

                if best_sse is not None and nsse + lb >= best_sse:
                    continue

                assigned[p] = ai
                used_pool.add(ridx)
                recurse(depth + 1, nsse)
                used_pool.remove(ridx)
                assigned[p] = None

                if self.max_nodes > 0 and self._nodes > self.max_nodes:
                    return

        recurse(0, 0.0)

        if best_assign is None:
            return SolverResult(None, None, self._nodes, self._complete)

        # convert assignment domain-indices -> pool indices by template position order
        out = [None] * self.k
        for p in range(self.k):
            ai = best_assign[p]
            out[p] = self.domains[p][ai]
        return SolverResult(best_sse, out, self._nodes, self._complete)


def main():
    ap = argparse.ArgumentParser(description="Prove globally optimal subset+assignment (fixed k) via exact branch-and-bound.")
    ap.add_argument("--ecoli-batch", default="src/ecoli_batch")
    ap.add_argument("--template-entry", default="DPO3B_ECOLI")
    ap.add_argument("--target-entry", default="SURA_ECOLI")
    ap.add_argument("--template-residues", default=DEFAULT_FULL19_TEMPLATE)
    ap.add_argument("--seed-residues", default="E379,H289,R288,F321,L377,L380,P346,L357,L360,I378,A367",
                    help="Target residues defining local neighborhood center(s).")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--k", type=int, required=True, help="Subset size to optimize exactly.")
    ap.add_argument("--blosum-matrix", default="src/pocket_pipeline/blosum62.txt")
    ap.add_argument("--blosum-threshold", type=int, default=0, help="Minimum substitution score (e.g., 0).")
    ap.add_argument("--progress-every", type=int, default=0, help="Print node progress every N nodes (0 disables).")
    ap.add_argument("--progress-subsets", type=int, default=500,
                    help="Print outer-loop subset progress every N subsets (0 disables).")
    ap.add_argument("--max-nodes", type=int, default=0, help="Safety cap; if hit, proof is incomplete.")
    ap.add_argument("--out", default="data/prove_global_optimum.tsv")
    args = ap.parse_args()

    mat = load_blosum(args.blosum_matrix)

    trows = load_residue_rows(Path(args.ecoli_batch) / args.template_entry / "residues.tsv")
    srows = load_residue_rows(Path(args.ecoli_batch) / args.target_entry / "residues.tsv")

    tmpl_tokens = parse_residue_list(args.template_residues)
    if args.k < 2 or args.k > len(tmpl_tokens):
        raise SystemExit(f"--k must be in [2,{len(tmpl_tokens)}]")

    template = []
    for aa, rs in tmpl_tokens:
        r = find_residue(trows, aa, rs)
        if r is None:
            raise SystemExit(f"Template residue not found: {aa}{rs}")
        template.append(r)

    seed_tokens = parse_residue_list(args.seed_residues) if args.seed_residues else []
    seed_pts = []
    for aa, rs in seed_tokens:
        r = find_residue(srows, aa, rs)
        if r is None:
            raise SystemExit(f"Seed residue not found in target: {aa}{rs}")
        seed_pts.append(r)
    if seed_pts:
        pool = [r for r in srows if any(dist(r["xyz"], q["xyz"]) <= args.radius for q in seed_pts)]
    else:
        pool = list(srows)
    if not pool:
        raise SystemExit("No target residues in pool after radius filter.")

    # Candidate domains for each template position over target pool indices.
    domains_all = []
    for tr in template:
        dom = []
        for i, sr in enumerate(pool):
            if blosum_score(mat, tr["aa"], sr["aa"]) >= args.blosum_threshold:
                dom.append(i)
        domains_all.append(dom)

    # Enumerate template subsets and solve each exactly.
    best_global_sse = None
    best_global = None
    subsets_total = 0
    subsets_pruned_empty = 0
    subsets_pruned_lb = 0
    nodes_total = 0
    complete = True

    all_idx = list(range(len(template)))
    for subset in combinations(all_idx, args.k):
        subsets_total += 1
        if args.progress_subsets > 0 and subsets_total % args.progress_subsets == 0:
            print(
                f"subset_progress {subsets_total} best_sse={best_global_sse} "
                f"pruned_empty={subsets_pruned_empty} pruned_lb={subsets_pruned_lb} nodes={nodes_total}",
                flush=True,
            )
        subset = list(subset)
        tmpl_points = [template[i]["xyz"] for i in subset]
        tmpl_labels = [f"{template[i]['aa']}{template[i]['resseq']}" for i in subset]
        domains = [domains_all[i] for i in subset]

        # Empty-domain prune.
        if any(len(d) == 0 for d in domains):
            subsets_pruned_empty += 1
            continue

        # Subset-level LB prune: sum pair minima (ignoring uniqueness).
        if best_global_sse is not None:
            lb = 0.0
            for a in range(args.k - 1):
                ia = subset[a]
                for b in range(a + 1, args.k):
                    ib = subset[b]
                    tref = dist(template[ia]["xyz"], template[ib]["xyz"])
                    dmin = float("inf")
                    for ra in domains[a]:
                        pa = pool[ra]["xyz"]
                        for rb in domains[b]:
                            pb = pool[rb]["xyz"]
                            e = dist(pa, pb) - tref
                            vv = e * e
                            if vv < dmin:
                                dmin = vv
                    lb += dmin
                    if lb >= best_global_sse:
                        break
                if lb >= best_global_sse:
                    break
            if lb >= best_global_sse:
                subsets_pruned_lb += 1
                continue

        solver = ExactSubsetSolver(
            tmpl_points=tmpl_points,
            tmpl_labels=tmpl_labels,
            pool_rows=pool,
            domains=domains,
            progress_every=args.progress_every,
            max_nodes=args.max_nodes,
        )
        res = solver.solve()
        nodes_total += res.nodes
        if not res.complete:
            complete = False
        if res.best_sse is None:
            continue
        if best_global_sse is None or res.best_sse < best_global_sse:
            best_global_sse = res.best_sse
            best_global = (subset, res.best_assign)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["target_entry", "k", "best_sse", "best_rmsd", "ratio_k_over_rmsd", "proven_complete",
                    "subsets_total", "subsets_pruned_empty", "subsets_pruned_lb", "nodes_total", "mapping"])
        if best_global is None:
            w.writerow([args.target_entry, args.k, "", "", "", int(complete),
                        subsets_total, subsets_pruned_empty, subsets_pruned_lb, nodes_total, ""])
        else:
            subset, assign = best_global
            rmsd = math.sqrt(best_global_sse / pair_count(args.k))
            ratio = args.k / rmsd if rmsd > 0 else float("inf")
            mapping = []
            for local_i, pool_i in enumerate(assign):
                ti = subset[local_i]
                tr = template[ti]
                sr = pool[pool_i]
                mapping.append(f"{tr['aa']}{tr['resseq']}->{sr['aa']}{sr['resseq']}")
            w.writerow([args.target_entry, args.k, f"{best_global_sse:.8f}", f"{rmsd:.8f}", f"{ratio:.8f}",
                        int(complete), subsets_total, subsets_pruned_empty, subsets_pruned_lb, nodes_total,
                        ";".join(mapping)])

    print(out)
    if best_global is None:
        print("status no_feasible_solution")
    else:
        subset, _assign = best_global
        rmsd = math.sqrt(best_global_sse / pair_count(args.k))
        ratio = args.k / rmsd if rmsd > 0 else float("inf")
        print("best_rmsd", f"{rmsd:.6f}")
        print("best_ratio", f"{ratio:.6f}")
        print("proven_complete", int(complete))
        print("subsets_total", subsets_total)
        print("subsets_pruned_empty", subsets_pruned_empty)
        print("subsets_pruned_lb", subsets_pruned_lb)
        print("nodes_total", nodes_total)


if __name__ == "__main__":
    main()
