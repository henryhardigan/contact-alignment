#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import time
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
from itertools import combinations
from itertools import permutations
try:
    import numpy as np
except Exception:
    np = None

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_FULL19_TEMPLATE = (
    "R152,L155,T172,D173,G174,H175,R176,L177,P242,R246,V247,N320,Y323,"
    "V344,V360,M362,P363,M364,R365"
)


def split_multi(s, seps=";|,"):
    if s is None:
        return []
    out = [str(s)]
    for sep in seps:
        nxt = []
        for part in out:
            nxt.extend(part.split(sep))
        out = nxt
    return [x.strip() for x in out if x.strip()]


def load_accession_alias_map(path):
    acc_to_entries = {}
    entry_to_acc = {}
    if not path or not os.path.exists(path):
        return acc_to_entries, entry_to_acc
    with open(path) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            acc = (row.get('uniprot_accession') or '').strip()
            if not acc:
                continue
            pe = (row.get('primary_entry') or '').strip()
            all_e = split_multi(row.get('all_entries'), seps=';')
            entries = []
            if pe:
                entries.append(pe)
            entries.extend(all_e)
            if acc not in acc_to_entries:
                acc_to_entries[acc] = set()
            for e in entries:
                if not e:
                    continue
                acc_to_entries[acc].add(e)
                if e not in entry_to_acc:
                    entry_to_acc[e] = set()
                entry_to_acc[e].add(acc)
    return acc_to_entries, entry_to_acc


def expand_alias_tokens(tokens, acc_to_entries, entry_to_acc, rounds=2):
    cur = set(tokens)
    for _ in range(rounds):
        add = set()
        for t in list(cur):
            add |= acc_to_entries.get(t, set())
            add |= entry_to_acc.get(t, set())
        if add.issubset(cur):
            break
        cur |= add
    return cur


def preferred_accession(accs):
    if not accs:
        return ''
    # Prefer 6-character accessions when available.
    six = sorted([a for a in accs if len(a) == 6])
    if six:
        return six[0]
    return sorted(accs)[0]


def canonical_uniprot_id(uid, acc_to_entries, entry_to_acc):
    if not uid:
        return uid
    uid = uid.strip().upper()
    exp = expand_alias_tokens({uid}, acc_to_entries, entry_to_acc, rounds=2)
    accs = {t for t in exp if t in acc_to_entries}
    if accs:
        return preferred_accession(accs)
    if uid in entry_to_acc and entry_to_acc[uid]:
        return preferred_accession({a.strip().upper() for a in entry_to_acc[uid] if a})
    return uid


def is_rabit_contaminant(uid, entry, acc_to_entries):
    e = (entry or "").upper()
    if "_RABIT" in e:
        return True
    u = (uid or "").upper()
    aliases = set()
    if u in acc_to_entries:
        aliases |= {a.upper() for a in acc_to_entries.get(u, set())}
    # Also check accession-normalized key if different casing/form.
    for k in (u,):
        if k in acc_to_entries:
            aliases |= {a.upper() for a in acc_to_entries.get(k, set())}
    return any("_RABIT" in a for a in aliases)


def parse_pattern_tokens(pattern: str):
    toks = pattern.replace(',', ' ').split()
    out = []
    for tok in toks:
        if tok.startswith('[') and tok.endswith(']'):
            out.append(set(tok[1:-1]))
        else:
            out.append(set(tok))
    return out


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
    for tok in [x.strip() for x in s.split(',') if x.strip()]:
        out.append(parse_residue_token(tok))
    return out


def split_clusters(s: str):
    return [x.strip() for x in s.split('|') if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def parse_index_spec_1based(spec: str, nmax: int):
    spec = (spec or "").strip().lower()
    if not spec:
        return set()
    if spec == "all":
        return set(range(nmax))
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a = int(a.strip()); b = int(b.strip())
            if a > b:
                a, b = b, a
            for k in range(a, b + 1):
                if 1 <= k <= nmax:
                    out.add(k - 1)
        else:
            k = int(part)
            if 1 <= k <= nmax:
                out.add(k - 1)
    return out


def dist(a, b):
    return math.dist(a, b)


def pair_vector(points):
    vec = []
    n = len(points)
    for i in range(n - 1):
        for j in range(i + 1, n):
            vec.append(dist(points[i], points[j]))
    return vec


def pairdist_rmsd_points(points_a, points_b):
    vec_a = pair_vector(points_a)
    vec_b = pair_vector(points_b)
    if not vec_a:
        return 0.0
    sse = 0.0
    for da, db in zip(vec_a, vec_b):
        e = da - db
        sse += e * e
    return math.sqrt(sse / len(vec_a))


def load_blosum_matrix(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"BLOSUM matrix not found: {path}")
    lines = []
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            lines.append(s)
    if not lines:
        raise SystemExit(f"Empty BLOSUM matrix file: {path}")
    header = lines[0].split()
    mat = {}
    for ln in lines[1:]:
        toks = ln.split()
        row = toks[0]
        vals = toks[1:]
        if len(vals) != len(header):
            continue
        mat[row] = {col: int(v) for col, v in zip(header, vals)}
    return mat


def _mj_matrix_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    preferred = root / "refs" / "mj_matrix.csv"
    if preferred.exists():
        return preferred
    return root / "mj_matrix.csv"


def load_mj_matrix():
    path = _mj_matrix_path()
    if not path.exists():
        raise SystemExit(f"MJ matrix not found: {path}")
    mj = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        aas = [x.strip().upper() for x in header[1:]]
        for row in r:
            a = row[0].strip().upper()
            vals = row[1:]
            mj[a] = {}
            for b, v in zip(aas, vals):
                try:
                    mj[a][b] = float(v)
                except Exception:
                    continue
    return mj


def mj_score(mj, a, b):
    return mj.get(a, {}).get(b, mj.get(b, {}).get(a, 0.0))


def degen_set_from_blosum(matrix, aa: str, thr: int):
    row = matrix.get(aa)
    if row is None:
        return {aa}
    out = {b for b, s in row.items() if b in AA20 and s >= thr}
    if aa in AA20:
        out.add(aa)
    return out if out else ({aa} if aa in AA20 else set())


def parse_peptide_index_map(spec: str, n: int):
    toks = [t.strip() for t in (spec or "").split(',') if t.strip()]
    if len(toks) != n:
        raise SystemExit(
            f"--degen-peptide-map length ({len(toks)}) must equal number of template positions ({n})"
        )
    out = []
    for t in toks:
        if t in ('0', '-', 'NA', 'na'):
            out.append(None)
            continue
        k = int(t)
        if k <= 0:
            out.append(None)
        else:
            out.append(k - 1)
    return out


def degen_set_topk_from_blosum(matrix, aa: str, k: int, min_score: int = 0):
    row = matrix.get(aa)
    if row is None or k <= 0:
        return {aa} if aa in AA20 else set()
    vals = [(b, s) for b, s in row.items() if b in AA20 and s >= min_score]
    if not vals:
        return {aa} if aa in AA20 else set()
    vals.sort(key=lambda t: (-t[1], t[0]))
    if k >= len(vals):
        out = {b for b, _ in vals}
        return out if out else ({aa} if aa in AA20 else set())
    kth = vals[k - 1][1]
    out = {b for b, s in vals if s >= kth}
    if aa in AA20:
        out.add(aa)
    return out if out else ({aa} if aa in AA20 else set())


def load_entry_to_uid(ecoli_batch: str):
    entry_to_uid = {}
    for meta in glob.glob(os.path.join(ecoli_batch, '*', 'meta.json')):
        try:
            with open(meta) as f:
                data = json.load(f)
            pdb = data.get('pdb', '')
            uid = os.path.splitext(os.path.basename(pdb))[0]
            entry = os.path.basename(os.path.dirname(meta))
            if uid:
                entry_to_uid[entry] = uid
        except Exception:
            continue
    return entry_to_uid


def load_residue_rows(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows.append({
                'aa': r['aa'],
                'resseq': r['resseq'],
                'res_id': r['res_id'],
                'xyz': (float(r['sc_x']), float(r['sc_y']), float(r['sc_z'])),
            })
    return rows


def read_template_points(ecoli_batch, entry, residues):
    path = os.path.join(ecoli_batch, entry, 'residues.tsv')
    if not os.path.exists(path):
        raise RuntimeError(f"Template entry not found: {path}")
    idx = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            idx[(r['aa'], r['resseq'])] = (float(r['sc_x']), float(r['sc_y']), float(r['sc_z']))
    pts = []
    missing = []
    for aa, rs in residues:
        p = idx.get((aa, rs))
        if p is None:
            missing.append(f"{aa}{rs}")
        else:
            pts.append(p)
    if missing:
        raise RuntimeError(f"Missing template residues in {entry}: {', '.join(missing)}")
    return pts


def enumerate_cluster_matches(residues, allowed_sets, template_points, cutoff, max_candidates, unordered=False):
    n = len(allowed_sets)
    def run_for_template_points(tp):
        slot_lists = [[r for r in residues if r['aa'] in allowed_sets[i]] for i in range(n)]
        if any(not sl for sl in slot_lists):
            return []

        if n == 1:
            # Do not truncate singleton clusters here; truncation by residue number can
            # drop the true mapping before geometry-based prefilters are applied.
            return [(0.0, [r]) for r in slot_lists[0]]

        tvec = pair_vector(tp)
        pairs = list(combinations(range(n), 2))
        pair_refs = {(i, j): tvec[k] for k, (i, j) in enumerate(pairs)}
        sse_limit = len(tvec) * (cutoff ** 2)

        chosen = [None] * n
        chosen_ids = set()
        out = []
        worst_sse_kept = None

        def maybe_keep_solution(sse):
            nonlocal worst_sse_kept
            pd = math.sqrt(sse / len(tvec))
            if pd > cutoff:
                return
            cand = (pd, list(chosen))
            if max_candidates <= 0:
                out.append(cand)
                return
            if len(out) < max_candidates:
                out.append(cand)
                if len(out) == max_candidates:
                    # track worst kept SSE for branch-and-bound pruning
                    worst_sse_kept = max((x[0] * x[0] * len(tvec) for x in out), default=None)
                return
            # Replace current worst only if this candidate is better
            if worst_sse_kept is None or sse >= worst_sse_kept:
                return
            wi = max(range(len(out)), key=lambda i: out[i][0])
            out[wi] = cand
            worst_sse_kept = max((x[0] * x[0] * len(tvec) for x in out), default=None)

        def rec(i, sse):
            if sse > sse_limit:
                return
            # Branch-and-bound: if we already have top-K and partial SSE cannot
            # beat the current worst kept SSE, no completion can improve it.
            if max_candidates > 0 and worst_sse_kept is not None and sse >= worst_sse_kept:
                return
            if i == n:
                maybe_keep_solution(sse)
                return
            for r in slot_lists[i]:
                rid = r['res_id']
                if rid in chosen_ids:
                    continue
                add = 0.0
                for j in range(i):
                    d_ref = pair_refs[(j, i)]
                    d_cur = dist(chosen[j]['xyz'], r['xyz'])
                    e = d_cur - d_ref
                    add += e * e
                new_sse = sse + add
                if new_sse > sse_limit:
                    continue
                chosen[i] = r
                chosen_ids.add(rid)
                rec(i + 1, new_sse)
                chosen_ids.remove(rid)
                chosen[i] = None

        rec(0, 0.0)
        # keep deterministic order by pd
        out.sort(key=lambda t: t[0])
        return out

    if not unordered or n <= 1:
        out = run_for_template_points(template_points)
        out.sort(key=lambda t: t[0])
        return out[:max_candidates] if max_candidates > 0 else out

    # Unordered cluster mode: allow free role-to-template-position assignment
    # by trying all permutations of template-point indices for this cluster.
    # Practical for small n (e.g., 2-4 role clusters).
    if n > 8:
        raise RuntimeError(f"unordered cluster length {n} is too large (max 8)")
    dedup = {}
    for perm in permutations(range(n)):
        tp = [template_points[i] for i in perm]
        # sel is returned in role order; convert to template-index order so
        # downstream inter-cluster geometry uses consistent cluster offsets.
        # perm[role_idx] = template_idx for this permutation.
        for pd, sel in run_for_template_points(tp):
            sel_t = [None] * n
            for role_idx, template_idx in enumerate(perm):
                sel_t[template_idx] = sel[role_idx]
            key = tuple(r['res_id'] for r in sel_t)
            prev = dedup.get(key)
            if prev is None or pd < prev[0]:
                dedup[key] = (pd, sel_t)
    out = list(dedup.values())
    out.sort(key=lambda t: t[0])
    return out[:max_candidates] if max_candidates > 0 else out


def find_assignment_singletons(
    candidates_by_cluster,
    inter_pair_refs=None,
    total_inter_pairs=0,
    inter_cutoff=None,
    inter_pair_tol=None,
):
    m = len(candidates_by_cluster)
    if m == 0:
        return None

    residues = [[sel[0] for _pd, sel in cand] for cand in candidates_by_cluster]
    if any(len(r) == 0 for r in residues):
        return None

    ids = [[r['res_id'] for r in rr] for rr in residues]
    xyz = [[r['xyz'] for r in rr] for rr in residues]
    use_np = np is not None
    if use_np:
        xyz_np = [np.asarray([r['xyz'] for r in rr], dtype=float) for rr in residues]
        refm = np.zeros((m, m), dtype=float)
        if inter_pair_refs:
            for i in range(m - 1):
                for j in range(i + 1, m):
                    refm[i, j] = inter_pair_refs.get((i, j), 0.0)
                    refm[j, i] = refm[i, j]

    def ref_dist(i, j):
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        return inter_pair_refs.get((a, b), 0.0) if inter_pair_refs else 0.0

    def err(i, ai, j, aj):
        if use_np:
            d = float(np.linalg.norm(xyz_np[i][ai] - xyz_np[j][aj]) - refm[i, j])
        else:
            d = dist(xyz[i][ai], xyz[j][aj]) - ref_dist(i, j)
        return d, d * d

    def err_vec(j, dom_idx, i, ai):
        # Vectorized error for candidates in cluster j vs fixed assignment (i, ai)
        if use_np:
            arr = xyz_np[j][dom_idx] - xyz_np[i][ai]
            d = np.sqrt((arr * arr).sum(axis=1)) - refm[j, i]
            return d
        out = []
        for bj in dom_idx:
            e, _ = err(j, bj, i, ai)
            out.append(e)
        return out

    def min_sum_e2_to_assigned(j, dom_j, assigned_items):
        """Admissible LB term: min over j-domain of summed j-vs-assigned SSE."""
        best = None
        for bj in dom_j:
            s = 0.0
            ok = True
            for i, ai in assigned_items:
                e, e2 = err(j, bj, i, ai)
                if inter_pair_tol is not None and abs(e) > inter_pair_tol:
                    ok = False
                    break
                s += e2
            if not ok:
                continue
            if best is None or s < best:
                best = s
        return best

    def min_pair_e2_between_domains(i, dom_i, j, dom_j):
        """Admissible LB edge: min SSE for any candidate pair across domains i/j."""
        best = None
        if len(dom_i) <= len(dom_j):
            for ai in dom_i:
                ev = err_vec(j, dom_j, i, ai)
                for k, bj in enumerate(dom_j):
                    e = float(ev[k])
                    if inter_pair_tol is not None and abs(e) > inter_pair_tol:
                        continue
                    e2 = e * e
                    if best is None or e2 < best:
                        best = e2
        else:
            for bj in dom_j:
                ev = err_vec(i, dom_i, j, bj)
                for k, ai in enumerate(dom_i):
                    e = float(ev[k])
                    if inter_pair_tol is not None and abs(e) > inter_pair_tol:
                        continue
                    e2 = e * e
                    if best is None or e2 < best:
                        best = e2
        return best

    sse_limit = None
    if inter_cutoff is not None and total_inter_pairs > 0:
        sse_limit = total_inter_pairs * (inter_cutoff ** 2)

    # Seed with two smallest domains and enumerate compatible seed pairs first.
    order0 = sorted(range(m), key=lambda i: len(residues[i]))
    if m >= 2:
        s1, s2 = order0[0], order0[1]
    else:
        s1, s2 = order0[0], None

    if s2 is None:
        best_idx = 0
        chosen = [None] * m
        chosen[s1] = (0.0, [residues[s1][best_idx]])
        return 0.0, 0.0, chosen

    seed_pairs = []
    for a in range(len(residues[s1])):
        id_a = ids[s1][a]
        for b in range(len(residues[s2])):
            if ids[s2][b] == id_a:
                continue
            e, e2 = err(s1, a, s2, b)
            if inter_pair_tol is not None and abs(e) > inter_pair_tol:
                continue
            if sse_limit is not None and e2 > sse_limit:
                continue
            seed_pairs.append((e2, a, b))
    if not seed_pairs:
        return None
    seed_pairs.sort(key=lambda t: t[0])

    all_domains = [list(range(len(residues[i]))) for i in range(m)]
    best = None

    def recurse(assigned, domains, used_ids, sse_assigned):
        nonlocal best
        if len(assigned) == m:
            inter_rmsd = 0.0 if total_inter_pairs == 0 else math.sqrt(sse_assigned / total_inter_pairs)
            if inter_cutoff is not None and total_inter_pairs > 0 and inter_rmsd > inter_cutoff:
                return
            chosen = [None] * m
            for i, ai in assigned.items():
                chosen[i] = (0.0, [residues[i][ai]])
            if best is None or inter_rmsd < best[1]:
                best = (0.0, inter_rmsd, chosen)
            return

        # MRV: choose unassigned cluster with smallest current domain.
        unassigned = [i for i in range(m) if i not in assigned]
        ci = min(unassigned, key=lambda i: len(domains[i]))
        dom_ci = domains[ci]
        if not dom_ci:
            return

        for ai in dom_ci:
            id_ai = ids[ci][ai]
            if id_ai in used_ids:
                continue

            add_sse = 0.0
            ok = True
            for j, aj in assigned.items():
                e, e2 = err(ci, ai, j, aj)
                if inter_pair_tol is not None and abs(e) > inter_pair_tol:
                    ok = False
                    break
                add_sse += e2
            if not ok:
                continue

            new_sse = sse_assigned + add_sse
            if sse_limit is not None and new_sse > sse_limit:
                continue

            new_assigned = dict(assigned)
            new_assigned[ci] = ai
            new_used = set(used_ids)
            new_used.add(id_ai)

            # Forward-check domains against the newly assigned residue.
            new_domains = [list(d) for d in domains]
            feasible = True
            for j in unassigned:
                if j == ci:
                    continue
                domj = new_domains[j]
                kept = []
                if domj:
                    ev = err_vec(j, domj, ci, ai)
                    for k, bj in enumerate(domj):
                        if ids[j][bj] in new_used:
                            continue
                        e = float(ev[k])
                        if inter_pair_tol is not None and abs(e) > inter_pair_tol:
                            continue
                        kept.append(bj)
                if not kept:
                    feasible = False
                    break
                new_domains[j] = kept
            if not feasible:
                continue

            # Early RMSD lower bound: unassigned-vs-assigned necessary SSE.
            if sse_limit is not None:
                lb = 0.0
                unassigned_now = [k for k in range(m) if k not in new_assigned]
                assigned_items = list(new_assigned.items())

                # 1) Admissible LB for unassigned-vs-assigned terms.
                for j in unassigned_now:
                    domj = new_domains[j]
                    best_sum = min_sum_e2_to_assigned(j, domj, assigned_items)
                    if best_sum is None:
                        feasible = False
                        break
                    lb += best_sum
                    if new_sse + lb > sse_limit:
                        feasible = False
                        break
                if not feasible:
                    continue

                # 2) Admissible LB for unassigned-vs-unassigned terms via MST on
                # per-cluster pair minima (tree cost <= any full pair-sum).
                if len(unassigned_now) >= 2:
                    w = {}
                    for a in range(len(unassigned_now) - 1):
                        u = unassigned_now[a]
                        dom_u = new_domains[u]
                        for b in range(a + 1, len(unassigned_now)):
                            v = unassigned_now[b]
                            dom_v = new_domains[v]
                            me2 = min_pair_e2_between_domains(u, dom_u, v, dom_v)
                            if me2 is None:
                                feasible = False
                                break
                            w[(u, v)] = me2
                        if not feasible:
                            break
                    if not feasible:
                        continue

                    # Prim MST
                    rem = set(unassigned_now)
                    start = unassigned_now[0]
                    rem.remove(start)
                    tree = {start}
                    mst = 0.0
                    while rem:
                        best_edge = None
                        best_node = None
                        for t in tree:
                            for v in rem:
                                key = (t, v) if t < v else (v, t)
                                ew = w.get(key)
                                if ew is None:
                                    continue
                                if best_edge is None or ew < best_edge:
                                    best_edge = ew
                                    best_node = v
                        if best_node is None:
                            feasible = False
                            break
                        mst += best_edge
                        tree.add(best_node)
                        rem.remove(best_node)

                    if not feasible:
                        continue
                    lb += mst
                    if new_sse + lb > sse_limit:
                        feasible = False
                        continue

            recurse(new_assigned, new_domains, new_used, new_sse)

    for _e2, a, b in seed_pairs:
        assigned = {s1: a, s2: b}
        used = {ids[s1][a], ids[s2][b]}
        sse0 = err(s1, a, s2, b)[1]
        domains = [list(d) for d in all_domains]

        feasible = True
        for j in range(m):
            if j in assigned:
                continue
            domj = domains[j]
            kept = []
            if domj:
                e1v = err_vec(j, domj, s1, a)
                e2v = err_vec(j, domj, s2, b)
                for k, bj in enumerate(domj):
                    if ids[j][bj] in used:
                        continue
                    e1 = float(e1v[k]); e2 = float(e2v[k])
                    if inter_pair_tol is not None and abs(e1) > inter_pair_tol:
                        continue
                    if inter_pair_tol is not None and abs(e2) > inter_pair_tol:
                        continue
                    kept.append(bj)
            if not kept:
                feasible = False
                break
            domains[j] = kept
        if not feasible:
            continue

        recurse(assigned, domains, used, sse0)

    return best


def find_assignment(
    candidates_by_cluster,
    cluster_offsets=None,
    inter_pair_refs=None,
    total_inter_pairs=0,
    inter_cutoff=None,
    inter_pair_tol=None,
    global_pair_refs=None,
    total_global_pairs=0,
    global_cutoff=None,
    best_assignment=False,
    seed_cluster_index=None,
    seed_best_only=False,
    cluster_cutoffs=None,
):
    m = len(candidates_by_cluster)
    inter_sse_limit = None
    if inter_cutoff is not None and total_inter_pairs > 0:
        inter_sse_limit = total_inter_pairs * (inter_cutoff ** 2)
    global_sse_limit = None
    if global_cutoff is not None and total_global_pairs > 0:
        global_sse_limit = total_global_pairs * (global_cutoff ** 2)

    # Seed-aware prefilter before branching:
    # keep only candidates in non-seed clusters that are pair-compatible with at
    # least one seed candidate. This is an exact necessary condition and cuts
    # domains early for broad singleton searches.
    if (
        seed_cluster_index is not None
        and 0 <= seed_cluster_index < m
        and cluster_offsets is not None
        and inter_pair_refs
        and (inter_pair_tol is not None or inter_sse_limit is not None)
    ):
        if seed_best_only and candidates_by_cluster[seed_cluster_index]:
            candidates_by_cluster[seed_cluster_index] = [candidates_by_cluster[seed_cluster_index][0]]
        seed_cands = candidates_by_cluster[seed_cluster_index]
        if not seed_cands:
            return None

        def seed_pair_sse(ci, sel_i, cj, sel_j):
            ids_i = {r['res_id'] for r in sel_i}
            for r in sel_j:
                if r['res_id'] in ids_i:
                    return None
            off_i = cluster_offsets[ci]
            off_j = cluster_offsets[cj]
            sse = 0.0
            for ai, ra in enumerate(sel_i):
                fa = off_i + ai
                for bj, rb in enumerate(sel_j):
                    fb = off_j + bj
                    key = (fa, fb) if fa < fb else (fb, fa)
                    ref = inter_pair_refs.get(key)
                    if ref is None:
                        continue
                    e = dist(ra['xyz'], rb['xyz']) - ref
                    if inter_pair_tol is not None and abs(e) > inter_pair_tol:
                        return None
                    sse += e * e
            return sse

        reduced = []
        for ci in range(m):
            if ci == seed_cluster_index:
                reduced.append(candidates_by_cluster[ci])
                continue
            kept = []
            for pd_i, sel_i in candidates_by_cluster[ci]:
                ok = False
                for _pd_s, sel_s in seed_cands:
                    sse = seed_pair_sse(ci, sel_i, seed_cluster_index, sel_s)
                    if sse is None:
                        continue
                    # Keep seed prefilter aligned with main semantics:
                    # local per-pair gate is controlled by inter_pair_tol, while
                    # global consistency is controlled by inter_cutoff. Do not
                    # apply additional per-cluster profile RMSD gating here.
                    if inter_sse_limit is not None and sse > inter_sse_limit:
                        continue
                    ok = True
                    break
                if ok:
                    kept.append((pd_i, sel_i))
            if not kept:
                return None
            reduced.append(kept)
        candidates_by_cluster = reduced

    # Fast path: singleton CSP search (dominant use case for full-pocket 19-template).
    if global_sse_limit is None and all(c and all(len(sel) == 1 for _pd, sel in c) for c in candidates_by_cluster):
        return find_assignment_singletons(
            candidates_by_cluster,
            inter_pair_refs=inter_pair_refs,
            total_inter_pairs=total_inter_pairs,
            inter_cutoff=inter_cutoff,
            inter_pair_tol=inter_pair_tol,
        )

    def disjoint_sel(sel_a, sel_b):
        ids_a = {r['res_id'] for r in sel_a}
        for r in sel_b:
            if r['res_id'] in ids_a:
                return False
        return True

    def cross_sse_with_refs(sel_a, off_a, sel_b, off_b, refs, pair_tol=None):
        if not disjoint_sel(sel_a, sel_b):
            return None
        sse = 0.0
        for ai, ra in enumerate(sel_a):
            fa = off_a + ai
            for bi, rb in enumerate(sel_b):
                fb = off_b + bi
                key = (fa, fb) if fa < fb else (fb, fa)
                ref = refs.get(key) if refs else None
                if ref is None:
                    continue
                e = dist(ra['xyz'], rb['xyz']) - ref
                if pair_tol is not None and abs(e) > pair_tol:
                    return None
                sse += e * e
        return sse

    def cross_sse(sel_a, off_a, sel_b, off_b):
        return cross_sse_with_refs(sel_a, off_a, sel_b, off_b, inter_pair_refs, inter_pair_tol)

    def intra_sse_with_refs(sel, off, refs):
        sse = 0.0
        n = len(sel)
        for i in range(n - 1):
            fi = off + i
            for j in range(i + 1, n):
                fj = off + j
                key = (fi, fj) if fi < fj else (fj, fi)
                ref = refs.get(key) if refs else None
                if ref is None:
                    continue
                e = dist(sel[i]['xyz'], sel[j]['xyz']) - ref
                sse += e * e
        return sse

    compat = {}

    # Early exact pruning:
    # 1) seed-star domain filtering (cheap, before all-pairs matrix build)
    # 2) pair-compatibility + AC-3
    if (inter_pair_tol is not None or inter_sse_limit is not None) and inter_pair_refs and cluster_offsets is not None:
        # Multi-seed star prefilter: keep candidates that are pair-feasible to at least
        # one candidate in several smallest-domain anchors. This is necessary (exact)
        # and cuts domains before O(n_i*n_j) all-pairs compatibility construction.
        dom = [list(range(len(c))) for c in candidates_by_cluster]
        n_seed = min(3, m)
        seeds = sorted(range(m), key=lambda i: len(dom[i]))[:n_seed]
        changed = True
        while changed:
            changed = False
            for seed in seeds:
                seed_keep = set(dom[seed])
                for j in range(m):
                    if j == seed:
                        continue
                    keep_seed_j = set()
                    keep_j = set()
                    off_s = cluster_offsets[seed]
                    off_j = cluster_offsets[j]
                    for a in dom[seed]:
                        _pds, sel_s = candidates_by_cluster[seed][a]
                        for b in dom[j]:
                            _pdj, sel_j = candidates_by_cluster[j][b]
                            sse_sj = cross_sse(sel_s, off_s, sel_j, off_j)
                            if sse_sj is None:
                                continue
                            if inter_sse_limit is not None and sse_sj > inter_sse_limit:
                                continue
                            keep_seed_j.add(a)
                            keep_j.add(b)
                    if not keep_j or not keep_seed_j:
                        return None
                    if len(keep_j) != len(dom[j]):
                        dom[j] = sorted(keep_j)
                        changed = True
                    seed_keep &= keep_seed_j
                if not seed_keep:
                    return None
                if len(seed_keep) != len(dom[seed]):
                    dom[seed] = sorted(seed_keep)
                    changed = True

        candidates_by_cluster = [
            [candidates_by_cluster[i][k] for k in dom[i]] for i in range(m)
        ]

        sizes = [len(c) for c in candidates_by_cluster]
        compat = {}
        pair_min_sse = {}
        inf = float('inf')

        def comp_get(i, a, j, b):
            if i < j:
                _ni, nj, mat, _sse, _rmin, _cmin = compat[(i, j)]
                return mat[a * nj + b] == 1
            _nj, ni, mat, _sse, _rmin, _cmin = compat[(j, i)]
            return mat[b * ni + a] == 1

        def comp_sse(i, a, j, b):
            if i < j:
                _ni, nj, mat, ssev, _rmin, _cmin = compat[(i, j)]
                idx = a * nj + b
                if mat[idx] != 1:
                    return None
                return ssev[idx]
            _nj, ni, mat, ssev, _rmin, _cmin = compat[(j, i)]
            idx = b * ni + a
            if mat[idx] != 1:
                return None
            return ssev[idx]

        def get_row_min(i, a, j):
            if i < j:
                _ni, _nj, _mat, _sse, rmin, _cmin = compat[(i, j)]
                return rmin[a]
            _nj, _ni, _mat, _sse, _rmin, cmin = compat[(j, i)]
            return cmin[a]

        for i in range(m - 1):
            off_i = cluster_offsets[i]
            ni = sizes[i]
            for j in range(i + 1, m):
                off_j = cluster_offsets[j]
                nj = sizes[j]
                mat = bytearray(ni * nj)
                ssev = [inf] * (ni * nj)
                row_min = [inf] * ni
                col_min = [inf] * nj
                pmin = inf
                for a, (_pdi, sel_i) in enumerate(candidates_by_cluster[i]):
                    base = a * nj
                    for b, (_pdj, sel_j) in enumerate(candidates_by_cluster[j]):
                        sse_ab = cross_sse(sel_i, off_i, sel_j, off_j)
                        if sse_ab is not None:
                            if inter_sse_limit is not None and sse_ab > inter_sse_limit:
                                continue
                            mat[base + b] = 1
                            ssev[base + b] = sse_ab
                            if sse_ab < row_min[a]:
                                row_min[a] = sse_ab
                            if sse_ab < col_min[b]:
                                col_min[b] = sse_ab
                            if sse_ab < pmin:
                                pmin = sse_ab
                compat[(i, j)] = (ni, nj, mat, ssev, row_min, col_min)
                pair_min_sse[(i, j)] = pmin

        domains = [set(range(len(candidates_by_cluster[i]))) for i in range(m)]
        neighbors = [[j for j in range(m) if j != i] for i in range(m)]
        q = deque((i, j) for i in range(m) for j in range(m) if i != j)

        while q:
            i, j = q.popleft()
            dom_i = domains[i]
            dom_j = domains[j]
            if not dom_i or not dom_j:
                return None
            remove = []
            for a in dom_i:
                ok = False
                for b in dom_j:
                    if comp_get(i, a, j, b):
                        ok = True
                        break
                if not ok:
                    remove.append(a)
            if remove:
                for a in remove:
                    domains[i].remove(a)
                if not domains[i]:
                    return None
                for k in neighbors[i]:
                    if k != j:
                        q.append((k, i))

        # RMSD lower-bound pruning before recursion:
        # 1) global future-future lower bound across all cluster pairs
        # 2) per-candidate necessary lower bound against all other clusters
        if inter_sse_limit is not None:
            lb_global = 0.0
            for i in range(m - 1):
                for j in range(i + 1, m):
                    p = pair_min_sse.get((i, j), inf)
                    if p == inf:
                        return None
                    lb_global += p
            if lb_global > inter_sse_limit:
                return None

            changed = True
            while changed:
                changed = False
                for i in range(m):
                    dom_i = domains[i]
                    if not dom_i:
                        return None
                    remove = []
                    for a in dom_i:
                        lb_a = 0.0
                        feasible_a = True
                        for j in range(m):
                            if j == i:
                                continue
                            best = get_row_min(i, a, j)
                            if best == inf:
                                feasible_a = False
                                break
                            lb_a += best
                            if lb_a > inter_sse_limit:
                                feasible_a = False
                                break
                        if not feasible_a:
                            remove.append(a)
                    if remove:
                        for a in remove:
                            domains[i].remove(a)
                        if not domains[i]:
                            return None
                        changed = True
                        for k in neighbors[i]:
                            q.append((k, i))

                while q:
                    i, j = q.popleft()
                    dom_i = domains[i]
                    dom_j = domains[j]
                    if not dom_i or not dom_j:
                        return None
                    remove = []
                    for a in dom_i:
                        ok = False
                        for b in dom_j:
                            if comp_get(i, a, j, b):
                                ok = True
                                break
                        if not ok:
                            remove.append(a)
                    if remove:
                        for a in remove:
                            domains[i].remove(a)
                        if not domains[i]:
                            return None
                        changed = True
                        for k in neighbors[i]:
                            if k != j:
                                q.append((k, i))

        reduced = []
        for i in range(m):
            keep = sorted(domains[i])
            reduced.append([candidates_by_cluster[i][k] for k in keep])
        candidates_by_cluster = reduced
        reduced_min_pair_sse = pair_min_sse
    else:
        reduced_min_pair_sse = {}

    # Global geometry lower-bound components for pruning.
    min_intra_global_sse = [0.0] * m
    min_cross_global_sse = {}
    global_intra_sse = [[] for _ in range(m)]
    global_cross_sse = {}
    if global_sse_limit is not None and global_pair_refs and cluster_offsets is not None:
        for i in range(m):
            off_i = cluster_offsets[i]
            vals = [intra_sse_with_refs(sel, off_i, global_pair_refs) for _pd, sel in candidates_by_cluster[i]]
            global_intra_sse[i] = vals
            min_intra_global_sse[i] = min(vals) if vals else 0.0
        for i in range(m - 1):
            off_i = cluster_offsets[i]
            ni = len(candidates_by_cluster[i])
            for j in range(i + 1, m):
                off_j = cluster_offsets[j]
                nj = len(candidates_by_cluster[j])
                best_ij = None
                ssev = [None] * (ni * nj)
                for ai, (_pdi, sel_i) in enumerate(candidates_by_cluster[i]):
                    base = ai * nj
                    for aj, (_pdj, sel_j) in enumerate(candidates_by_cluster[j]):
                        sse_ij = cross_sse_with_refs(sel_i, off_i, sel_j, off_j, global_pair_refs, pair_tol=None)
                        ssev[base + aj] = sse_ij
                        if sse_ij is None:
                            continue
                        if best_ij is None or sse_ij < best_ij:
                            best_ij = sse_ij
                global_cross_sse[(i, j)] = (nj, ssev)
                # If no disjoint pair exists at LB stage, keep zero LB (admissible).
                min_cross_global_sse[(i, j)] = best_ij if best_ij is not None else 0.0

    if seed_cluster_index is not None and 0 <= seed_cluster_index < m:
        if seed_best_only and candidates_by_cluster[seed_cluster_index]:
            candidates_by_cluster[seed_cluster_index] = [candidates_by_cluster[seed_cluster_index][0]]
        rest = [i for i in range(m) if i != seed_cluster_index]
        rest.sort(key=lambda i: len(candidates_by_cluster[i]))
        order = [seed_cluster_index] + rest
    else:
        order = sorted(range(m), key=lambda i: len(candidates_by_cluster[i]))
    chosen = [None] * m
    chosen_idx = [None] * m
    used_ids = set()
    best = None
    cand_idsets = [[{r['res_id'] for r in sel} for _pd, sel in cands] for cands in candidates_by_cluster]
    has_compat = bool(compat)

    # Remaining-pairs lower bound (future-future) by recursion depth.
    remaining_pair_lb = [0.0] * (m + 1)
    if inter_sse_limit is not None and reduced_min_pair_sse:
        for k in range(m - 1, -1, -1):
            rem = order[k:]
            s = 0.0
            for a in range(len(rem) - 1):
                i = rem[a]
                for b in range(a + 1, len(rem)):
                    j = rem[b]
                    key = (i, j) if i < j else (j, i)
                    s += reduced_min_pair_sse.get(key, 0.0)
            remaining_pair_lb[k] = s

    # Global LB caches by recursion depth.
    remaining_global_intra_lb = [0.0] * (m + 1)
    remaining_global_pair_lb = [0.0] * (m + 1)
    if global_sse_limit is not None:
        for k in range(m - 1, -1, -1):
            rem = order[k:]
            remaining_global_intra_lb[k] = sum(min_intra_global_sse[i] for i in rem)
            s = 0.0
            for a in range(len(rem) - 1):
                i = rem[a]
                for b in range(a + 1, len(rem)):
                    j = rem[b]
                    key = (i, j) if i < j else (j, i)
                    s += min_cross_global_sse.get(key, 0.0)
            remaining_global_pair_lb[k] = s

    def rec(k, score, inter_sse, global_sse):
        nonlocal best
        if k == m:
            inter_rmsd = 0.0 if total_inter_pairs == 0 else math.sqrt(inter_sse / total_inter_pairs)
            if inter_cutoff is not None and total_inter_pairs > 0 and inter_rmsd > inter_cutoff:
                return False
            global_rmsd = 0.0 if total_global_pairs == 0 else math.sqrt(global_sse / total_global_pairs)
            if global_cutoff is not None and total_global_pairs > 0 and global_rmsd > global_cutoff:
                return False
            if best is None or score < best[0]:
                best = (score, inter_rmsd, [x for x in chosen])
            return True

        ci = order[k]
        for ai, (pd, sel) in enumerate(candidates_by_cluster[ci]):
            ids = cand_idsets[ci][ai]
            if used_ids & ids:
                continue
            new_score = score + pd
            if best_assignment and best is not None and new_score >= best[0]:
                continue

            add_inter_sse = 0.0
            add_global_sse = 0.0
            feasible = True
            off_i = cluster_offsets[ci] if cluster_offsets is not None else 0
            if (inter_sse_limit is not None or inter_pair_tol is not None) and inter_pair_refs and cluster_offsets is not None:
                for cj, item in enumerate(chosen):
                    if item is None:
                        continue
                    if has_compat:
                        aj = chosen_idx[cj]
                        sse_ij = comp_sse(ci, ai, cj, aj)
                    else:
                        _pdj, sel_j = item
                        off_j = cluster_offsets[cj]
                        sse_ij = cross_sse(sel, off_i, sel_j, off_j)
                    if sse_ij is None:
                        feasible = False
                        break
                    add_inter_sse += sse_ij
                if not feasible:
                    continue
                if inter_sse_limit is not None and inter_sse + add_inter_sse > inter_sse_limit:
                    continue
                if inter_sse_limit is not None:
                    lb_ff = remaining_pair_lb[k + 1] if (k + 1) <= m else 0.0
                    if inter_sse + add_inter_sse + lb_ff > inter_sse_limit:
                        continue

            if global_sse_limit is not None and global_pair_refs and cluster_offsets is not None:
                add_global_sse += global_intra_sse[ci][ai]
                for cj, item in enumerate(chosen):
                    if item is None:
                        continue
                    aj = chosen_idx[cj]
                    if ci < cj:
                        nj, ssev = global_cross_sse[(ci, cj)]
                        sse_ij = ssev[ai * nj + aj]
                    else:
                        nj, ssev = global_cross_sse[(cj, ci)]
                        sse_ij = ssev[aj * nj + ai]
                    if sse_ij is None:
                        feasible = False
                        break
                    add_global_sse += sse_ij
                if not feasible:
                    continue
                if global_sse + add_global_sse > global_sse_limit:
                    continue
                assigned_ids = order[:k + 1]
                future_ids = order[k + 1:]
                lb_future = remaining_global_intra_lb[k + 1] + remaining_global_pair_lb[k + 1]
                lb_assigned_future = 0.0
                for ia in assigned_ids:
                    for iu in future_ids:
                        key = (ia, iu) if ia < iu else (iu, ia)
                        lb_assigned_future += min_cross_global_sse.get(key, 0.0)
                if global_sse + add_global_sse + lb_future + lb_assigned_future > global_sse_limit:
                    continue

            chosen[ci] = (pd, sel)
            chosen_idx[ci] = ai
            used_ids.update(ids)

            # Forward-checking with exact lower-bound accumulation:
            # each future cluster must have at least one candidate compatible with all
            # current choices; for RMSD, accumulate the minimum possible additional SSE
            # from future-vs-chosen pairs.
            rem_after = m - (k + 1)
            if rem_after >= 2 and (inter_pair_tol is not None or inter_sse_limit is not None) and inter_pair_refs and cluster_offsets is not None:
                lb_add = 0.0
                feasible = True
                for ku in range(k + 1, m):
                    cu = order[ku]
                    off_u = cluster_offsets[cu]
                    best_u = None
                    for au, (_pdu, sel_u) in enumerate(candidates_by_cluster[cu]):
                        ids_u = cand_idsets[cu][au]
                        if used_ids & ids_u:
                            continue
                        sse_u = 0.0
                        ok_u = True
                        for cj, itemj in enumerate(chosen):
                            if itemj is None:
                                continue
                            if has_compat:
                                aj = chosen_idx[cj]
                                sse_uj = comp_sse(cu, au, cj, aj)
                            else:
                                _pdj, sel_j = itemj
                                off_j = cluster_offsets[cj]
                                sse_uj = cross_sse(sel_u, off_u, sel_j, off_j)
                            if sse_uj is None:
                                ok_u = False
                                break
                            sse_u += sse_uj
                        if ok_u:
                            if best_u is None or sse_u < best_u:
                                best_u = sse_u
                    if best_u is None:
                        feasible = False
                        break
                    lb_add += best_u
                    if inter_sse_limit is not None and inter_sse + add_inter_sse + lb_add > inter_sse_limit:
                        feasible = False
                        break
                if not feasible:
                    used_ids.difference_update(ids)
                    chosen[ci] = None
                    continue

            found = rec(k + 1, new_score, inter_sse + add_inter_sse, global_sse + add_global_sse)
            used_ids.difference_update(ids)
            chosen[ci] = None
            chosen_idx[ci] = None
            if found and not best_assignment:
                return True
        return False

    rec(0, 0.0, 0.0, 0.0)
    if best is None:
        return None
    return best[0], best[1], best[2]


def load_pulldown(path):
    if not path:
        return None
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            rows.append(dict(row))
    return rows


def pulldown_row_target_id(row, acc_to_entries, entry_to_acc):
    """Return canonical target ID for a pulldown row (uid-first policy)."""
    candidates = []
    for col in ('uid', 'uniprot_id', 'mapped_token', 'entry'):
        if col not in row:
            continue
        vals = split_multi(row.get(col), seps=';|,')
        for v in vals:
            v = (v or '').strip().upper()
            if v:
                candidates.append(v)
    if not candidates:
        return ''
    canonical = [canonical_uniprot_id(v, acc_to_entries, entry_to_acc) for v in candidates]
    for c in canonical:
        if c in acc_to_entries:
            return c
    return canonical[0]


def load_pulldown_seed_candidates(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, newline='') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            uid = (row.get('uniprot_id') or '').strip().upper()
            if not uid:
                continue
            toks = split_multi(row.get('seed_candidates'), seps='|;,')
            out[uid] = {(t or '').strip().upper() for t in toks if (t or '').strip()}
    return out


def build_pulldown_token_index(pulldown_rows, acc_to_entries, entry_to_acc, pulldown_seed_candidates):
    token_to_targets = {}
    if not pulldown_rows:
        return token_to_targets
    for row in pulldown_rows:
        tid = pulldown_row_target_id(row, acc_to_entries, entry_to_acc)
        if not tid:
            continue
        seeds = {tid}
        for c in ('uid', 'uniprot_id', 'entry', 'mapped_token'):
            if c in row:
                seeds |= {(x or '').strip().upper() for x in split_multi(row.get(c), seps=';|,') if (x or '').strip()}
        seeds |= pulldown_seed_candidates.get(tid, set())
        expanded = expand_alias_tokens(seeds, acc_to_entries, entry_to_acc, rounds=2)
        for tok in expanded:
            if tok not in token_to_targets:
                token_to_targets[tok] = set()
            token_to_targets[tok].add(tid)
    return token_to_targets


def match_pulldown_targets(uid, entry, token_to_targets, acc_to_entries, entry_to_acc):
    seeds = set()
    if uid:
        seeds.add(uid.strip().upper())
    if entry:
        seeds.add(entry.strip().upper())
    expanded = expand_alias_tokens(seeds, acc_to_entries, entry_to_acc, rounds=2)
    out = set()
    for tok in expanded:
        out |= token_to_targets.get(tok, set())
    return out


def evaluate_entry(task):
    (
        entry,
        uid,
        ecoli_batch,
        clusters,
        max_candidates_per_cluster,
        cluster_offsets,
        inter_pair_refs,
        inter_pairs_count,
        inter_cluster_rmsd_cutoff,
        inter_pair_tol,
        global_pair_refs,
        total_global_pairs,
        global_pairdist_cutoff,
        best_assignment,
        seed_cluster_index,
        seed_best_only,
    ) = task
    path = os.path.join(ecoli_batch, entry, 'residues.tsv')
    if not os.path.exists(path):
        return None
    residues = load_residue_rows(path)

    m = len(clusters)
    if m == 0:
        return None

    inter_sse_limit = None
    if inter_cluster_rmsd_cutoff is not None and inter_pairs_count > 0:
        inter_sse_limit = inter_pairs_count * (inter_cluster_rmsd_cutoff ** 2)

    # Two-stage default:
    # Stage 1: enumerate cluster-local candidates only (no inter-cluster gating).
    # Stage 2: apply inter-pair / inter-cluster filters in find_assignment.
    cand_lists = [None] * m
    local_max_candidates = 1 if m == 1 else max_candidates_per_cluster
    for ci in range(m):
        # For single-cluster runs, global pairdist cutoff is the effective
        # candidate-generation cutoff when provided.
        eff_cutoff = clusters[ci]['cutoff']
        if (
            m == 1
            and global_pairdist_cutoff is not None
            and len(clusters[ci]['template_points']) > 1
        ):
            eff_cutoff = global_pairdist_cutoff
        cand = enumerate_cluster_matches(
            residues,
            clusters[ci]['allowed'],
            clusters[ci]['template_points'],
            eff_cutoff,
            local_max_candidates,
            unordered=clusters[ci].get('unordered', False),
        )
        if not cand:
            return None
        cand_lists[ci] = cand

    assign = find_assignment(
        cand_lists,
        cluster_offsets=cluster_offsets,
        inter_pair_refs=inter_pair_refs,
        total_inter_pairs=inter_pairs_count,
        inter_cutoff=inter_cluster_rmsd_cutoff,
        inter_pair_tol=inter_pair_tol,
        global_pair_refs=global_pair_refs,
        total_global_pairs=total_global_pairs,
        global_cutoff=global_pairdist_cutoff,
        best_assignment=best_assignment,
        seed_cluster_index=seed_cluster_index,
        seed_best_only=seed_best_only,
        cluster_cutoffs=[c['cutoff'] for c in clusters],
    )
    if assign is None:
        return None
    sum_cluster_pd, inter_pd, chosen = assign
    template_pts = []
    selected_pts = []
    for ci, c in enumerate(clusters):
        template_pts.extend(c['template_points'])
        _pd, sel = chosen[ci]
        selected_pts.extend([r['xyz'] for r in sel])
    global_pd = pairdist_rmsd_points(template_pts, selected_pts)
    if global_pairdist_cutoff is not None and global_pd > global_pairdist_cutoff:
        return None
    row = {
        'uniprot_id': uid,
        'entry': entry,
        'sum_cluster_pairdist_rmsd': f"{sum_cluster_pd:.4f}",
        'global_pairdist_rmsd': f"{global_pd:.4f}",
        'inter_pairdist_rmsd': f"{inter_pd:.4f}",
    }
    for i, c in enumerate(clusters, start=1):
        pd, sel = chosen[i - 1]
        row[f'cluster{i}_pattern'] = c['pattern_raw']
        row[f'cluster{i}'] = ','.join(f"{r['aa']}{r['resseq']}" for r in sel)
        row[f'cluster{i}_pairdist_rmsd'] = f"{pd:.4f}"
    return row


def solve_one_assignment(
    residues,
    clusters,
    max_candidates_per_cluster,
    cluster_offsets,
    inter_pair_refs,
    inter_pairs_count,
    inter_cluster_rmsd_cutoff,
    inter_pair_tol,
    global_pair_refs,
    total_global_pairs,
    global_pairdist_cutoff,
    best_assignment,
    seed_cluster_index,
    seed_best_only,
):
    m = len(clusters)
    if m == 0:
        return None
    cand_lists = [None] * m
    local_max_candidates = 1 if m == 1 else max_candidates_per_cluster
    for ci in range(m):
        # Keep single-cluster behavior consistent with evaluate_entry:
        # when provided, global cutoff is the early candidate prune.
        eff_cutoff = clusters[ci]['cutoff']
        if (
            m == 1
            and global_pairdist_cutoff is not None
            and len(clusters[ci]['template_points']) > 1
        ):
            eff_cutoff = global_pairdist_cutoff
        cand = enumerate_cluster_matches(
            residues,
            clusters[ci]['allowed'],
            clusters[ci]['template_points'],
            eff_cutoff,
            local_max_candidates,
            unordered=clusters[ci].get('unordered', False),
        )
        if not cand:
            return None
        cand_lists[ci] = cand
    assign = find_assignment(
        cand_lists,
        cluster_offsets=cluster_offsets,
        inter_pair_refs=inter_pair_refs,
        total_inter_pairs=inter_pairs_count,
        inter_cutoff=inter_cluster_rmsd_cutoff,
        inter_pair_tol=inter_pair_tol,
        global_pair_refs=global_pair_refs,
        total_global_pairs=total_global_pairs,
        global_cutoff=global_pairdist_cutoff,
        best_assignment=best_assignment,
        seed_cluster_index=seed_cluster_index,
        seed_best_only=seed_best_only,
        cluster_cutoffs=[c['cutoff'] for c in clusters],
    )
    return assign


def evaluate_entry_multiplicity(task):
    (
        entry,
        uid,
        ecoli_batch,
        clusters,
        max_candidates_per_cluster,
        cluster_offsets,
        inter_pair_refs,
        inter_pairs_count,
        inter_cluster_rmsd_cutoff,
        inter_pair_tol,
        global_pair_refs,
        total_global_pairs,
        global_pairdist_cutoff,
        best_assignment,
        seed_cluster_index,
        seed_best_only,
        multiplicity_max_per_protein,
    ) = task
    path = os.path.join(ecoli_batch, entry, 'residues.tsv')
    if not os.path.exists(path):
        return None
    residues = load_residue_rows(path)
    available = list(residues)
    multiplicity = 0
    first_row = None
    while multiplicity < multiplicity_max_per_protein:
        assign = solve_one_assignment(
            available,
            clusters,
            max_candidates_per_cluster,
            cluster_offsets,
            inter_pair_refs,
            inter_pairs_count,
            inter_cluster_rmsd_cutoff,
            inter_pair_tol,
            global_pair_refs,
            total_global_pairs,
            global_pairdist_cutoff,
            best_assignment,
            seed_cluster_index,
            seed_best_only,
        )
        if assign is None:
            break
        sum_cluster_pd, inter_pd, chosen = assign
        used = set()
        for _pd, sel in chosen:
            for r in sel:
                used.add(r['res_id'])
        if not used:
            break
        multiplicity += 1
        if first_row is None:
            template_pts = []
            selected_pts = []
            for ci, c in enumerate(clusters):
                template_pts.extend(c['template_points'])
                _pd, sel = chosen[ci]
                selected_pts.extend([r['xyz'] for r in sel])
            global_pd = pairdist_rmsd_points(template_pts, selected_pts)
            first_row = {
                'uniprot_id': uid,
                'entry': entry,
                'sum_cluster_pairdist_rmsd': f"{sum_cluster_pd:.4f}",
                'global_pairdist_rmsd': f"{global_pd:.4f}",
                'inter_pairdist_rmsd': f"{inter_pd:.4f}",
            }
            for i, c in enumerate(clusters, start=1):
                pd, sel = chosen[i - 1]
                first_row[f'cluster{i}_pattern'] = c['pattern_raw']
                first_row[f'cluster{i}'] = ','.join(f"{r['aa']}{r['resseq']}" for r in sel)
                first_row[f'cluster{i}_pairdist_rmsd'] = f"{pd:.4f}"
        available = [r for r in available if r['res_id'] not in used]
        if not available:
            break
    if multiplicity == 0:
        return None
    out = {'uniprot_id': uid, 'entry': entry, 'multiplicity_disjoint': multiplicity}
    if first_row is not None:
        out.update(first_row)
    return out


def build_clusters(args):
    clusters = []
    use_general = args.clusters_template is not None
    if use_general:
        tparts = split_clusters(args.clusters_template)
        if not tparts:
            raise SystemExit('--clusters-template is empty')
        if len(tparts) > 19:
            raise SystemExit('max supported clusters is 19')

        if args.clusters_pattern:
            pparts = split_clusters(args.clusters_pattern)
            if len(pparts) != len(tparts):
                raise SystemExit('cluster count mismatch: --clusters-template vs --clusters-pattern')
        else:
            mat = load_blosum_matrix(args.blosum_matrix)
            flat_tmpl = []
            for ts in tparts:
                flat_tmpl.extend(parse_residue_list(ts))
            pep_map = None
            pep_seq = None
            mj = None
            if args.degen_peptide_seq or args.degen_peptide_map:
                if not args.degen_peptide_seq or not args.degen_peptide_map:
                    raise SystemExit('Both --degen-peptide-seq and --degen-peptide-map are required for peptide-aware degeneration.')
                pep_seq = list(args.degen_peptide_seq.strip().upper())
                pep_map = parse_peptide_index_map(args.degen_peptide_map, len(flat_tmpl))
                mj = load_mj_matrix()
            pparts = []
            pos = 0
            for ts in tparts:
                tmpl = parse_residue_list(ts)
                toks = []
                for aa, _ in tmpl:
                    if args.degen_top_k is not None:
                        dset_set = degen_set_topk_from_blosum(
                            mat, aa, args.degen_top_k, args.degen_top_min_score
                        )
                    else:
                        dset_set = degen_set_from_blosum(mat, aa, args.degen_threshold)
                    # Optional peptide-complement-aware filter:
                    # keep only BLOSUM-allowed substitutions whose MJ contact with
                    # mapped peptide residue is favorable enough.
                    if pep_map is not None:
                        pi = pep_map[pos]
                        if pi is not None and 0 <= pi < len(pep_seq):
                            pa = pep_seq[pi]
                            if args.degen_mj_relative:
                                # Default behavior: allow substitutions that are
                                # equal-or-better than the template residue MJ
                                # against the mapped peptide residue.
                                base_mj = mj_score(mj, pa, aa)
                                filtered = {
                                    x for x in dset_set
                                    if mj_score(mj, pa, x) <= (base_mj + args.degen_mj_relative_max_worse)
                                }
                            else:
                                filtered = {
                                    x for x in dset_set
                                    if mj_score(mj, pa, x) <= args.degen_mj_max
                                }
                            if filtered:
                                dset_set = filtered
                    if not dset_set:
                        dset_set = {aa}
                    dset = ''.join(sorted(dset_set))
                    toks.append(f'[{dset}]')
                    pos += 1
                pparts.append(' '.join(toks))

        if args.clusters_cutoffs:
            cvals = parse_float_list(args.clusters_cutoffs)
            if len(cvals) == 1:
                cvals = cvals * len(tparts)
            elif len(cvals) != len(tparts):
                raise SystemExit('cluster count mismatch: --clusters-cutoffs')
        else:
            # Default: disable per-cluster internal pairdist cutoff unless
            # explicitly provided via --clusters-cutoffs.
            cvals = [float('inf')] * len(tparts)
        unordered_idx = parse_index_spec_1based(args.clusters_unordered, len(tparts))

        for i, (ts, ps, cf) in enumerate(zip(tparts, pparts, cvals), start=1):
            tmpl = parse_residue_list(ts)
            patt = parse_pattern_tokens(ps)
            if len(tmpl) != len(patt) or len(tmpl) < 1:
                raise SystemExit(f'cluster{i}: template/pattern length mismatch or zero length')
            if len(tmpl) > 19:
                raise SystemExit(f'cluster{i}: max supported cluster length is 19')
            clusters.append({
                'name': f'cluster{i}',
                'template_res': tmpl,
                'pattern_raw': ps,
                'allowed': patt,
                'cutoff': float(cf),
                'unordered': (i - 1) in unordered_idx,
            })
        return clusters

    # Legacy two-cluster fallback
    c1_t = parse_residue_list(args.cluster1_template)
    c2_t = parse_residue_list(args.cluster2_template)
    c1_p = parse_pattern_tokens(args.cluster1_pattern)
    c2_p = parse_pattern_tokens(args.cluster2_pattern)
    if len(c1_t) != len(c1_p) or len(c1_t) < 1:
        raise SystemExit('cluster1 template/pattern must have same length >= 1')
    if len(c2_t) != len(c2_p) or len(c2_t) < 1:
        raise SystemExit('cluster2 template/pattern must have same length >= 1')
    clusters.append({'name': 'cluster1', 'template_res': c1_t, 'pattern_raw': args.cluster1_pattern, 'allowed': c1_p, 'cutoff': float(args.cluster1_cutoff), 'unordered': False})
    clusters.append({'name': 'cluster2', 'template_res': c2_t, 'pattern_raw': args.cluster2_pattern, 'allowed': c2_p, 'cutoff': float(args.cluster2_cutoff), 'unordered': False})
    return clusters


def main():
    ap = argparse.ArgumentParser(description='Hard pairwise-RMSD pocket search (simple mode).')
    ap.add_argument('--ecoli-batch', default='src/ecoli_batch')
    ap.add_argument('--template-entry', default='DPO3B_ECOLI')

    # Legacy 2-cluster interface
    ap.add_argument('--cluster1-template', default='H175,D173,R176,Y323')
    ap.add_argument('--cluster2-template', default='T172,G174,L177')
    ap.add_argument('--cluster1-pattern', default='[HNQ] [DE] [RK] Y')
    ap.add_argument('--cluster2-pattern', default='[GAVLITS] [GAVLITS] [GAVLITS]')
    ap.add_argument('--cluster1-cutoff', type=float, default=3.0)
    ap.add_argument('--cluster2-cutoff', type=float, default=3.0)

    # General interface
    ap.add_argument('--clusters-template', default=DEFAULT_FULL19_TEMPLATE,
                    help='Cluster templates separated by "|".')
    ap.add_argument('--clusters-pattern', default=None,
                    help='Cluster patterns separated by "|". If omitted, auto-degenerates from BLOSUM.')
    ap.add_argument('--clusters-cutoffs', default=None,
                    help='Comma-separated cutoffs per cluster or one broadcast value.')
    ap.add_argument('--clusters-unordered', default='',
                    help='1-based cluster indices with free internal assignment (e.g., "1" or "1,3-4" or "all").')
    ap.add_argument('--blosum-matrix', default='src/pocket_pipeline/blosum62.txt')
    ap.add_argument('--degen-threshold', type=int, default=0,
                    help='BLOSUM threshold for auto-degeneration when patterns omitted.')
    ap.add_argument('--degen-top-k', type=int, default=None,
                    help='Use top-K BLOSUM substitutions per position (ties included) when patterns omitted.')
    ap.add_argument('--degen-top-min-score', type=int, default=0,
                    help='Minimum BLOSUM score allowed for --degen-top-k substitutions.')
    ap.add_argument('--degen-peptide-seq', default='',
                    help='Optional peptide sequence for complement-aware degeneration (used only when --clusters-pattern is omitted).')
    ap.add_argument('--degen-peptide-map', default='',
                    help='Comma-separated 1-based peptide position per template position (flattened cluster order). Use 0 to skip a position.')
    ap.add_argument('--degen-mj-relative', action='store_true', default=True,
                    help='Peptide-aware degeneration default: keep substitutions with MJ equal-or-better than template-residue MJ at each mapped position.')
    ap.add_argument('--no-degen-mj-relative', dest='degen_mj_relative', action='store_false',
                    help='Disable relative MJ filter and use absolute --degen-mj-max threshold instead.')
    ap.add_argument('--degen-mj-relative-max-worse', type=float, default=5.0,
                    help='In relative MJ mode, allow substitutions up to this much worse than template-contact MJ (default: +5.0).')
    ap.add_argument('--degen-mj-max', type=float, default=0.0,
                    help='For peptide-aware degeneration with --no-degen-mj-relative: keep substitutions with raw MJ <= this threshold (default: 0.0).')
    ap.add_argument('--max-candidates-per-cluster', type=int, default=0,
                    help='Per-cluster candidate cap (0 = no cap, exact candidate retention).')
    ap.add_argument('--inter-cluster-rmsd-cutoff', type=float, default=None,
                    help='Hard cutoff on RMSD over inter-cluster pair distances.')
    ap.add_argument('--inter-pair-tol', type=float, default=None,
                    help='Hard per-pair |distance error| pre-filter across clusters.')
    ap.add_argument('--global-pairdist-cutoff', type=float, default=None,
                    help='Hard cutoff on pairwise-distance RMSD over the full selected template positions.')
    ap.add_argument('--best-assignment', action='store_true',
                    help='Optimize to lowest summed cluster pairdist RMSD per protein (slower). Default: first feasible assignment.')
    ap.add_argument('--seed-cluster-index', type=int, default=None,
                    help='0-based cluster index to evaluate first (optional).')
    ap.add_argument('--seed-best-only', action='store_true',
                    help='Use only top candidate of --seed-cluster-index as seed.')
    ap.add_argument('--jobs', type=int, default=1,
                    help='Number of worker processes for per-protein search.')
    ap.add_argument('--parallel-backend', choices=('auto', 'process', 'thread'), default='auto',
                    help='Parallel executor backend. auto=process then thread fallback when process pools are unavailable.')

    ap.add_argument('--pulldown', default='data/r3_mapped_ac_ge2_minus_r1r2_sanitized.tsv',
                    help='Pulldown target TSV used as the search universe (pulldown-only semantics).')
    ap.add_argument('--pulldown-mapping-tsv', default='',
                    help='Optional mapping TSV (with seed_candidates) used by pulldown matcher.')
    ap.add_argument('--multiplicity-sweep', action='store_true',
                    help='Count disjoint motif assignments per protein and output multiplicity sweep.')
    ap.add_argument('--multiplicity-max-per-protein', type=int, default=100,
                    help='Upper bound on disjoint assignments counted per protein.')
    ap.add_argument('--accession-map', default='data/accession_to_entry_map.tsv',
                    help='Accession/entry alias mapping table for robust ID canonicalization.')
    ap.add_argument('--out', default='pairdist_hardcut.tsv')
    ap.add_argument('--progress-every', type=int, default=25,
                    help='Print progress every N proteins (0 disables).')
    args = ap.parse_args()

    clusters = build_clusters(args)
    for c in clusters:
        c['template_points'] = read_template_points(args.ecoli_batch, args.template_entry, c['template_res'])

    flat_template = []
    flat_cluster = []
    for ci, c in enumerate(clusters):
        for p in c['template_points']:
            flat_template.append(p)
            flat_cluster.append(ci)
    global_pair_refs = {}
    total_global_pairs = 0
    for i in range(len(flat_template) - 1):
        for j in range(i + 1, len(flat_template)):
            global_pair_refs[(i, j)] = dist(flat_template[i], flat_template[j])
            total_global_pairs += 1
    inter_pairs = []
    inter_pair_refs = {}
    for i in range(len(flat_template) - 1):
        for j in range(i + 1, len(flat_template)):
            if flat_cluster[i] != flat_cluster[j]:
                inter_pairs.append((i, j))
                inter_pair_refs[(i, j)] = dist(flat_template[i], flat_template[j])
    cluster_offsets = []
    off = 0
    for c in clusters:
        cluster_offsets.append(off)
        off += len(c['template_points'])

    entry_to_uid = load_entry_to_uid(args.ecoli_batch)
    entries = sorted(entry_to_uid.keys())
    acc_to_entries, entry_to_acc = load_accession_alias_map(args.accession_map)

    pulldown = load_pulldown(args.pulldown)
    pulldown_seed_candidates = load_pulldown_seed_candidates(args.pulldown_mapping_tsv)
    pulldown_token_index = build_pulldown_token_index(
        pulldown, acc_to_entries, entry_to_acc, pulldown_seed_candidates
    )
    if pulldown is not None:
        kept = []
        for e in entries:
            uid_raw = entry_to_uid.get(e, '')
            uid_norm = canonical_uniprot_id(uid_raw, acc_to_entries, entry_to_acc)
            if is_rabit_contaminant(uid_norm, e, acc_to_entries):
                continue
            if match_pulldown_targets(uid_raw, e, pulldown_token_index, acc_to_entries, entry_to_acc):
                kept.append(e)
        entries = kept

    tasks = []
    for e in entries:
        uid_raw = entry_to_uid.get(e, '')
        uid = canonical_uniprot_id(uid_raw, acc_to_entries, entry_to_acc)
        if is_rabit_contaminant(uid, e, acc_to_entries):
            continue
        if uid:
            base = (
                e, uid, args.ecoli_batch, clusters, args.max_candidates_per_cluster,
                cluster_offsets, inter_pair_refs, len(inter_pairs),
                args.inter_cluster_rmsd_cutoff, args.inter_pair_tol,
                global_pair_refs, total_global_pairs, args.global_pairdist_cutoff,
                args.best_assignment,
                args.seed_cluster_index, args.seed_best_only
            )
            if args.multiplicity_sweep:
                tasks.append(base + (args.multiplicity_max_per_protein,))
            else:
                tasks.append(base)

    out_rows = []
    total_tasks = len(tasks)
    t_start = time.time()
    def log_progress(done, hits, stage):
        if args.progress_every <= 0:
            return
        if done != total_tasks and (done % args.progress_every) != 0:
            return
        elapsed = time.time() - t_start
        pct = (100.0 * done / total_tasks) if total_tasks else 100.0
        print(f'progress {stage} {done}/{total_tasks} ({pct:.1f}%) hits={hits} elapsed_s={elapsed:.1f}', flush=True)

    can_parallel = args.jobs > 1
    if can_parallel:
        fn = evaluate_entry_multiplicity if args.multiplicity_sweep else evaluate_entry
        ran_parallel = False
        last_err = None

        def run_parallel(executor_cls, stage, chunksize=16, mp_context=None):
            attempt_rows = []
            kwargs = {'max_workers': args.jobs}
            if mp_context is not None:
                kwargs['mp_context'] = mp_context
            with executor_cls(**kwargs) as ex:
                for i, row in enumerate(ex.map(fn, tasks, chunksize=chunksize), start=1):
                    if row is not None:
                        attempt_rows.append(row)
                    log_progress(i, len(attempt_rows), stage)
            return attempt_rows

        backend = args.parallel_backend
        process_allowed = backend in ('auto', 'process')
        thread_allowed = backend in ('auto', 'thread')

        process_available = False
        process_probe_err = None
        if process_allowed:
            try:
                with ProcessPoolExecutor(max_workers=1) as ex:
                    list(ex.map(abs, [1], chunksize=1))
                process_available = True
            except Exception as e:
                process_probe_err = e
                if backend == 'auto':
                    print('parallel_process_unavailable_thread_fallback', e)
                else:
                    print('parallel_process_unavailable', e)

        # Important: each backend attempt is all-or-nothing.
        # If an attempt fails mid-map, discard partial rows to avoid duplicates.
        if process_available:
            try:
                out_rows = run_parallel(ProcessPoolExecutor, 'parallel-default', chunksize=16)
                ran_parallel = True
                print('parallel_backend process-default')
            except Exception as e:
                last_err = e

            # If default process pool fails, try spawn context.
            if not ran_parallel:
                try:
                    out_rows = run_parallel(
                        ProcessPoolExecutor,
                        'parallel-spawn',
                        chunksize=16,
                        mp_context=mp.get_context("spawn"),
                    )
                    ran_parallel = True
                    print('parallel_backend process-spawn')
                except Exception as e2:
                    last_err = e2

        # ThreadPool fallback supports sandbox environments where process pools are blocked.
        if not ran_parallel and thread_allowed:
            try:
                out_rows = run_parallel(ThreadPoolExecutor, 'parallel-thread', chunksize=1)
                ran_parallel = True
                print('parallel_backend thread')
            except Exception as e3:
                last_err = e3

        # Final fallback: deterministic serial evaluation.
        if not ran_parallel:
            if process_probe_err is not None and last_err is None:
                last_err = process_probe_err
            print(f'parallel_fallback_serial {last_err}')
            for i, t in enumerate(tasks, start=1):
                row = evaluate_entry_multiplicity(t) if args.multiplicity_sweep else evaluate_entry(t)
                if row is not None:
                    out_rows.append(row)
                log_progress(i, len(out_rows), 'serial-fallback')
    else:
        for i, t in enumerate(tasks, start=1):
            row = evaluate_entry_multiplicity(t) if args.multiplicity_sweep else evaluate_entry(t)
            if row is not None:
                out_rows.append(row)
            log_progress(i, len(out_rows), 'serial')

    if args.multiplicity_sweep:
        out_rows.sort(key=lambda r: (
            -int(r.get('multiplicity_disjoint', 0)),
            float(r.get('global_pairdist_rmsd', '9999')),
            (r.get('entry') or ''),
        ))
        detail_out = args.out.replace('.tsv', '_per_entry.tsv') if args.out.endswith('.tsv') else (args.out + '_per_entry.tsv')
        detail_headers = ['rank', 'uniprot_id', 'entry', 'multiplicity_disjoint',
                          'sum_cluster_pairdist_rmsd', 'global_pairdist_rmsd', 'inter_pairdist_rmsd']
        for i in range(1, len(clusters) + 1):
            detail_headers.extend([f'cluster{i}_pattern', f'cluster{i}', f'cluster{i}_pairdist_rmsd'])
        with open(detail_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=detail_headers, delimiter='\t')
            w.writeheader()
            for i, r in enumerate(out_rows, start=1):
                rr = {k: r.get(k, '') for k in detail_headers if k != 'rank'}
                rr['rank'] = i
                w.writerow(rr)

        class_by_target = {}
        for pr in (pulldown or []):
            tid = pulldown_row_target_id(pr, acc_to_entries, entry_to_acc)
            if tid and 'class' in pr:
                class_by_target[tid] = pr.get('class', '')
        mult_by_target = {}
        for r in out_rows:
            uid = r.get('uniprot_id') or ''
            entry = r.get('entry') or ''
            mval = int(r.get('multiplicity_disjoint', 0) or 0)
            targets = match_pulldown_targets(uid, entry, pulldown_token_index, acc_to_entries, entry_to_acc) if pulldown is not None else {(uid or '').strip().upper()}
            if not targets:
                # Keep non-pulldown hits in sweep accounting as explicit class.
                fallback = (uid or '').strip().upper() or (entry or '').strip().upper()
                if fallback:
                    tid = f"NOT_PULLDOWN::{fallback}"
                    targets = {tid}
                    class_by_target[tid] = 'Not_pulldown'
            for t in targets:
                prev = mult_by_target.get(t, 0)
                if mval > prev:
                    mult_by_target[t] = mval
        max_k = max(mult_by_target.values()) if mult_by_target else 0
        sweep_rows = []
        for k in range(1, max_k + 1):
            ids = [t for t, v in mult_by_target.items() if v >= k]
            row = {
                'k': k,
                'targets_ge_k': len(ids),
                'APIM_only': 0,
                'Both': 0,
                'EYFP_only': 0,
                'Not_pulldown': 0,
            }
            if class_by_target:
                for t in ids:
                    c = class_by_target.get(t, '')
                    if c in ('APIM_only', 'Both', 'EYFP_only', 'Not_pulldown'):
                        row[c] += 1
                    elif not c:
                        row['Not_pulldown'] += 1
            sweep_rows.append(row)
        with open(args.out, 'w', newline='') as f:
            headers = ['k', 'targets_ge_k', 'APIM_only', 'Both', 'EYFP_only', 'Not_pulldown']
            w = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
            w.writeheader()
            for r in sweep_rows:
                w.writerow(r)
        print(args.out)
        print('count', len(sweep_rows))
        print('detail', detail_out)
        print('targets', len(mult_by_target))
    else:
        # Rank by global geometry fit first; use inter-cluster fit, then summed cluster fit.
        out_rows.sort(key=lambda r: (
            float(r['global_pairdist_rmsd']),
            float(r['inter_pairdist_rmsd']),
            float(r['sum_cluster_pairdist_rmsd']),
        ))

        headers = [
            'rank',
            'uniprot_id',
            'entry',
            'sum_cluster_pairdist_rmsd',
            'global_pairdist_rmsd',
            'inter_pairdist_rmsd',
        ]
        for i in range(1, len(clusters) + 1):
            headers.extend([f'cluster{i}_pattern', f'cluster{i}', f'cluster{i}_pairdist_rmsd'])

        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
            w.writeheader()
            for i, r in enumerate(out_rows, start=1):
                rr = dict(r)
                rr['rank'] = i
                w.writerow(rr)

        print(args.out)
        if pulldown is not None:
            overlap_targets = set()
            for r in out_rows:
                uid = r.get('uniprot_id') or ''
                entry = r.get('entry') or ''
                overlap_targets |= match_pulldown_targets(uid, entry, pulldown_token_index, acc_to_entries, entry_to_acc)
            print('count', len(out_rows))
            print('overlap_with_pulldown', len(overlap_targets))
        else:
            print('count', len(out_rows))


if __name__ == '__main__':
    main()
