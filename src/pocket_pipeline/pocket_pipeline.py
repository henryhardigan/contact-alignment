#!/usr/bin/env python3
# ruff: noqa: E501
"""Pocket scoring pipeline (sidechain-centroid distances).

Implements:
- Template from DPO3B_ECOLI DNA clamp loader pocket (19 residues; see DNAN_POCKET_RESIDUES_DPO3B)
- Candidate generation with centroid radius filter
- Optional min/max pair distance, prefilter tolerance
- Unweighted or weighted RMSD over pairwise distances
- Top-M metrics with pulldown enrichment/odds

Usage examples:
  python3 pocket_pipeline.py --mode metrics
  python3 pocket_pipeline.py --pattern "H [RK] [LIV] [DE] Y" --mode metrics
  python3 pocket_pipeline.py --pattern "H [RK] [LIVM] [DE] Y" --mode rank-pulldown
  python3 pocket_pipeline.py --pattern "H [RK] 4[VILM] [DE] Y"

Outputs TSVs in cwd.

Note on similarity metrics:
- For cross-structure pocket comparisons, report pairwise-distance RMSD
  (RMSD over all intra-pocket pair distances). This is rigid-body invariant
  and more stable than coordinate RMSD when global orientations differ.
"""
import argparse
import csv
import glob
import heapq
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
import signal
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineResult:
    candidates_raw: list
    entry_to_uid: dict
    pulldown: set

# Variant definition:
# - Tokens are produced by parse_pattern(), preserving order.
# - For bracket tokens like "[LIVM]", variants are any non-empty subset of letters,
#   rendered as "[...]" with letters sorted.
# - For fixed tokens like "H" or "Y", no variation unless explicitly enabled.
# - The number of variants is the product of (2^k - 1) for each bracket of size k.

# --- multiprocessing globals (set once per worker) ---
_G = {}
_PAIRS_CACHE = {}
_TDIST_CACHE = {}


def _init_variant_worker(shared):
    """
    Runs once per worker process. Stores large read-only objects in globals
    so each task only passes a motif string.
    """
    global _G
    _G = shared


def _variant_task(motif: str):
    args = _G["args"]
    entry_to_uid = _G["entry_to_uid"]
    pulldown = _G["pulldown"]
    N_univ = _G["N_univ"]
    M = _G["variant_M"]
    M_effective = M if M is not None else 10**9
    M_label = M if M is not None else "ALL"

    _, candidates_raw_v, _, _, _ = build_candidates(
        motif,
        args,
        _G["res_cache"],
        entry_to_uid,
        _G["tmpl_base"],
        fixed_mapping=True,
        tmpl_meta=_G.get("tmpl_meta"),
    )
    if not candidates_raw_v:
        return None

    # Align variant-search behavior with pocket-search by merging intra-protein pockets.
    candidates_raw_v = merge_pockets_within_uid(candidates_raw_v, args)

    top_sites = compute_top_sites_for_m(candidates_raw_v, args, M_effective)
    if not top_sites:
        return None

    hit_proteins = {s["uid"] for s in top_sites}
    P = len(hit_proteins)
    if P == 0:
        return None
    P_pulldown = len(hit_proteins & pulldown)

    bg = (len(pulldown) / N_univ) if N_univ else 0.0
    fold = ((P_pulldown / P) / bg) if (P > 0 and bg > 0) else math.inf

    a = P_pulldown
    b = P - P_pulldown
    c = len(pulldown) - P_pulldown
    d = (N_univ - len(pulldown)) - b
    odds = (a * d) / (b * c) if b > 0 and c > 0 else float('inf')
    p = hypergeom_sf(a - 1, N_univ, len(pulldown), P)

    return {
        "motif": motif,
        "M": M_label,
        "P": P,
        "P_pulldown": P_pulldown,
        "fold_enrichment": fold,
        "odds_ratio": odds,
        "p_value": p,
        "hit_proteins": hit_proteins,
    }

# -------------------- config helpers --------------------

def parse_pattern(pattern: str) -> list[tuple[str, str]]:
    """Parse pattern like 'H [RK] [LIVM] [DE] Y' into class list.
    Returns list of (class_key, allowed_aa_string).
    Supports multiplicity tokens like '4[VILM]' or '4 aliphatics'.
    """
    tokens = parse_pattern_tokens(pattern)
    out = []
    cls_ord = []
    for tok in tokens:
        aas = tok['letters']
        cls_ord.append(chr(ord('A') + len(cls_ord)))
        out.append((cls_ord[-1], aas))
    return out


def parse_pattern_tokens(pattern: str, return_info: bool = False):
    tokens = pattern.replace(',', ' ').split()
    out = []
    used_count_or_alias = False
    i = 0
    aliases = {
        'aliphatic': '[VILM]',
        'aliphatics': '[VILM]',
        'basic': '[RKH]',
        'acidic': '[DE]',
        'aromatic': '[FWY]',
        'polar': '[STNQ]',
    }
    while i < len(tokens):
        tok = tokens[i]
        count = 1
        # handle "4[VILM]" or "4H" forms
        m = re.match(r'^(\d+)(\[.*\]|[A-Za-z]+)$', tok)
        if m:
            count = int(m.group(1))
            tok = m.group(2)
            used_count_or_alias = True
        elif tok.isdigit() and i + 1 < len(tokens):
            # handle "4 [VILM]" or "4 aliphatics"
            count = int(tok)
            tok = tokens[i + 1]
            i += 1
            used_count_or_alias = True

        if tok.lower() in aliases:
            tok = aliases[tok.lower()]
            used_count_or_alias = True

        if tok.startswith('[') and tok.endswith(']'):
            letters = tok[1:-1]
            for _ in range(count):
                out.append({'raw': tok, 'letters': letters, 'bracket': True})
        else:
            for _ in range(count):
                out.append({'raw': tok, 'letters': tok, 'bracket': False})
        i += 1
    if return_info:
        return out, {'used_count_or_alias': used_count_or_alias}
    return out


def load_pulldown(path: str) -> set:
    with open(path, newline='') as f:
        return set(row['uniprot_id'] for row in csv.DictReader(f, delimiter='\t'))


def load_entry_to_uid(ecoli_batch: str) -> dict[str, str]:
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
        except Exception as e:
            print(f"Warning: failed to read {meta}: {e}", file=sys.stderr)
    return entry_to_uid


def load_residues(path: str, allowed: set = None) -> dict[str, list[dict]]:
    res = defaultdict(list)
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            aa = row['aa']
            if allowed is not None and aa not in allowed:
                continue
            res[aa].append({
                'aa': aa,
                'res_id': row['res_id'],
                'resseq': row['resseq'],
                'x': float(row['sc_x']),
                'y': float(row['sc_y']),
                'z': float(row['sc_z']),
                'xyz': (float(row['sc_x']), float(row['sc_y']), float(row['sc_z'])),
            })
    return res


def dist(a, b) -> float:
    return math.dist((a['x'], a['y'], a['z']), (b['x'], b['y'], b['z']))


def distc(a, b) -> float:
    return math.dist(a, b)


DNAN_POCKET_RESIDUES_DPO3B = [
    ('R', '152'),
    ('L', '155'),
    ('T', '172'),
    ('D', '173'),
    ('G', '174'),
    ('H', '175'),
    ('R', '176'),
    ('L', '177'),
    ('P', '242'),
    ('R', '246'),
    ('V', '247'),
    ('V', '360'),
    ('M', '362'),
    ('N', '320'),
    ('Y', '323'),
    ('V', '344'),
    ('P', '363'),
    ('M', '364'),
    ('R', '365'),
]

DEFAULT_TEMPLATE_PATTERN = (
    "[RK] [VILM] [STNQ] [DE] [GAS] [HY] [RK] [VILM] P [RK] "
    "[VILM] [VILM] [VILM] [STNQ] [HY] [VILM] P [VILM] [RK]"
)


def template_from_dpo3b(ecoli_batch: str, return_meta: bool = False):
    """Template for DNA clamp loader pocket (two subsites) in DPO3B_ECOLI."""
    return template_from_entry_residues(
        ecoli_batch,
        'DPO3B_ECOLI',
        DNAN_POCKET_RESIDUES_DPO3B,
        return_meta=return_meta,
    )


def _build_unordered_mapping(
    classes: list[tuple[str, str]],
    tmpl_meta: dict[str, dict],
) -> dict[str, str]:
    """
    Assign class keys to template labels based on allowed AA sets, ignoring order.
    Prioritize identity matches when a class is a single AA.
    """
    class_keys = [ck for ck, _ in classes]
    allowed_map = {ck: set(aas) for ck, aas in classes}
    tmpl_labels = list(tmpl_meta.keys())

    options = {}
    for ck in class_keys:
        allowed = allowed_map[ck]
        opts = [
            lab for lab in tmpl_labels
            if tmpl_meta[lab]['aa'] in allowed
        ]
        if len(allowed) == 1:
            preferred = next(iter(allowed))
            opts = sorted(
                opts,
                key=lambda lab: 0 if tmpl_meta[lab]['aa'] == preferred else 1,
            )
        options[ck] = opts

    # Backtracking assignment, smallest option set first.
    order = sorted(class_keys, key=lambda k: len(options[k]))
    used = set()
    mapping = {}

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        ck = order[idx]
        for lab in options[ck]:
            if lab in used:
                continue
            used.add(lab)
            mapping[ck] = lab
            if backtrack(idx + 1):
                return True
            used.remove(lab)
            mapping.pop(ck, None)
        return False

    if not backtrack(0):
        raise RuntimeError('Failed to map classes to template residues (unordered mapping).')
    return mapping


def resolve_mapping(
    classes: list[tuple[str, str]],
    tmpl_base: dict[str, tuple[float, float, float]],
    tmpl_meta: dict[str, dict] | None,
    fixed_mapping: bool,
) -> dict[str, str]:
    class_keys = [c for c, _ in classes]
    if tmpl_meta:
        return _build_unordered_mapping(classes, tmpl_meta)
    if fixed_mapping:
        if all(ck in tmpl_base for ck in class_keys):
            return {ck: ck for ck in class_keys}
        pos_labels = ['H', 'R', 'L', 'D', 'Y']
        if len(classes) > len(pos_labels):
            raise RuntimeError('Pattern length exceeds supported template positions (H,R,L,D,Y).')
        mapping = {}
        for i, (ck, _) in enumerate(classes):
            label = pos_labels[i]
            if label not in tmpl_base:
                raise RuntimeError(f'Template missing label {label} for fixed mapping.')
            mapping[ck] = label
        return mapping
    mapping = {}
    for ck, aas in classes:
        if 'H' in aas and 'H' not in mapping.values():
            mapping[ck] = 'H'
        elif 'R' in aas and 'R' not in mapping.values():
            mapping[ck] = 'R'
        elif 'L' in aas and 'L' not in mapping.values():
            mapping[ck] = 'L'
        elif 'D' in aas and 'D' not in mapping.values():
            mapping[ck] = 'D'
        elif 'Y' in aas and 'Y' not in mapping.values():
            mapping[ck] = 'Y'
    for ck, aas in classes:
        if ck not in mapping:
            for aa in aas:
                if aa in tmpl_base and aa not in mapping.values():
                    mapping[ck] = aa
                    break
    if any(ck not in mapping for ck, _ in classes):
        raise RuntimeError('Failed to map all classes to template residues.')
    return mapping

def template_from_entry_residues(
    ecoli_batch: str,
    entry: str,
    residues: list[tuple[str, str]],
    return_meta: bool = False,
):
    path = os.path.join(ecoli_batch, entry, 'residues.tsv')
    res = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            res.append(row)

    def rc(r):
        return (float(r['sc_x']), float(r['sc_y']), float(r['sc_z']))

    tmpl = {}
    meta = {}
    for i, (aa, resseq) in enumerate(residues):
        hit = next((r for r in res if r['aa'] == aa and r['resseq'] == resseq), None)
        if hit is None:
            raise RuntimeError(f'Template residue {aa}{resseq} not found in {entry}')
        key = chr(ord('A') + i)
        tmpl[key] = rc(hit)
        meta[key] = {'aa': aa, 'resseq': resseq}
    if return_meta:
        return tmpl, meta
    return tmpl


WEIGHTS = {
    ('H','D'): 2.0,
    ('D','Y'): 2.0,
    ('R','Y'): 2.0,
    ('L','Y'): 2.0,
    ('H','R'): 0.5,
    ('H','Y'): 0.5,
}


def build_candidates(
    pattern: str,
    args,
    res_cache,
    entry_to_uid,
    tmpl_base,
    fixed_mapping: bool = False,
    Tdist_override=None,
    tmpl_meta=None,
):
    classes = parse_pattern(pattern)
    class_keys = [c for c, _ in classes]

    # AA class helper for match counting
    class_map = {
        'aliphatic': set('VILM'),
        'basic': set('RKH'),
        'acidic': set('DE'),
        'aromatic': set('FWY'),
        'polar': set('STNQ'),
        'special': set('PG'),
    }
    def aa_class(a):
        for k, v in class_map.items():
            if a in v:
                return k
        return 'other'

    # mapping from class key to template residue label
    mapping = resolve_mapping(classes, tmpl_base, tmpl_meta, fixed_mapping)

    Tcoords = {ck: tmpl_base[mapping[ck]] for ck, _ in classes}
    Tpoints = [Tcoords[ck] for ck in class_keys]
    ck_key = tuple(class_keys)
    if ck_key in _PAIRS_CACHE:
        pairs = _PAIRS_CACHE[ck_key]
    else:
        pairs = pair_list(class_keys)
        _PAIRS_CACHE[ck_key] = pairs

    tdist_key = (ck_key, tuple((ck, mapping[ck]) for ck in class_keys))
    if Tdist_override is not None:
        Tdist = Tdist_override
    elif tdist_key in _TDIST_CACHE:
        Tdist = _TDIST_CACHE[tdist_key]
    else:
        Tdist = build_tdist_from_tcoords(Tcoords, class_keys)
        _TDIST_CACHE[tdist_key] = Tdist

    # cluster scoring config (optional)
    cluster_groups = []
    Tcluster_centroids = []
    Tcluster_pair_dists = []
    Tcluster_internal_dists = []
    cluster_idx = {}
    Tcentroid = None
    Tcdist = None
    template_mean_d = None
    if getattr(args, "cluster_groups", None):
        cluster_groups = parse_cluster_groups(args.cluster_groups, class_keys)
        for i, grp in enumerate(cluster_groups):
            for ck in grp:
                cluster_idx[ck] = i
        # template cluster centroids
        for grp in cluster_groups:
            pts = [Tcoords[ck] for ck in grp]
            Tcluster_centroids.append(centroid(pts))
        # template inter-cluster centroid distances
        for i in range(len(Tcluster_centroids)):
            for j in range(i + 1, len(Tcluster_centroids)):
                Tcluster_pair_dists.append(distc(Tcluster_centroids[i], Tcluster_centroids[j]))
        # template internal distances per cluster (flattened)
        for grp in cluster_groups:
            if len(grp) < 2:
                continue
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    Tcluster_internal_dists.append(distc(Tcoords[grp[i]], Tcoords[grp[j]]))
        # template centroid distances for filtering
        Tcentroid = bbox_center([Tcoords[k] for k in class_keys])
        Tcdist = [distc(Tcoords[k], Tcentroid) for k in class_keys]

    # Default min/max pair bounds to template if not provided.
    if args.min_pair is None:
        if cluster_groups:
            internal_t = [
                Tdist[p] for p in pairs
                if cluster_idx[p[0]] == cluster_idx[p[1]]
            ]
            min_t = min(internal_t) if internal_t else 0.0
        else:
            min_t = min(Tdist.values()) if Tdist else 0.0
        args.min_pair = max(0.0, min_t - 2.0)
    if args.max_pair is None:
        if cluster_groups:
            internal_t = [
                Tdist[p] for p in pairs
                if cluster_idx[p[0]] == cluster_idx[p[1]]
            ]
            max_t = max(internal_t) if internal_t else 0.0
            args.max_pair = (max_t + 2.0) if internal_t else float('inf')
        else:
            args.max_pair = (max(Tdist.values()) + 2.0) if Tdist else float('inf')

    # Default radius/pair-prune to template-derived values if not provided.
    if args.radius is None or args.min_radius is None:
        template_coords = [Tcoords[k] for k in class_keys]
        if template_coords:
            tc = bbox_center(template_coords)
            dists = [distc(p, tc) for p in template_coords]
            mean_d = sum(dists) / len(dists)
            var_d = sum((x - mean_d) ** 2 for x in dists) / len(dists)
            std_d = math.sqrt(var_d)
            template_mean_d = mean_d
            if args.radius is None:
                args.radius = mean_d + 2.0 * std_d
            if args.min_radius is None:
                # Allow small slack below template min distance to avoid over-filtering near-center residues.
                args.min_radius = max(0.0, min(dists) - 1.0)
        else:
            if args.radius is None:
                args.radius = 8.0
            if args.min_radius is None:
                args.min_radius = 0.0
    if args.pair_prune is None:
        args.pair_prune = args.max_pair if args.max_pair and math.isfinite(args.max_pair) else 2 * args.radius

    pair_prune = args.pair_prune

    require_residues = getattr(args, "require_residues", None) or []
    require_template = getattr(args, "require_template", None) or {}
    candidates = []  # (score, entry, uid, residues...)
    total_entries = len(res_cache)
    for i_entry, (e, res_by_aa) in enumerate(res_cache.items(), start=1):
        if getattr(args, "abort_flag", None) and args.abort_flag.get("hit"):
            break
        if args.progress_every and i_entry % args.progress_every == 0:
            print(f"progress {i_entry}/{total_entries} entries", file=sys.stderr)
        uid = entry_to_uid.get(e)
        if not uid:
            continue
        # spatial grid for fast pair_prune checks
        grid_res = []
        for aa_list in res_by_aa.values():
            grid_res.extend(aa_list)
        # upfront radius prefilter on individual residues (bbox center of template)
        if args.min_radius is not None or args.radius is not None:
            tcenter = bbox_center([Tcoords[k] for k in class_keys])
            for aa_key in list(res_by_aa.keys()):
                filtered = [
                    r for r in res_by_aa[aa_key]
                    if (
                        args.radius is None
                        or (
                            distc(r['xyz'], tcenter) <= args.radius
                            or (
                                args.radius_far_slack is not None
                                and template_mean_d is not None
                                and distc(r['xyz'], tcenter) >= template_mean_d
                                and distc(r['xyz'], tcenter) <= args.radius + args.radius_far_slack
                            )
                        )
                    )
                    and (args.min_radius is None or distc(r['xyz'], tcenter) >= args.min_radius)
                ]
                res_by_aa[aa_key] = filtered
        grid = build_spatial_grid(grid_res, pair_prune)
        neighbor_ids = {}
        byclass = defaultdict(list)
        for ck, aas in classes:
            for aa in aas:
                if aa in res_by_aa:
                    byclass[ck].extend(res_by_aa[aa])
        if args.pos_radius is not None and args.pos_radius > 0:
            for ck in class_keys:
                tpt = Tcoords[ck]
                byclass[ck] = [r for r in byclass[ck] if distc(r['xyz'], tpt) <= args.pos_radius]
        if args.template_first:
            for ck in class_keys:
                tpt = Tcoords[ck]
                byclass[ck].sort(key=lambda r: distc(r['xyz'], tpt))
        if args.debug_template_entry and e == args.debug_template_entry and tmpl_meta:
            missing = []
            filtered = []
            for ck in class_keys:
                lab = mapping.get(ck)
                meta = tmpl_meta.get(lab) if lab else None
                if not meta:
                    missing.append(f"{ck}:no_meta")
                    continue
                aa = meta['aa']
                resseq = meta['resseq']
                if aa not in res_by_aa:
                    missing.append(f"{ck}:{aa}{resseq}:aa_missing")
                    continue
                hit = next((r for r in res_by_aa[aa] if r['resseq'] == resseq), None)
                if hit is None:
                    missing.append(f"{ck}:{aa}{resseq}:res_missing")
                    continue
                if hit not in byclass[ck]:
                    filtered.append(f"{ck}:{aa}{resseq}")
            if missing or filtered:
                print(f"[debug-template] entry={e} missing={missing} filtered={filtered}", file=sys.stderr)
            else:
                print(f"[debug-template] entry={e} all template residues present after filters", file=sys.stderr)
        if (
            args.debug_template_combo
            and args.debug_template_entry
            and e == args.debug_template_entry
            and tmpl_meta
            and args.segment_search
            and cluster_groups
        ):
            # Build exact template combo and evaluate segment-search gates.
            chosen_map = {}
            ok = True
            for ck in class_keys:
                lab = mapping.get(ck)
                meta = tmpl_meta.get(lab) if lab else None
                if not meta:
                    ok = False
                    break
                aa = meta['aa']
                resseq = meta['resseq']
                hit = next((r for r in res_by_aa.get(aa, []) if r['resseq'] == resseq), None)
                if hit is None:
                    ok = False
                    break
                chosen_map[ck] = hit

            if not ok:
                print("[debug-template-combo] exact combo could not be built", file=sys.stderr)
            else:
                # Stage: prefilter radius and centroid RMSD
                ordered = [chosen_map[k] for k in class_keys]
                coords = [c['xyz'] for c in ordered]
                c_pf = bbox_center(coords)
                max_r = max(distc(p, c_pf) for p in coords)
                min_r = min(distc(p, c_pf) for p in coords)
                max_ok = (
                    max_r <= args.radius
                    or (
                        args.radius_far_slack is not None
                        and template_mean_d is not None
                        and max_r >= template_mean_d
                        and max_r <= args.radius + args.radius_far_slack
                    )
                )
                radius_ok = max_ok and (min_r >= args.min_radius)

                cd_ok = True
                cd_val = None
                if args.stage0_centroid_rmsd is not None and Tcdist is not None:
                    cd = [distc(p, c_pf) for p in coords]
                    cd_val = rmsd_over_pairs(cd, Tcdist)
                    cd_ok = cd_val <= args.stage0_centroid_rmsd

                # Segment-search centroid RMSD between clusters
                Ccentroids = []
                Tcentroids_sub = []
                for gi, grp in enumerate(cluster_groups):
                    pts = [chosen_map[ck]['xyz'] for ck in grp]
                    Ccentroids.append(centroid(pts))
                    Tcentroids_sub.append(Tcluster_centroids[gi])
                Cpair = []
                Tpair = []
                for i in range(len(Ccentroids)):
                    for j in range(i + 1, len(Ccentroids)):
                        Cpair.append(distc(Ccentroids[i], Ccentroids[j]))
                        Tpair.append(distc(Tcentroids_sub[i], Tcentroids_sub[j]))
                rmsd_centroid = rmsd_over_pairs(Cpair, Tpair) if Tpair else 0.0
                max_score_ok = True
                if args.max_score is not None and rmsd_centroid > args.max_score:
                    max_score_ok = False

                print(
                    "[debug-template-combo]",
                    f"radius_ok={radius_ok}",
                    f"centroid_ok={cd_ok}",
                    f"max_score_ok={max_score_ok}",
                    f"rmsd_centroid={fmt_sig(rmsd_centroid)}",
                    f"cd_rmsd={fmt_sig(cd_val)}",
                    file=sys.stderr,
                )
        if args.min_match is None:
            if any(len(byclass[c]) == 0 for c in class_keys):
                continue
        else:
            available = sum(1 for c in class_keys if len(byclass[c]) > 0)
            if available < args.min_match:
                continue
            # Cluster-level availability gate (early skip when max possible < min_match).
            if args.segment_search and cluster_groups:
                max_possible = 0
                for grp in cluster_groups:
                    max_possible += sum(1 for ck in grp if len(byclass[ck]) > 0)
                if max_possible < args.min_match:
                    continue

        heap = []
        cand_before = len(candidates)

        class_keys_sorted = sorted(class_keys, key=lambda k: len(byclass[k]))

        # precompute cluster membership for early centroid checks
        cluster_members = {}
        for idx_c, grp in enumerate(cluster_groups):
            cluster_members[idx_c] = set(grp)

        # Segmented search: solve each cluster independently, then stitch by centroid RMSD.
        if args.segment_search and cluster_groups:
            if args.debug_segment_entry and e == args.debug_segment_entry:
                for gi, grp in enumerate(cluster_groups, start=1):
                    counts = {ck: len(byclass[ck]) for ck in grp}
                    present = sum(1 for v in counts.values() if v > 0)
                    print(
                        f"[debug] entry={e} cluster {gi} keys={len(grp)} present={present} counts={counts}",
                        file=sys.stderr,
                    )
            def identity_bonus_for(ck, r):
                if not tmpl_meta or ck not in mapping:
                    return 0.0
                allowed = None
                for k, aas in classes:
                    if k == ck:
                        allowed = aas
                        break
                if allowed is None:
                    return 0.0
                lab = mapping[ck]
                meta = tmpl_meta.get(lab)
                if not meta:
                    return 0.0
                if r['aa'] == meta['aa'] and r['resseq'] == meta['resseq']:
                    return args.identity_bonus if len(allowed) == 1 else args.identity_bonus_degen
                return 0.0

            chosen_map = {}
            ok_segments = True
            for grp in cluster_groups:
                grp_keys = list(grp)
                tmpl_cost = None
                if args.min_match is not None and any(len(byclass[k]) == 0 for k in grp_keys):
                    # allow skipping this cluster entirely in K-of-18 mode
                    chosen_map.setdefault("__cluster_candidates__", []).append([({}, 0.0)])
                    continue
                # beam over this cluster only
                beam = [({}, 0.0)]
                expansions = 0
                for ck in sorted(grp_keys, key=lambda k: len(byclass[k])):
                    new_beam = []
                    for sub_map, cost in beam:
                        for r in byclass[ck]:
                            used_ids = {v['res_id'] for v in sub_map.values()}
                            if r['res_id'] in used_ids:
                                continue
                            new_map = dict(sub_map)
                            new_map[ck] = r
                            added_cost = 0.0
                            if len(new_map) >= 2:
                                for other_ck, other_r in new_map.items():
                                    if other_ck == ck:
                                        continue
                                    a, b = (ck, other_ck)
                                    if (a, b) not in Tdist:
                                        a, b = (other_ck, ck)
                                    dc = distc(r['xyz'], other_r['xyz'])
                                    d_ref = Tdist[(a, b)]
                                    added_cost += (dc - d_ref) ** 2
                            # apply identity bonus (also for singleton clusters)
                            added_cost -= identity_bonus_for(ck, r)
                            new_beam.append((new_map, cost + added_cost))
                            expansions += 1
                    if args.beam_width is not None and args.beam_width > 0:
                        new_beam.sort(key=lambda t: t[1])
                        beam = new_beam[:args.beam_width]
                    else:
                        beam = new_beam
                if not beam:
                    ok_segments = False
                    break
                # keep top-N assignments in this cluster (cap to avoid combinatorial explosion)
                beam.sort(key=lambda t: t[1])
                if args.cluster_top_n and args.cluster_top_n > 0:
                    top_n = beam[:args.cluster_top_n]
                else:
                    top_n = beam
                if args.debug_segment_entry and e == args.debug_segment_entry:
                    # compute template cluster cost and rank if possible
                    if tmpl_meta:
                        tmpl_map = {}
                        ok = True
                        for ck in grp_keys:
                            lab = mapping.get(ck)
                            meta = tmpl_meta.get(lab) if lab else None
                            if not meta:
                                ok = False
                                break
                            # find exact residue in byclass
                            hit = next(
                                (r for r in byclass[ck]
                                 if r['aa'] == meta['aa'] and r['resseq'] == meta['resseq']),
                                None,
                            )
                            if hit is None:
                                ok = False
                                break
                            tmpl_map[ck] = hit
                        if ok:
                            # compute cost for template cluster assignment
                            tcost = 0.0
                            for a_idx, a in enumerate(grp_keys):
                                for b in grp_keys[a_idx + 1:]:
                                    aa, bb = (a, b)
                                    if (aa, bb) not in Tdist:
                                        aa, bb = (bb, aa)
                                    dc = distc(tmpl_map[a]['xyz'], tmpl_map[b]['xyz'])
                                    d_ref = Tdist[(aa, bb)]
                                    tcost += (dc - d_ref) ** 2
                            tcost -= sum(identity_bonus_for(ck, tmpl_map[ck]) for ck in grp_keys)
                            tmpl_cost = tcost
                            rank = 1 + sum(1 for _, c in beam if c < tcost)
                        else:
                            rank = None
                            tmpl_cost = None
                    else:
                        rank = None
                    rank_str = "na" if rank is None else str(rank)
                    cost_str = "na" if tmpl_cost is None else fmt_sig(tmpl_cost)
                    print(
                        f"[debug] entry={e} cluster_beam_expansions={expansions} topN={len(top_n)} "
                        f"tmpl_rank={rank_str} tmpl_cost={cost_str}",
                        file=sys.stderr,
                    )
                # stash for stitching
                chosen_map.setdefault("__cluster_candidates__", []).append(top_n)

            if not ok_segments:
                continue

            # Seed exact template combo first when available.
            # stitch clusters: take cartesian product of top-N per cluster, keep best by centroid RMSD
            cluster_cands = chosen_map.pop("__cluster_candidates__", [])
            if args.min_match is not None:
                # allow dropping entire clusters to enable K-of-18 selection
                cluster_cands = [cands + [({}, 0.0)] for cands in cluster_cands]
            if not cluster_cands:
                continue
            if args.debug_segment_entry and e == args.debug_segment_entry:
                sizes = [len(c) for c in cluster_cands]
                print(f"[debug] entry={e} cluster_cands_sizes={sizes}", file=sys.stderr)
            stitched = []
            stitched_total = 0
            stitched_pass = 0
            stitched_dup = 0
            stitched_short = 0
            stitched_max_len = 0

            # Prune early if remaining clusters can't reach min_match.
            max_sizes = [max((len(m) for m, _ in cands), default=0) for cands in cluster_cands]
            suffix_max = [0] * (len(max_sizes) + 1)
            for i in range(len(max_sizes) - 1, -1, -1):
                suffix_max[i] = suffix_max[i + 1] + max_sizes[i]

            def stitch_rec(idx, merged, used_ids):
                nonlocal stitched_total, stitched_pass, stitched_dup, stitched_short, stitched_max_len
                cur_len = len(merged)
                if args.min_match is not None:
                    if cur_len + suffix_max[idx] < args.min_match:
                        return
                if idx == len(cluster_cands):
                    if args.min_match is None and cur_len != len(class_keys):
                        return
                    if args.min_match is not None and cur_len < args.min_match:
                        if cur_len > stitched_max_len:
                            stitched_max_len = cur_len
                        stitched_short += 1
                        return
                    stitched_total += 1
                    # prefilter on radius/centroid before scoring
                    keys_pf = list(merged.keys()) if args.min_match is not None else list(class_keys)
                    ordered_pf = [merged[k] for k in keys_pf]
                    coords_pf = [c['xyz'] for c in ordered_pf]
                    c_pf = bbox_center(coords_pf)
                    max_r_pf = max(distc(p, c_pf) for p in coords_pf)
                    min_r_pf = min(distc(p, c_pf) for p in coords_pf)
                    max_ok = (
                        max_r_pf <= args.radius
                        or (
                            args.radius_far_slack is not None
                            and template_mean_d is not None
                            and max_r_pf >= template_mean_d
                            and max_r_pf <= args.radius + args.radius_far_slack
                        )
                    )
                    if not max_ok or min_r_pf < args.min_radius:
                        return
                    if args.stage0_centroid_rmsd is not None and Tcdist is not None:
                        cd_pf = [distc(p, c_pf) for p in coords_pf]
                        if args.min_match is not None:
                            tcenter = bbox_center([Tcoords[k] for k in keys_pf])
                            tcd_pf = [distc(Tcoords[k], tcenter) for k in keys_pf]
                            if rmsd_over_pairs(cd_pf, tcd_pf) > args.stage0_centroid_rmsd:
                                return
                        else:
                            if rmsd_over_pairs(cd_pf, Tcdist) > args.stage0_centroid_rmsd:
                                return
                    stitched_pass += 1
                    # centroid-pair RMSD between clusters (subset-aware for min_match)
                    Ccentroids = []
                    Tcentroids_sub = []
                    for gi, grp in enumerate(cluster_groups):
                        if all(ck in merged for ck in grp):
                            pts = [merged[ck]['xyz'] for ck in grp]
                            Ccentroids.append(centroid(pts))
                            Tcentroids_sub.append(Tcluster_centroids[gi])
                    Cpair = []
                    Tpair = []
                    for i in range(len(Ccentroids)):
                        for j in range(i + 1, len(Ccentroids)):
                            Cpair.append(distc(Ccentroids[i], Ccentroids[j]))
                            Tpair.append(distc(Tcentroids_sub[i], Tcentroids_sub[j]))
                    rmsd_centroid = rmsd_over_pairs(Cpair, Tpair) if Tpair else 0.0
                    if args.cluster_pair_tol is not None and args.cluster_pair_tol > 0:
                        if rmsd_centroid > args.cluster_pair_tol:
                            return
                    if args.max_score is not None and rmsd_centroid > args.max_score:
                        return
                    stitched.append((rmsd_centroid, dict(merged)))
                    return

                for sub_map, _ in cluster_cands[idx]:
                    if args.min_match is None:
                        ids = {v['res_id'] for v in sub_map.values()}
                        if used_ids & ids:
                            stitched_dup += 1
                            continue
                        used_ids.update(ids)
                        merged.update(sub_map)
                        stitch_rec(idx + 1, merged, used_ids)
                        for rid in ids:
                            used_ids.remove(rid)
                        for k in sub_map.keys():
                            merged.pop(k, None)
                    else:
                        merged.update(sub_map)
                        # resolve duplicates by keeping the best assignment per residue
                        resid_to_items = defaultdict(list)
                        for ck, r in merged.items():
                            resid_to_items[r['res_id']].append((ck, r))
                        dedup = {}
                        for items in resid_to_items.values():
                            if len(items) == 1:
                                ck, r = items[0]
                                dedup[ck] = r
                                continue
                            items.sort(
                                key=lambda it: (
                                    -identity_bonus_for(it[0], it[1]),
                                    it[0],
                                )
                            )
                            ck, r = items[0]
                            dedup[ck] = r
                        merged.clear()
                        merged.update(dedup)
                        stitch_rec(idx + 1, merged, used_ids)

            stitch_rec(0, {}, set())
            if not stitched:
                if args.debug_segment_entry and e == args.debug_segment_entry:
                    print(
                        f"[debug] entry={e} stitched_total={stitched_total} stitched_pass={stitched_pass} "
                        f"dup_skip={stitched_dup} short_skip={stitched_short} "
                        f"max_len={stitched_max_len}",
                        file=sys.stderr,
                    )
                continue
            stitched.sort(key=lambda t: t[0])
            stitched = stitched[:20]
            if args.debug_segment_entry and e == args.debug_segment_entry:
                print(f"[debug] entry={e} stitched_total={stitched_total} stitched_pass={stitched_pass}", file=sys.stderr)
                # per-cluster template presence in top-5
                for gi, grp in enumerate(cluster_groups, start=1):
                    tmpl_ok = False
                    for sub_map, _ in cluster_cands[gi - 1]:
                        ok = True
                        for ck in grp:
                            lab = mapping.get(ck)
                            if not lab or not tmpl_meta:
                                ok = False
                                break
                            aa = tmpl_meta[lab]['aa']
                            resseq = tmpl_meta[lab]['resseq']
                            r = sub_map.get(ck)
                            if not r or r['aa'] != aa or r['resseq'] != resseq:
                                ok = False
                                break
                        if ok:
                            tmpl_ok = True
                            break
                    print(f"[debug] cluster {gi} template_in_top5={tmpl_ok} size={len(grp)}", file=sys.stderr)
                print(f"[debug] entry={e} top {len(stitched)} stitched combos", file=sys.stderr)
                for i_dbg, (s_dbg, m_dbg) in enumerate(stitched, start=1):
                    ordered_dbg = [m_dbg[k] for k in class_keys if k in m_dbg]
                    coords_dbg = [c['xyz'] for c in ordered_dbg]
                    c_dbg = bbox_center(coords_dbg)
                    max_r = max(distc(p, c_dbg) for p in coords_dbg)
                    min_r = min(distc(p, c_dbg) for p in coords_dbg)
                    cd_ok = True
                    cd_val = None
                    if args.stage0_centroid_rmsd is not None and Tcdist is not None:
                        cd = [distc(p, c_dbg) for p in coords_dbg]
                        cd_val = rmsd_over_pairs(cd, Tcdist)
                        cd_ok = cd_val <= args.stage0_centroid_rmsd
                    labels_dbg = ",".join(f"{r['aa']}{r['resseq']}" for r in ordered_dbg)
                    cd_str = fmt_sig(cd_val) if cd_val is not None else "na"
                    print(f"[debug] {i_dbg} score={fmt_sig(s_dbg)} min_r={fmt_sig(min_r)} max_r={fmt_sig(max_r)} cd_rmsd={cd_str} cd_ok={cd_ok} residues={labels_dbg}", file=sys.stderr)

            for score, chosen_map in stitched:
                if args.max_score is not None and score > args.max_score:
                    continue
                if args.min_match is None and len(chosen_map) != len(class_keys):
                    continue
                if args.min_match is not None and len(chosen_map) < args.min_match:
                    continue
                if require_residues:
                    ok_req = True
                    for aa_req, resseq_req in require_residues:
                        if not any(
                            r["aa"] == aa_req and r["resseq"] == resseq_req
                            for r in chosen_map.values()
                        ):
                            ok_req = False
                            break
                    if not ok_req:
                        continue

                # min-match: keep all available positions, only enforce len >= min_match
                select_keys = [ck for ck in class_keys if ck in chosen_map]
                if args.min_match is not None and tmpl_meta:
                    match_count = 0
                    for ck in select_keys:
                        lab = mapping.get(ck)
                        if not lab:
                            continue
                        tmpl_aa = tmpl_meta[lab]['aa']
                        cand_aa = chosen_map[ck]['aa']
                        if aa_class(tmpl_aa) == aa_class(cand_aa):
                            match_count += 1
                    if match_count < args.min_match:
                        continue

                if len(select_keys) < (args.min_match or 0):
                    continue
                ordered = [chosen_map[k] for k in select_keys]
                coords = [c['xyz'] for c in ordered]
                c = bbox_center(coords)
                if max(distc(p, c) for p in coords) > args.radius:
                    continue
                if min(distc(p, c) for p in coords) < args.min_radius:
                    continue
                if args.stage0_centroid_rmsd is not None and Tcdist is not None:
                    cd = [distc(p, c) for p in coords]
                    tcenter = bbox_center([Tcoords[k] for k in select_keys])
                    tcd = [distc(Tcoords[k], tcenter) for k in select_keys]
                    if rmsd_over_pairs(cd, tcd) > args.stage0_centroid_rmsd:
                        continue

                sel_pairs = list(itertools.combinations(select_keys, 2))
                dist_list = tuple(
                    distc(chosen_map[a]['xyz'], chosen_map[b]['xyz'])
                    for a, b in sel_pairs
                )
                resid_set = set(r['res_id'] for r in ordered)

                # recompute score on selected keys
                score_sel = score
                if cluster_groups:
                    Ccentroids = []
                    Tcentroids = []
                    for grp in cluster_groups:
                        pts = [chosen_map[ck]['xyz'] for ck in grp if ck in select_keys]
                        if not pts:
                            continue
                        Ccentroids.append(centroid(pts))
                        tpts = [Tcoords[ck] for ck in grp if ck in select_keys]
                        Tcentroids.append(centroid(tpts))
                    Cpair = []
                    Tpair = []
                    for i in range(len(Ccentroids)):
                        for j in range(i + 1, len(Ccentroids)):
                            Cpair.append(distc(Ccentroids[i], Ccentroids[j]))
                            Tpair.append(distc(Tcentroids[i], Tcentroids[j]))
                    if Cpair:
                        score_sel = rmsd_over_pairs(Cpair, Tpair)
                    else:
                        # Fallback: centroid-distance RMSD on selected keys when no centroid pairs exist.
                        coords_sel = [chosen_map[k]['xyz'] for k in select_keys]
                        c_sel = bbox_center(coords_sel)
                        cd_sel = [distc(p, c_sel) for p in coords_sel]
                        tcenter = bbox_center([Tcoords[k] for k in select_keys])
                        tcd = [distc(Tcoords[k], tcenter) for k in select_keys]
                        score_sel = rmsd_over_pairs(cd_sel, tcd) if cd_sel else score

                payload = (score_sel, e, uid, resid_set, dist_list, *ordered)
                score_key = round(score_sel, 6)
                key = (-score_key, len(resid_set))
                if len(heap) < args.per_protein_cap:
                    heapq.heappush(heap, (key, payload))
                else:
                    if key > heap[0][0]:
                        heapq.heapreplace(heap, (key, payload))
                # ensure max-length candidate is kept when score is 0
                if score_key == 0.0:
                    best0 = getattr(args, "_best_zero", {})
                    prev = best0.get(e)
                    if prev is None or len(resid_set) > prev[0]:
                        best0[e] = (len(resid_set), payload)
                        args._best_zero = best0

            for _, payload in heap:
                candidates.append(payload)
            if args.stop_after_first_non_template and e != args.template_entry and len(candidates) > cand_before:
                candidates_raw = list(candidates)
                return candidates, candidates_raw, classes, mapping, Tdist
            continue

        # Beam search (no backtracking)
        beam = [({}, 0.0)]  # (chosen_map, partial_cost)
        for ck in class_keys_sorted:
            if getattr(args, "abort_flag", None) and args.abort_flag.get("hit"):
                break
            new_beam = []
            for chosen_map, cost in beam:
                if getattr(args, "abort_flag", None) and args.abort_flag.get("hit"):
                    break
                for r in byclass[ck]:
                    if getattr(args, "abort_flag", None) and args.abort_flag.get("hit"):
                        break
                    ok = True
                    # If pair_prune <= 0, neighbor pruning is disabled (no-op).
                    # Otherwise, only compare against chosen residues within the pair_prune grid neighborhood.
                    if chosen_map and pair_prune > 0 and not cluster_groups:
                        if r['res_id'] not in neighbor_ids:
                            neighbor_ids[r['res_id']] = {
                                rr['res_id'] for rr in neighbors_from_grid(grid, r, pair_prune)
                            }
                        nearby_ids = neighbor_ids[r['res_id']]
                        for prev_ck, prev in chosen_map.items():
                            if prev['res_id'] not in nearby_ids:
                                ok = False
                                break
                            if distc(r['xyz'], prev['xyz']) > pair_prune:
                                ok = False
                                break
                    if not ok:
                        continue
                    new_map = dict(chosen_map)
                    new_map[ck] = r
                    # partial prefilter on any available pair distances
                    added_cost = 0.0
                    if len(new_map) >= 2:
                        ok2 = True
                        for other_ck, other_r in new_map.items():
                            if other_ck == ck:
                                continue
                            a, b = (ck, other_ck)
                            if cluster_groups and cluster_idx[a] != cluster_idx[b]:
                                continue
                            if (a, b) not in Tdist:
                                a, b = (other_ck, ck)
                            dc = distc(r['xyz'], other_r['xyz'])
                            d_ref = Tdist[(a, b)]
                            added_cost += (dc - d_ref) ** 2
                        if not ok2:
                            continue
                    new_beam.append((new_map, cost + added_cost))
                # keep top-K by partial cost
            if args.beam_width is not None and args.beam_width > 0:
                new_beam.sort(key=lambda t: t[1])
                beam = new_beam[:args.beam_width]
            else:
                beam = new_beam

        for chosen_map, _ in beam:
            if getattr(args, "abort_flag", None) and args.abort_flag.get("hit"):
                break
            if len(chosen_map) != len(class_keys):
                continue
            ordered = [chosen_map[k] for k in class_keys]
            coords = [c['xyz'] for c in ordered]
            c = bbox_center(coords)
            if max(distc(p, c) for p in coords) > args.radius:
                continue
            if min(distc(p, c) for p in coords) < args.min_radius:
                continue
            if args.stage0_centroid_rmsd is not None and Tcdist is not None:
                cd = [distc(p, c) for p in coords]
                if rmsd_over_pairs(cd, Tcdist) > args.stage0_centroid_rmsd:
                    continue
            dist_list = tuple(
                distc(chosen_map[a]['xyz'], chosen_map[b]['xyz'])
                for a, b in pairs
            )
            if min(dist_list) < args.min_pair or max(dist_list) > args.max_pair:
                continue
            if args.stage1_rmsd_tol is not None:
                if kabsch_rmsd(coords, Tpoints) > args.stage1_rmsd_tol:
                    continue
            if args.use_weighted:
                num = 0.0
                den = 0.0
                for (p, dc) in zip(pairs, dist_list):
                    aa1 = mapping[p[0]]
                    aa2 = mapping[p[1]]
                    w = WEIGHTS.get((aa1, aa2), WEIGHTS.get((aa2, aa1), 1.0))
                    num += w * (dc - Tdist[p])**2
                    den += w
                score = math.sqrt(num/den) if den > 0 else float('inf')
            else:
                score = math.sqrt(
                    sum((dc - Tdist[p])**2 for p, dc in zip(pairs, dist_list))
                    / len(dist_list)
                )
            if getattr(args, "cluster_groups", None):
                coords_by_key = {ck: chosen_map[ck]['xyz'] for ck in class_keys}
                score = cluster_score_from_precomputed(
                    coords_by_key,
                    cluster_groups,
                    Tcluster_pair_dists,
                    Tcluster_internal_dists,
                    args.cluster_weight_centroid,
                    args.cluster_weight_internal,
                )
            elif args.blend_centroid:
                if args.centroid_peptide:
                    mj = getattr(args, "mj_matrix", None)
                    if mj is None and _G:
                        mj = _G.get("mj_matrix")
                    if mj is None:
                        raise RuntimeError("centroid_peptide enabled but MJ matrix not loaded")
                    motif_aas = [r["aa"] for r in ordered]
                    tmpl_aas = [mapping[ck] for ck, _ in classes]
                    cd_rmsd = centroid_peptide_rmsd(
                        [r["xyz"] for r in ordered],
                        motif_aas,
                        [tmpl_base[mapping[ck]] for ck, _ in classes],
                        tmpl_aas,
                        args,
                        mj,
                    )
                else:
                    # centroid-distance RMSD across positions (alignment-invariant)
                    cd_rmsd = centroid_dist_rmsd(coords, Tpoints)
                score = args.weight_pair * score + args.weight_centroid * cd_rmsd
            resid_set = set(r['res_id'] for r in ordered)
            if require_residues:
                ok_req = True
                for aa_req, resseq_req in require_residues:
                    if not any(r["aa"] == aa_req and r["resseq"] == resseq_req for r in ordered):
                        ok_req = False
                        break
                if not ok_req:
                    continue
            if require_template:
                ok_req = True
                ordered_map = {ck: r for ck, r in zip(class_keys, ordered)}
                inv_map = {label: ck for ck, label in mapping.items()}
                for label, resseq_req in require_template.items():
                    if label not in inv_map:
                        ok_req = False
                        break
                    r = ordered_map[inv_map[label]]
                    if r["aa"] != label or r["resseq"] != resseq_req:
                        ok_req = False
                        break
                if not ok_req:
                    continue
            payload = (score, e, uid, resid_set, dist_list, *ordered)
            score_key = round(score, 6)
            key = (-score_key, len(resid_set))
            if len(heap) < args.per_protein_cap:
                heapq.heappush(heap, (key, payload))
            else:
                if key > heap[0][0]:
                    heapq.heapreplace(heap, (key, payload))
            # ensure max-length candidate is kept when score is 0
            if score_key == 0.0:
                best0 = getattr(args, "_best_zero", {})
                prev = best0.get(e)
                if prev is None or len(resid_set) > prev[0]:
                    best0[e] = (len(resid_set), payload)
                    args._best_zero = best0

        # inject max-length zero-score candidate per entry if missing
        best0 = getattr(args, "_best_zero", {})
        if e in best0:
            _, payload0 = best0[e]
            if not any(payload0 is p for _, p in heap):
                heapq.heappush(heap, ((0.0, len(payload0[3])), payload0))
        for _, payload in heap:
            candidates.append(payload)
        if args.stop_after_first_non_template and e != args.template_entry and len(candidates) > cand_before:
            candidates_raw = list(candidates)
            return candidates, candidates_raw, classes, mapping, Tdist

    candidates.sort(key=lambda t: t[0])
    candidates_raw = list(candidates)

    # Optional MJ normalization across pockets (peptide-contact MJ on motif residues)
    if args.mj_norm:
        mj = getattr(args, "mj_matrix", None)
        if mj is None:
            mj = load_mj_matrix()
        mj_vals = []
        mj_meta = []
        for payload in candidates_raw:
            score0, e, uid, resid_set, dist_list, *ordered = payload
            motif_points = [r["xyz"] for r in ordered]
            motif_aas = [r["aa"] for r in ordered]
            mj_sum, n_contacts = peptide_contact_mj(motif_points, motif_aas, args, mj)
            mj_metric = (-mj_sum / n_contacts) if n_contacts > 0 else 0.0
            mj_vals.append(mj_metric)
            mj_meta.append((mj_sum, n_contacts, mj_metric))

        mean = sum(mj_vals) / len(mj_vals) if mj_vals else 0.0
        var = sum((x - mean) ** 2 for x in mj_vals) / len(mj_vals) if mj_vals else 0.0
        std = math.sqrt(var) if var > 0 else 1.0

        rescored = []
        for payload, meta in zip(candidates_raw, mj_meta):
            score0, e, uid, resid_set, dist_list, *ordered = payload
            mj_metric = meta[2]
            mj_norm = (mj_metric - mean) / std
            score = score0 + args.weight_mj * mj_norm
            rescored.append((score, e, uid, resid_set, dist_list, *ordered))
        rescored.sort(key=lambda t: t[0])
        candidates_raw = rescored
    return candidates, candidates_raw, classes, mapping, Tdist


def pair_list(keys: list[str]) -> list[tuple[str,str]]:
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            pairs.append((keys[i], keys[j]))
    return pairs


def centroid(coords: list[tuple[float,float,float]]) -> tuple[float,float,float]:
    n = len(coords)
    return (
        sum(c[0] for c in coords)/n,
        sum(c[1] for c in coords)/n,
        sum(c[2] for c in coords)/n,
    )

def bbox_center(coords: list[tuple[float,float,float]]) -> tuple[float,float,float]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0)


def max_centroid_radius(coords: list[tuple[float,float,float]]) -> float:
    # Use bounding-box center for radius derived from template points.
    c = bbox_center(coords)
    return max(distc(p, c) for p in coords)


def build_spatial_grid(residues: list[dict], cell_size: float):
    grid = defaultdict(list)
    if cell_size <= 0:
        return grid
    inv = 1.0 / cell_size
    for r in residues:
        x, y, z = r['xyz']
        ix = math.floor(x * inv)
        iy = math.floor(y * inv)
        iz = math.floor(z * inv)
        grid[(ix, iy, iz)].append(r)
    return grid


def neighbors_from_grid(grid, r, cell_size: float):
    inv = 1.0 / cell_size
    x, y, z = r['xyz']
    ix = math.floor(x * inv)
    iy = math.floor(y * inv)
    iz = math.floor(z * inv)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = (ix + dx, iy + dy, iz + dz)
                if cell in grid:
                    yield from grid[cell]


def jaccard(a, b) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def kabsch_rmsd(P, Q) -> float:
    # Legacy name kept for CLI/backward compatibility.
    # Pipeline now uses rigid-body-invariant pairwise-distance RMSD.
    if len(P) != len(Q):
        return float("inf")
    n = len(P)
    if n < 2:
        return 0.0
    d1 = []
    d2 = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            d1.append(distc(P[i], P[j]))
            d2.append(distc(Q[i], Q[j]))
    return rmsd_over_pairs(d1, d2)


def _mj_matrix_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    preferred = root / "refs" / "mj_matrix.csv"
    if preferred.exists():
        return preferred
    # Backward-compatible fallback for older layouts.
    return root / "mj_matrix.csv"


def load_mj_matrix() -> dict[str, dict[str, float]]:
    path = _mj_matrix_path()
    if not path.exists():
        raise FileNotFoundError(f"MJ matrix not found at {path}")
    mj = {}
    with path.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip().replace('\ufeff', '') for h in header]
        aas = header[1:]
        for row in reader:
            aa = row[0].strip()
            vals = row[1:]
            mj[aa] = {b: float(v) for b, v in zip(aas, vals)}
    return mj


def mj_score(mj: dict[str, dict[str, float]], a: str, b: str) -> float:
    if a in mj and b in mj[a]:
        return mj[a][b]
    if b in mj and a in mj[b]:
        return mj[b][a]
    return 0.0


def distance_weighted_mj(raw_mj: float, d: float) -> float:
    # Closer contacts contribute more strongly by default.
    w = math.exp(-0.25 * d)
    return raw_mj * w


def clash_penalty(d: float, radius: float, weight: float) -> float:
    if d >= radius:
        return 0.0
    return weight * (radius - d) ** 2


_PEPTIDE_SC_OFFSET = {
    'G': 0.0,
    'A': 1.1,
    'S': 1.2,
    'C': 1.3,
    'P': 1.3,
    'D': 1.6,
    'N': 1.6,
    'T': 1.6,
    'E': 2.0,
    'Q': 2.0,
    'H': 2.0,
    'V': 2.0,
    'I': 2.2,
    'L': 2.2,
    'M': 2.2,
    'K': 2.6,
    'R': 2.8,
    'F': 2.7,
    'Y': 2.8,
    'W': 3.0,
}


def peptide_sidechain_pseudoatoms(helix_points, peptide_seq):
    """
    Build one pseudo sidechain point per peptide residue.
    Offsets are placed radially outward from the helix axis.
    """
    import numpy as np
    pts = [np.array(p, dtype=float) for p in helix_points]
    if not pts:
        return []
    axis = np.array(pca_axis(helix_points), dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = np.mean(np.stack(pts, axis=0), axis=0)

    out = []
    for i, p in enumerate(pts):
        r = p - center
        r_perp = r - np.dot(r, axis) * axis
        if np.linalg.norm(r_perp) < 1e-8:
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(tmp, axis))) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            r_perp = np.cross(axis, tmp)
        r_hat = r_perp / (np.linalg.norm(r_perp) + 1e-12)
        aa = peptide_seq[i] if i < len(peptide_seq) else 'A'
        offset = _PEPTIDE_SC_OFFSET.get(aa, 2.0)
        q = p + offset * r_hat
        out.append((float(q[0]), float(q[1]), float(q[2])))
    return out


def parse_pose_strict_protein_points(spec: str) -> set[str]:
    """
    Parse protein proxy atom modes for strict pose clashes.
    Accepted tokens: ca, sc
    """
    s = (spec or "").strip().lower()
    if not s:
        return {"ca", "sc"}
    out = set()
    for tok in [t.strip() for t in s.split(",") if t.strip()]:
        if tok not in {"ca", "sc"}:
            raise RuntimeError(
                f"Invalid --pose-strict-protein-points token: {tok} (allowed: ca,sc)"
            )
        out.add(tok)
    if not out:
        raise RuntimeError(
            "--pose-strict-protein-points must include at least one of ca,sc"
        )
    return out


def build_pose_strict_protein_xyz(rows, modes: set[str]):
    """
    Build protein proxy coordinates used for strict clash screening.
    """
    pts = []
    use_ca = "ca" in modes
    use_sc = "sc" in modes
    for row in rows:
        if use_ca:
            pts.append((float(row["ca_x"]), float(row["ca_y"]), float(row["ca_z"])))
        if use_sc:
            pts.append((float(row["sc_x"]), float(row["sc_y"]), float(row["sc_z"])))
    return pts


def pose_has_strict_clash(helix_points, sidechain_pts, protein_xyz, min_dist: float) -> bool:
    """
    Strict clash gate:
    - peptide CA proxies (helix points) + peptide sidechain proxies
    - versus selected protein proxies (CA/SC).
    """
    if min_dist <= 0:
        return False
    if protein_xyz is None:
        return False
    pep_pts = list(helix_points) + list(sidechain_pts)
    if not pep_pts:
        return False
    import numpy as np

    P = np.asarray(pep_pts, dtype=float)
    Q = protein_xyz
    if P.size == 0 or Q.size == 0:
        return False
    d2 = np.sum((P[:, None, :] - Q[None, :, :]) ** 2, axis=2)
    return float(np.min(d2)) < (min_dist * min_dist)


def parse_peptide_capacity_map(spec: str) -> dict[str, int]:
    """
    Parse residue capacity map like "W:2,F:2,Y:2".
    Returns dict of single-letter AA -> integer capacity (>=1).
    """
    cap = {}
    if not spec:
        return cap
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    for part in parts:
        if ':' not in part:
            raise RuntimeError(f"Invalid --peptide-capacity-map token: {part}")
        aa, val = part.split(':', 1)
        aa = aa.strip().upper()
        if len(aa) != 1 or not aa.isalpha():
            raise RuntimeError(f"Invalid residue code in --peptide-capacity-map: {aa}")
        try:
            n = int(val.strip())
        except ValueError:
            raise RuntimeError(f"Invalid capacity in --peptide-capacity-map: {part}")
        if n < 1:
            raise RuntimeError(f"Capacity must be >=1 in --peptide-capacity-map: {part}")
        cap[aa] = n
    return cap


def peptide_position_capacities(peptide_seq: str, capacity_map: dict[str, int] | None) -> dict[int, int]:
    pep = list(peptide_seq or "")
    cap = {i: 1 for i in range(len(pep))}
    if capacity_map:
        for i, aa in enumerate(pep):
            if aa in capacity_map:
                cap[i] = capacity_map[aa]
    return cap


def pca_axis(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    import numpy as np
    P = np.array(points, dtype=float)
    P = P - P.mean(axis=0)
    cov = P.T @ P
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return (float(axis[0]), float(axis[1]), float(axis[2]))


def helix_positions(center, axis, n, rise, radius, rot_deg):
    import numpy as np
    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(tmp, axis))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    mid = (n - 1) / 2.0
    positions = []
    for i in range(n):
        z = (i - mid) * rise
        theta = math.radians(rot_deg * (i - mid))
        pos = np.array(center) + axis * z + radius * (math.cos(theta) * u + math.sin(theta) * v)
        positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
    return positions


def _rotate_axis(axis, tilt_deg, azimuth_deg):
    import numpy as np
    a = np.array(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(tmp, a))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, tmp)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    phi = math.radians(azimuth_deg)
    tilt_vec = math.cos(phi) * u + math.sin(phi) * v
    theta = math.radians(tilt_deg)
    new_a = math.cos(theta) * a + math.sin(theta) * tilt_vec
    new_a = new_a / (np.linalg.norm(new_a) + 1e-12)
    return (float(new_a[0]), float(new_a[1]), float(new_a[2]))


def _build_helix_basis(axis):
    import numpy as np
    a = np.array(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(tmp, a))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, tmp)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return a, u, v


def _bend_axis(axis, bend_deg, azimuth_deg):
    import numpy as np
    a, u, v = _build_helix_basis(axis)
    phi = math.radians(azimuth_deg)
    bend_vec = math.cos(phi) * u + math.sin(phi) * v
    theta = math.radians(bend_deg)
    new_a = math.cos(theta) * a + math.sin(theta) * bend_vec
    new_a = new_a / (np.linalg.norm(new_a) + 1e-12)
    return (float(new_a[0]), float(new_a[1]), float(new_a[2]))


def kinked_helix_positions(center, axis, n, rise, radius, rot_deg, hinge_idx, bend_deg, bend_azimuth_deg):
    """
    Build a peptide helix with a single hinge (kink) after hinge_idx.
    Positions [0..hinge_idx] follow the base axis; [hinge_idx+1..] follow bent axis.
    """
    if n <= 1:
        return helix_positions(center, axis, n, rise, radius, rot_deg)
    hinge_idx = max(0, min(n - 2, int(hinge_idx)))

    base = helix_positions(center, axis, n, rise, radius, rot_deg)
    # Anchor second segment at the next residue after hinge to keep continuity.
    pivot = base[hinge_idx + 1]
    axis2 = _bend_axis(axis, bend_deg, bend_azimuth_deg)
    seg2 = helix_positions(pivot, axis2, n - (hinge_idx + 1), rise, radius, rot_deg)

    out = list(base[:hinge_idx + 1]) + list(seg2)
    return out


def peptide_helix_ensemble(center, axis, n, args, rng):
    """
    Build an ensemble of peptide helices around the default ideal helix parameters.
    The first member is always the unperturbed helix for backward compatibility.
    """
    ens_n = max(1, int(getattr(args, "peptide_ensemble", 1)))
    out = []
    out.append(
        helix_positions(center, axis, n, args.peptide_rise, args.peptide_radius, args.peptide_rot)
    )
    for _ in range(ens_n - 1):
        rise = args.peptide_rise + rng.uniform(-args.peptide_rise_jitter, args.peptide_rise_jitter)
        radius = args.peptide_radius + rng.uniform(-args.peptide_radius_jitter, args.peptide_radius_jitter)
        rot = args.peptide_rot + rng.uniform(-args.peptide_rot_jitter, args.peptide_rot_jitter)
        tilt = rng.uniform(-args.peptide_tilt_jitter_deg, args.peptide_tilt_jitter_deg)
        az = rng.uniform(0.0, 360.0)
        shifted_center = (
            center[0] + rng.uniform(-args.peptide_center_shift, args.peptide_center_shift),
            center[1] + rng.uniform(-args.peptide_center_shift, args.peptide_center_shift),
            center[2] + rng.uniform(-args.peptide_center_shift, args.peptide_center_shift),
        )
        axis_pert = _rotate_axis(axis, tilt, az)
        if getattr(args, "peptide_kink_enable", False) and n >= 3:
            hinge_lo = max(0, int(getattr(args, "peptide_kink_min_index", 2)))
            hinge_hi = min(n - 2, int(getattr(args, "peptide_kink_max_index", n - 3)))
            if hinge_hi < hinge_lo:
                hinge_lo, hinge_hi = 1, n - 2
            hinge_idx = rng.randint(hinge_lo, hinge_hi)
            bend_deg = rng.uniform(0.0, args.peptide_kink_max_bend_deg)
            bend_az = rng.uniform(0.0, 360.0)
            out.append(
                kinked_helix_positions(
                    shifted_center,
                    axis_pert,
                    n,
                    rise,
                    radius,
                    rot,
                    hinge_idx,
                    bend_deg,
                    bend_az,
                )
            )
        else:
            out.append(helix_positions(shifted_center, axis_pert, n, rise, radius, rot))
    return out


def centroid_peptide_rmsd(motif_points, motif_aas, tmpl_points, tmpl_aas, args, mj):
    # build helix positions for candidate and template from their motif PCA axes
    c_center = centroid(motif_points)
    t_center = centroid(tmpl_points)
    c_axis = pca_axis(motif_points)
    t_axis = pca_axis(tmpl_points)
    pep = list(args.peptide_seq)
    seed_base = getattr(args, "peptide_ensemble_seed", 123)
    rng_c = random.Random(f"{seed_base}:c:{len(motif_points)}")
    rng_t = random.Random(f"{seed_base}:t:{len(tmpl_points)}")
    c_helix_ens = peptide_helix_ensemble(c_center, c_axis, len(pep), args, rng_c)
    t_helix_ens = peptide_helix_ensemble(t_center, t_axis, len(pep), args, rng_t)

    def best_contact_dists(points, aas, helix_positions):
        # One-sided capacities on peptide positions (default map: W:2).
        cap = peptide_position_capacities(args.peptide_seq, getattr(args, "peptide_capacity_map", None))
        hard_min = float(getattr(args, "peptide_hard_clash_min_dist", 2.0))
        sidechain_pts = peptide_sidechain_pseudoatoms(helix_positions, pep)
        if hard_min > 0:
            # Early helix-level prune: reject this helix before assignment work.
            for hp in sidechain_pts:
                for p in points:
                    if distc(p, hp) < hard_min:
                        return None

        # candidate positions per residue (within cutoff), else nearest
        candidates = []
        for p, aa in zip(points, aas):
            opts = []
            for i, hp in enumerate(sidechain_pts):
                d = distc(p, hp)
                if hard_min > 0 and d < hard_min:
                    continue
                if d <= args.peptide_cutoff:
                    s = distance_weighted_mj(mj_score(mj, pep[i], aa), d) + clash_penalty(
                        d, args.peptide_clash_radius, args.peptide_clash_weight
                    )
                    opts.append((i, s, d))
            if not opts:
                # fallback to nearest non-clashing if none within cutoff
                allowed = [
                    i for i, hp in enumerate(sidechain_pts)
                    if (hard_min <= 0 or distc(p, hp) >= hard_min)
                ]
                if not allowed:
                    return None
                best_i = min(allowed, key=lambda i: distc(p, sidechain_pts[i]))
                d = distc(p, sidechain_pts[best_i])
                s = distance_weighted_mj(mj_score(mj, pep[best_i], aa), d) + clash_penalty(
                    d, args.peptide_clash_radius, args.peptide_clash_weight
                )
                opts.append((best_i, s, d))
            candidates.append(opts)

        best = None
        best_dists = None
        used = {i: 0 for i in range(len(pep))}

        def rec(idx, total_s, total_d, dists):
            nonlocal best, best_dists
            if idx == len(candidates):
                key = (total_s, total_d)
                if best is None or key < best:
                    best = key
                    best_dists = list(dists)
                return
            for i, s, d in candidates[idx]:
                if used[i] >= cap[i]:
                    continue
                used[i] += 1
                dists.append(d)
                rec(idx + 1, total_s + s, total_d + d, dists)
                dists.pop()
                used[i] -= 1

        rec(0, 0.0, 0.0, [])
        return best_dists

    def best_over_ensemble(points, aas, helix_ensemble):
        best = None
        best_d = None
        for hel in helix_ensemble:
            dists = best_contact_dists(points, aas, hel)
            if not dists:
                continue
            key = sum(dists)
            if best is None or key < best:
                best = key
                best_d = dists
        if best_d is None:
            return None
        return best_d

    c_d = best_over_ensemble(motif_points, motif_aas, c_helix_ens)
    t_d = best_over_ensemble(tmpl_points, tmpl_aas, t_helix_ens)
    if c_d is None or t_d is None:
        # Infeasible under hard-clash constraints.
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c_d, t_d)) / len(c_d))


def peptide_contact_mj(motif_points, motif_aas, args, mj):
    # peptide helix based on motif PCA axis and centroid
    center = centroid(motif_points)
    axis = pca_axis(motif_points)
    pep = list(args.peptide_seq)
    seed_base = getattr(args, "peptide_ensemble_seed", 123)
    rng = random.Random(f"{seed_base}:mj:{len(motif_points)}")
    helix_ens = peptide_helix_ensemble(center, axis, len(pep), args, rng)

    # One-sided capacities on peptide positions (default map: W:2).
    cap = peptide_position_capacities(args.peptide_seq, getattr(args, "peptide_capacity_map", None))

    def score_one_helix(helix):
        sidechain_pts = peptide_sidechain_pseudoatoms(helix, pep)
        hard_min = float(getattr(args, "peptide_hard_clash_min_dist", 2.0))
        if hard_min > 0:
            # Early helix-level prune: reject this helix before assignment work.
            for hp in sidechain_pts:
                for p in motif_points:
                    if distc(p, hp) < hard_min:
                        return 0.0, 0
        candidates = []
        for p, aa in zip(motif_points, motif_aas):
            opts = []
            for i, hp in enumerate(sidechain_pts):
                d = distc(p, hp)
                if hard_min > 0 and d < hard_min:
                    continue
                if d <= args.peptide_cutoff:
                    s = distance_weighted_mj(mj_score(mj, pep[i], aa), d) + clash_penalty(
                        d, args.peptide_clash_radius, args.peptide_clash_weight
                    )
                    opts.append((i, s, d))
            if not opts:
                # fallback to nearest non-clashing if none within cutoff
                allowed = [
                    i for i, hp in enumerate(sidechain_pts)
                    if (hard_min <= 0 or distc(p, hp) >= hard_min)
                ]
                if not allowed:
                    return 0.0, 0
                best_i = min(allowed, key=lambda i: distc(p, sidechain_pts[i]))
                d = distc(p, sidechain_pts[best_i])
                s = distance_weighted_mj(mj_score(mj, pep[best_i], aa), d) + clash_penalty(
                    d, args.peptide_clash_radius, args.peptide_clash_weight
                )
                opts.append((best_i, s, d))
            candidates.append(opts)

        best = None
        best_sum = None
        best_n = None
        used = {i: 0 for i in range(len(pep))}

        def rec(idx, total_s, n_contacts):
            nonlocal best, best_sum, best_n
            if idx == len(candidates):
                key = (total_s, n_contacts)
                if best is None or key < best:
                    best = key
                    best_sum = total_s
                    best_n = n_contacts
                return
            for i, s, _d in candidates[idx]:
                if used[i] >= cap[i]:
                    continue
                used[i] += 1
                rec(idx + 1, total_s + s, n_contacts + 1)
                used[i] -= 1

        rec(0, 0.0, 0)
        if best_sum is None or best_n == 0:
            return 0.0, 0
        return best_sum, best_n

    best = None
    for helix in helix_ens:
        ssum, sn = score_one_helix(helix)
        key = (ssum, sn)
        if best is None or key < best[0]:
            best = (key, ssum, sn)
    return best[1], best[2]


def parse_pose_anchor_spec(spec: str):
    """
    Parse optional anchor constraints in the form:
      "4:V247,M362;3:D173"
    Returns list of (pep_index_1based, [target_labels]).
    """
    out = []
    s = (spec or "").strip()
    if not s:
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise RuntimeError(f"Invalid --pose-anchor token: {part}")
        pi_s, labs_s = part.split(":", 1)
        pep_idx = int(pi_s.strip())
        labs = [x.strip() for x in labs_s.split(",") if x.strip()]
        if not labs:
            raise RuntimeError(f"Invalid --pose-anchor token (no labels): {part}")
        out.append((pep_idx, labs))
    return out


def _greedy_pose_contacts(
    motif_points,
    motif_labels,
    motif_aas,
    sidechain_pts,
    pep,
    args,
    mj,
    helix_points=None,
):
    """
    Fast per-pose contact assignment:
    - hard-clash helix prefilter
    - collect motif<->peptide candidates within cutoff
    - greedy bipartite selection by (score, distance) with peptide capacities
    """
    strict_min = float(getattr(args, "pose_strict_clash_min_dist", 2.0))
    if getattr(args, "pose_strict_clash", False):
        if pose_has_strict_clash(
            helix_points or [],
            sidechain_pts,
            getattr(args, "pose_strict_protein_xyz", None),
            strict_min,
        ):
            return None

    hard_min = float(getattr(args, "peptide_hard_clash_min_dist", 2.0))
    if hard_min > 0:
        for hp in sidechain_pts:
            for p in motif_points:
                if distc(p, hp) < hard_min:
                    return None

    # Optional anchored residue constraints.
    anchors = getattr(args, "pose_anchor_parsed", [])
    if anchors:
        anchor_max = float(getattr(args, "pose_anchor_max_dist", 6.5))
        anchor_balance = float(getattr(args, "pose_anchor_balance", 2.5))
        pos_by_label = {lab: p for lab, p in zip(motif_labels, motif_points)}
        for pep_idx_1, labels in anchors:
            if pep_idx_1 < 1 or pep_idx_1 > len(sidechain_pts):
                return None
            hp = sidechain_pts[pep_idx_1 - 1]
            dvals = []
            for lab in labels:
                p = pos_by_label.get(lab)
                if p is None:
                    return None
                d = distc(hp, p)
                dvals.append(d)
                if d > anchor_max:
                    return None
            if len(dvals) >= 2 and (max(dvals) - min(dvals)) > anchor_balance:
                return None

    cap = peptide_position_capacities(args.peptide_seq, getattr(args, "peptide_capacity_map", None))
    fav_thr = float(getattr(args, "pose_favorable_threshold", 0.0))
    req_pep = set(getattr(args, "pose_require_pep_idx_set", set()))
    edges = []
    for mi, (p, aa, lab) in enumerate(zip(motif_points, motif_aas, motif_labels)):
        for i, hp in enumerate(sidechain_pts):
            d = distc(p, hp)
            if hard_min > 0 and d < hard_min:
                continue
            if d > args.peptide_cutoff:
                continue
            raw_mj = mj_score(mj, pep[i], aa)
            s = distance_weighted_mj(raw_mj, d) + clash_penalty(
                d, args.peptide_clash_radius, args.peptide_clash_weight
            )
            edges.append((s, d, mi, i, lab, aa, pep[i], raw_mj))

    if not edges:
        return None
    edges.sort(key=lambda t: (t[0], t[1]))
    used_motif = set()
    used_pep = {i: 0 for i in range(len(pep))}
    contacts = []
    for s, d, mi, i, lab, aa, pep_aa, raw_mj in edges:
        if mi in used_motif:
            continue
        if used_pep[i] >= cap[i]:
            continue
        used_motif.add(mi)
        used_pep[i] += 1
        contacts.append((lab, aa, i + 1, pep_aa, d, s, raw_mj))

    if len(contacts) < int(getattr(args, "pose_scan_min_contacts", 1)):
        return None
    req_res = set(getattr(args, "pose_require_pocket_residues_set", set()))
    if req_res:
        used_res = {c[0] for c in contacts}
        if not req_res.issubset(used_res):
            return None
    used_pep_idx = {c[2] for c in contacts}
    if req_pep and not req_pep.issubset(used_pep_idx):
        return None
    n_fav = sum(1 for c in contacts if c[6] < fav_thr)
    if n_fav < int(getattr(args, "pose_min_favorable", 0)):
        return None
    total_s = sum(c[5] for c in contacts)
    mean_d = sum(c[4] for c in contacts) / len(contacts)
    return total_s, len(contacts), mean_d, n_fav, contacts


def run_pose_scan_mode(args):
    if not args.peptide_seq:
        raise RuntimeError("--mode pose-scan requires --peptide-seq")
    if not args.only_entry:
        raise RuntimeError("--mode pose-scan requires --only-entry ENTRY")

    path = os.path.join(args.ecoli_batch, args.only_entry, "residues.tsv")
    if not os.path.exists(path):
        raise RuntimeError(f"Entry residues not found: {path}")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)

    if getattr(args, "pose_strict_clash", False):
        strict_pts = build_pose_strict_protein_xyz(rows, args.pose_strict_protein_points_set)
        if not strict_pts:
            raise RuntimeError(
                "Strict clash enabled but no protein proxy points were built."
            )
        import numpy as np

        args.pose_strict_protein_xyz = np.asarray(strict_pts, dtype=float)
    else:
        args.pose_strict_protein_xyz = None

    if args.template_residues:
        toks = [t.strip() for t in args.template_residues.split(",") if t.strip()]
    else:
        toks = [f"{aa}{rs}" for aa, rs in DNAN_POCKET_RESIDUES_DPO3B]
    wanted = set(toks)
    by_label = {}
    for r in rows:
        lab = f"{r['aa']}{r['resseq']}"
        if lab in wanted:
            by_label[lab] = r
    missing = [t for t in toks if t not in by_label]
    if missing:
        raise RuntimeError(f"Missing residues in {args.only_entry}: {', '.join(missing)}")

    motif_labels = list(toks)
    motif_points = [
        (float(by_label[lab]["sc_x"]), float(by_label[lab]["sc_y"]), float(by_label[lab]["sc_z"]))
        for lab in motif_labels
    ]
    motif_aas = [lab[0] for lab in motif_labels]

    center = centroid(motif_points)
    axis = pca_axis(motif_points)
    pep = list(args.peptide_seq)
    mj = load_mj_matrix()
    rng = random.Random(args.peptide_ensemble_seed)

    # Parse anchors once.
    try:
        args.pose_anchor_parsed = parse_pose_anchor_spec(args.pose_anchor)
    except Exception as e:
        raise RuntimeError(str(e))
    req = set()
    for tok in [t.strip() for t in (args.pose_require_pep_idx or "").split(",") if t.strip()]:
        req.add(int(tok))
    args.pose_require_pep_idx_set = req
    args.pose_require_pocket_residues_set = {
        t.strip() for t in (args.pose_require_pocket_residues or "").split(",") if t.strip()
    }

    top_n = max(1, int(args.pose_scan_top_n))
    all_hits = []
    contact_counts = {}
    residue_counts = {lab: 0 for lab in motif_labels}
    accepted = 0

    for _ in range(max(1, int(args.pose_scan_trials))):
        # random center in sphere around motif centroid
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            z = rng.uniform(-1, 1)
            if x * x + y * y + z * z <= 1:
                break
        rr = args.pose_scan_center_radius * (rng.random() ** (1 / 3))
        c = (center[0] + rr * x, center[1] + rr * y, center[2] + rr * z)
        helix_ens = peptide_helix_ensemble(c, axis, len(pep), args, rng)
        for hel in helix_ens:
            sidechain_pts = peptide_sidechain_pseudoatoms(hel, pep)
            ev = _greedy_pose_contacts(
                motif_points,
                motif_labels,
                motif_aas,
                sidechain_pts,
                pep,
                args,
                mj,
                helix_points=hel,
            )
            if ev is None:
                continue
            total_s, n_contacts, mean_d, n_fav, contacts = ev
            accepted += 1
            for lab, _aa, pep_idx, pep_aa, _d, _s, _raw in contacts:
                residue_counts[lab] = residue_counts.get(lab, 0) + 1
                k = f"{lab}<->pep{pep_idx}:{pep_aa}"
                contact_counts[k] = contact_counts.get(k, 0) + 1
            all_hits.append((total_s, -n_contacts, mean_d, -n_fav, contacts))

    all_hits.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    top_hits = all_hits[:top_n]

    out = args.out or "pose_scan.tsv"
    out_freq = out.replace(".tsv", "_contact_freq.tsv") if out.endswith(".tsv") else out + "_contact_freq.tsv"
    out_res = out.replace(".tsv", "_residue_freq.tsv") if out.endswith(".tsv") else out + "_residue_freq.tsv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "total_mj", "n_contacts", "n_favorable", "mean_distance", "contact"])
        rank = 1
        for total_s, neg_n, mean_d, neg_fav, contacts in top_hits:
            for lab, _aa, pep_idx, pep_aa, d, s, raw in sorted(contacts, key=lambda x: x[4]):
                w.writerow([rank, f"{total_s:.4f}", -neg_n, -neg_fav, f"{mean_d:.4f}",
                            f"{lab}<->pep{pep_idx}:{pep_aa};d={d:.3f};mj={s:.1f};raw={raw:.1f}"])
            rank += 1

    with open(out_freq, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["contact", "count", "fraction_accepted_poses"])
        denom = accepted if accepted > 0 else 1
        for k, c in sorted(contact_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([k, c, f"{c / denom:.4f}"])

    with open(out_res, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pocket_residue", "count", "fraction_accepted_poses"])
        denom = accepted if accepted > 0 else 1
        for lab, c in sorted(residue_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([lab, c, f"{c / denom:.4f}"])

    print(out)
    print(out_freq)
    print(out_res)
    print("accepted_poses", accepted)
    print("top_n", len(top_hits))


def site_match(args, resid_set_a, points_a, resid_set_b, points_b) -> bool:
    ok = False
    if args.jaccard_merge_threshold and args.jaccard_merge_threshold > 0:
        if jaccard(resid_set_a, resid_set_b) >= args.jaccard_merge_threshold:
            ok = True
    if not ok and args.align_merge_threshold and args.align_merge_threshold > 0:
        if kabsch_rmsd(points_a, points_b) <= args.align_merge_threshold:
            ok = True
    return ok


def build_tdist_from_tcoords(Tcoords: dict[str, tuple[float, float, float]], class_keys: list[str]) -> dict[tuple[str, str], float]:
    pairs = pair_list(class_keys)
    return {p: distc(Tcoords[p[0]], Tcoords[p[1]]) for p in pairs}


def permute_tdist(Tdist: dict[tuple[str, str], float], rng) -> dict[tuple[str, str], float]:
    pairs = list(Tdist.keys())
    vals = [Tdist[p] for p in pairs]
    rng.shuffle(vals)
    return {p: v for p, v in zip(pairs, vals)}


def jitter_tdist(Tdist: dict[tuple[str, str], float], sigma: float, rng) -> dict[tuple[str, str], float]:
    return {p: (v + rng.gauss(0, sigma)) for p, v in Tdist.items()}


def centroid_dist_rmsd(points: list[tuple[float, float, float]], tmpl_points: list[tuple[float, float, float]]) -> float:
    c = bbox_center(points)
    cd = [distc(p, c) for p in points]
    tc = bbox_center(tmpl_points)
    tcd = [distc(p, tc) for p in tmpl_points]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(cd, tcd)) / len(cd))


def cluster_score_from_precomputed(
    coords_by_key: dict[str, tuple[float, float, float]],
    cluster_groups: list[list[str]],
    Tcluster_pair_dists: list[float],
    Tcluster_internal_dists: list[float],
    weight_centroid: float,
    weight_internal: float,
) -> float:
    Ccentroids = []
    for grp in cluster_groups:
        pts = [coords_by_key[ck] for ck in grp]
        Ccentroids.append(centroid(pts))
    Cpair = []
    for i in range(len(Ccentroids)):
        for j in range(i + 1, len(Ccentroids)):
            Cpair.append(distc(Ccentroids[i], Ccentroids[j]))
    Cinternal = []
    for grp in cluster_groups:
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                Cinternal.append(distc(coords_by_key[grp[i]], coords_by_key[grp[j]]))
    rmsd_centroid = rmsd_over_pairs(Cpair, Tcluster_pair_dists) if Tcluster_pair_dists else 0.0
    rmsd_internal = rmsd_over_pairs(Cinternal, Tcluster_internal_dists) if Tcluster_internal_dists else 0.0
    return weight_centroid * rmsd_centroid + weight_internal * rmsd_internal


def _assign_site(
    sites,
    per_uid_site_idx,
    args,
    uid,
    entry,
    score,
    resid_set,
    points,
):
    for si in per_uid_site_idx[uid]:
        s = sites[si]
        if site_match(args, resid_set, points, s['resid_set'], s['points']):
            s['count'] += 1
            return True, si
    sites.append({
        'uid': uid,
        'entry': entry,
        'score': score,
        'resid_set': resid_set,
        'points': points,
        'count': 1,
    })
    per_uid_site_idx[uid].append(len(sites) - 1)
    return False, len(sites) - 1

def compute_top_sites_for_m(candidates_raw, args, M):
    sites = []  # list of dict: {uid, entry, score, resid_set, points, count}
    per_uid_site_idx = defaultdict(list)  # uid -> indices into sites
    for score, entry, uid, resid_set, dist_list, *rest in candidates_raw:
        points = [r['xyz'] for r in rest]
        assigned, _ = _assign_site(
            sites, per_uid_site_idx, args, uid, entry, score, resid_set, points
        )
        if not assigned and len(sites) >= M:
            break
    return sites


def compute_site_metrics_for_ms(candidates_raw, args, Ms: list[int], pulldown: set):
    Ms_sorted = sorted(Ms)
    target_idx = 0
    sites = []  # list of dict: {uid, entry, score, resid_set, points, count}
    per_uid_site_idx = defaultdict(list)  # uid -> indices into sites
    P_set = set()
    P_pulldown = 0
    sum_counts = 0
    max_deg = 0
    results = {}

    for score, entry, uid, resid_set, dist_list, *rest in candidates_raw:
        points = [r['xyz'] for r in rest]
        assigned, si = _assign_site(
            sites, per_uid_site_idx, args, uid, entry, score, resid_set, points
        )
        sum_counts += 1
        if assigned:
            s = sites[si]
            if s['count'] > max_deg:
                max_deg = s['count']
        else:
            if uid not in P_set:
                P_set.add(uid)
                if uid in pulldown:
                    P_pulldown += 1
            if max_deg < 1:
                max_deg = 1
            while target_idx < len(Ms_sorted) and len(sites) >= Ms_sorted[target_idx]:
                mean_deg = sum_counts / len(sites) if sites else 0.0
                results[Ms_sorted[target_idx]] = {
                    'P': len(P_set),
                    'P_pulldown': P_pulldown,
                    'mean_deg': mean_deg,
                    'max_deg': max_deg,
                }
                target_idx += 1
            if target_idx >= len(Ms_sorted):
                break

    for m in Ms_sorted:
        if m not in results:
            mean_deg = sum_counts / len(sites) if sites else 0.0
            results[m] = {
                'P': len(P_set),
                'P_pulldown': P_pulldown,
                'mean_deg': mean_deg,
                'max_deg': max_deg,
            }
    return results


def merge_pockets_within_uid(candidates, args):
    """
    Merge pockets within each protein using union-find:
    connect if Jaccard >= threshold OR alignment RMSD <= threshold.
    Returns filtered candidate list.
    """
    if not ((args.jaccard_merge_threshold and args.jaccard_merge_threshold > 0) or
            (args.align_merge_threshold and args.align_merge_threshold > 0)):
        return candidates

    # group candidates by protein
    per_uid = defaultdict(list)
    for idx, cand in enumerate(candidates):
        per_uid[cand[2]].append((idx, cand))

    selected = [False] * len(candidates)
    for uid, items in per_uid.items():
        if len(items) == 1:
            selected[items[0][0]] = True
            continue

        # union-find over indices within this protein
        local_indices = [i for i, _ in items]
        parent = {i: i for i in local_indices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # precompute residue sets and points
        resid_sets = {}
        points = {}
        for i, cand in items:
            resid_sets[i] = cand[3]
            points[i] = [(r['x'], r['y'], r['z']) for r in cand[5:]]

        for i_idx in range(len(local_indices)):
            i = local_indices[i_idx]
            for j_idx in range(i_idx + 1, len(local_indices)):
                j = local_indices[j_idx]
                ok = False
                if args.jaccard_merge_threshold and args.jaccard_merge_threshold > 0:
                    if jaccard(resid_sets[i], resid_sets[j]) >= args.jaccard_merge_threshold:
                        ok = True
                if not ok and args.align_merge_threshold and args.align_merge_threshold > 0:
                    if len(points[i]) == len(points[j]):
                        if kabsch_rmsd(points[i], points[j]) <= args.align_merge_threshold:
                            ok = True
                if ok:
                    union(i, j)

        # pick best score per component
        best = {}  # root -> (score, idx)
        for i, cand in items:
            r = find(i)
            if r not in best:
                best[r] = (cand[0], i)
            else:
                best_score, best_i = best[r]
                cand_score = cand[0]
                cand_key = round(cand_score, 6)
                best_key = round(best_score, 6)
                if cand_key < best_key:
                    best[r] = (cand_score, i)
                elif cand_key == best_key:
                    # Prefer larger residue count on (near) tie.
                    if len(resid_sets[i]) > len(resid_sets[best_i]):
                        best[r] = (cand_score, i)
        for _, best_i in best.values():
            selected[best_i] = True

    return [cand for i, cand in enumerate(candidates) if selected[i]]


PREFERRED_ENTRY_BY_UID = {
    "P0DM85": "RDCA_ECOLI",
    "P0AD27": "PBGA_ECOLI",
    "P39267": "RDCB_ECOLI",
}


def dedupe_candidates_by_uid(candidates):
    """
    Keep one candidate per UID.
    Primary key: lowest score.
    Ties: preferred canonical entry by UID, then lexicographic entry name.
    """
    best = {}
    for cand in candidates:
        score, entry, uid, resid_set, dist_list, *res = cand
        if uid not in best:
            best[uid] = cand
            continue
        prev = best[uid]
        pscore, pentry = prev[0], prev[1]
        skey = round(score, 6)
        pkey = round(pscore, 6)
        if skey < pkey:
            best[uid] = cand
            continue
        if skey > pkey:
            continue
        pref = PREFERRED_ENTRY_BY_UID.get(uid)
        if pref:
            if entry == pref and pentry != pref:
                best[uid] = cand
                continue
            if pentry == pref and entry != pref:
                continue
        if entry < pentry:
            best[uid] = cand

    out = list(best.values())
    out.sort(key=lambda t: t[0])
    return out


def cluster_seed_search(
    pattern: str,
    args,
    res_cache,
    entry_to_uid,
    tmpl_base,
    fixed_mapping: bool = False,
    tmpl_meta=None,
):
    """
    Greedy cluster-seed search:
    1) Find best match to a single cluster.
    2) Fix those residues and iteratively add best matches for remaining clusters,
       requiring no residue overlap and preferring proximity to existing clusters
       via centroid-pair RMSD to template cluster centroids.
    Stops when no valid cluster hit remains.
    """
    classes = parse_pattern(pattern)
    class_keys = [c for c, _ in classes]
    mapping = resolve_mapping(classes, tmpl_base, tmpl_meta, fixed_mapping)

    Tcoords = {ck: tmpl_base[mapping[ck]] for ck, _ in classes}
    cluster_groups = parse_cluster_groups(args.cluster_groups, class_keys)
    # Optional 1-based cluster indices that must be present in final match.
    required_clusters = set()
    if getattr(args, "require_clusters", None):
        for tok in str(args.require_clusters).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    lo = int(a.strip())
                    hi = int(b.strip())
                except ValueError:
                    continue
                if lo > hi:
                    lo, hi = hi, lo
                for idx1 in range(lo, hi + 1):
                    if 1 <= idx1 <= len(cluster_groups):
                        required_clusters.add(idx1 - 1)
            else:
                try:
                    idx1 = int(tok)
                except ValueError:
                    continue
                if 1 <= idx1 <= len(cluster_groups):
                    required_clusters.add(idx1 - 1)
    # Optional 1-based cluster indices to prioritize for seed selection.
    priority_seed = set()
    if getattr(args, "cluster_seed_priority", None):
        for tok in str(args.cluster_seed_priority).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idx1 = int(tok)
            except ValueError:
                continue
            if 1 <= idx1 <= len(cluster_groups):
                priority_seed.add(idx1 - 1)

    # Template-derived defaults for radius filters (align with build_candidates behavior).
    if args.radius is None or args.min_radius is None:
        template_coords = [Tcoords[k] for k in class_keys]
        if template_coords:
            tc = bbox_center(template_coords)
            dists = [distc(p, tc) for p in template_coords]
            mean_d = sum(dists) / len(dists)
            var_d = sum((x - mean_d) ** 2 for x in dists) / len(dists)
            std_d = math.sqrt(var_d)
            if args.radius is None:
                args.radius = mean_d + 2.0 * std_d
            if args.min_radius is None:
                args.min_radius = max(0.0, min(dists) - 1.0)
        else:
            if args.radius is None:
                args.radius = 8.0
            if args.min_radius is None:
                args.min_radius = 0.0

    # template cluster centroids and internal distances
    Tcluster_centroids = []
    Tcluster_internal = []
    for grp in cluster_groups:
        pts = [Tcoords[ck] for ck in grp]
        Tcluster_centroids.append(centroid(pts))
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                Tcluster_internal.append(distc(Tcoords[grp[i]], Tcoords[grp[j]]))

    # helper: internal RMSD for a cluster assignment
    def internal_rmsd(keys, chosen_map):
        Cinternal = []
        Tinternal = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                Cinternal.append(distc(chosen_map[a]['xyz'], chosen_map[b]['xyz']))
                Tinternal.append(distc(Tcoords[a], Tcoords[b]))
        return rmsd_over_pairs(Cinternal, Tinternal) if Cinternal else 0.0

    # helper: centroid RMSD between selected clusters and candidate cluster
    def centroid_pair_rmsd(selected_centroids, selected_tcentroids, cand_centroid, cand_tcentroid):
        Cpair = []
        Tpair = []
        for c, t in zip(selected_centroids, selected_tcentroids):
            Cpair.append(distc(c, cand_centroid))
            Tpair.append(distc(t, cand_tcentroid))
        return rmsd_over_pairs(Cpair, Tpair) if Cpair else 0.0

    # AA class helper for match counting
    class_map = {
        'aliphatic': set('VILM'),
        'basic': set('RKH'),
        'acidic': set('DE'),
        'aromatic': set('FWY'),
        'polar': set('STNQ'),
        'special': set('PG'),
    }
    def aa_class(a):
        for k, v in class_map.items():
            if a in v:
                return k
        return 'other'

    require_residues = getattr(args, "require_residues", None) or []
    require_template = getattr(args, "require_template", None) or {}
    mj = None
    if args.mj_norm:
        mj = getattr(args, "mj_matrix", None)
        if mj is None:
            mj = load_mj_matrix()

    candidates = []
    total_entries = len(res_cache)
    partial_clusters = (
        args.cluster_seed_min_residues is not None
        and args.cluster_seed_min_residues < len(class_keys)
    )
    for i_entry, (e, res_by_aa) in enumerate(res_cache.items(), start=1):
        if getattr(args, "abort_flag", None) and args.abort_flag.get("hit"):
            break
        if args.progress_every and i_entry % args.progress_every == 0:
            print(f"progress {i_entry}/{total_entries} entries", file=sys.stderr)

        uid = entry_to_uid.get(e)
        if not uid:
            continue

        # upfront radius prefilter on individual residues (bbox center of template)
        if args.min_radius is not None or args.radius is not None:
            tcenter = bbox_center([Tcoords[k] for k in class_keys])
            for aa_key in list(res_by_aa.keys()):
                filtered = [
                    r for r in res_by_aa[aa_key]
                    if (
                        args.radius is None
                        or distc(r['xyz'], tcenter) <= args.radius
                    )
                    and (args.min_radius is None or distc(r['xyz'], tcenter) >= args.min_radius)
                ]
                res_by_aa[aa_key] = filtered

        # build per-class residue lists
        byclass = defaultdict(list)
        for ck, aas in classes:
            for aa in aas:
                if aa in res_by_aa:
                    byclass[ck].extend(res_by_aa[aa])
        if args.pos_radius is not None and args.pos_radius > 0:
            for ck in class_keys:
                tpt = Tcoords[ck]
                byclass[ck] = [r for r in byclass[ck] if distc(r['xyz'], tpt) <= args.pos_radius]
        if args.template_first:
            for ck in class_keys:
                tpt = Tcoords[ck]
                byclass[ck].sort(key=lambda r: distc(r['xyz'], tpt))

        # precompute candidate lists per cluster (top-N by internal RMSD)
        cluster_candidates = []
        for grp in cluster_groups:
            grp_keys = list(grp)
            if (not partial_clusters) and any(len(byclass[k]) == 0 for k in grp_keys):
                cluster_candidates.append([])
                continue
            beam = [({}, 0.0)]
            for ck in sorted(grp_keys, key=lambda k: len(byclass[k])):
                if partial_clusters and len(byclass[ck]) == 0:
                    continue
                new_beam = []
                for sub_map, cost in beam:
                    used_ids = {v['res_id'] for v in sub_map.values()}
                    for r in byclass[ck]:
                        if r['res_id'] in used_ids:
                            continue
                        new_map = dict(sub_map)
                        new_map[ck] = r
                        new_beam.append((new_map, cost))
                if args.beam_width is not None and args.beam_width > 0:
                    new_beam = sorted(new_beam, key=lambda t: t[1])[:args.beam_width]
                beam = new_beam
                if partial_clusters and not beam:
                    break
            # score beam by internal RMSD + identity bonus
            scored = []
            for m, _ in beam:
                if (not partial_clusters) and len(m) != len(grp_keys):
                    continue
                if partial_clusters and len(m) == 0:
                    continue
                bonus = 0.0
                if tmpl_meta:
                    for ck in m.keys():
                        lab = mapping.get(ck)
                        meta = tmpl_meta.get(lab) if lab else None
                        if not meta:
                            continue
                        allowed = next((aas for k, aas in classes if k == ck), "")
                        if m[ck]['aa'] == meta['aa'] and m[ck]['resseq'] == meta['resseq']:
                            bonus += args.identity_bonus if len(allowed) == 1 else args.identity_bonus_degen
                ir = internal_rmsd(list(m.keys()), m) - bonus
                # centroid for this cluster
                pts = [m[ck]['xyz'] for ck in m.keys()]
                ccent = centroid(pts)
                scored.append((ir, m, ccent))
            scored.sort(key=lambda t: t[0])
            if args.cluster_top_n and args.cluster_top_n > 0:
                scored = scored[:args.cluster_top_n]
            cluster_candidates.append(scored)
        # Early per-protein reject: if a required cluster has no candidates at all.
        if required_clusters and any(len(cluster_candidates[gi]) == 0 for gi in required_clusters):
            continue

        # beam search across clusters (beam=1 => greedy)
        beam_width = max(1, int(args.cluster_seed_beam or 1))
        beam = [{
            "score": 0.0,
            "selected": {},
            "selected_clusters": [],
            "selected_centroids": [],
            "selected_tcentroids": [],
            "used_ids": set(),
            "remaining": set(range(len(cluster_groups))),
        }]

        while True:
            expanded = []
            progressed = False
            for state in beam:
                # Early prune state when any still-required cluster has no valid non-overlapping assignment.
                if required_clusters:
                    missing_required = [gi for gi in required_clusters if gi not in state["selected_clusters"]]
                    impossible = False
                    for gi in missing_required:
                        has_valid = False
                        for _, m, _ in cluster_candidates[gi]:
                            if any(r['res_id'] in state["used_ids"] for r in m.values()):
                                continue
                            has_valid = True
                            break
                        if not has_valid:
                            impossible = True
                            break
                    if impossible:
                        continue
                best_for_state = None
                first_pick = (len(state["selected_clusters"]) == 0)
                for gi in list(state["remaining"]):
                    if first_pick and priority_seed and gi not in priority_seed:
                        continue
                    for ir, m, ccent in cluster_candidates[gi]:
                        if any(r['res_id'] in state["used_ids"] for r in m.values()):
                            continue
                        cr = centroid_pair_rmsd(
                            state["selected_centroids"],
                            state["selected_tcentroids"],
                            ccent,
                            Tcluster_centroids[gi],
                        )
                        score = args.cluster_weight_internal * ir + args.cluster_weight_centroid * cr
                        if best_for_state is None or score < best_for_state[0]:
                            best_for_state = (score, gi, m, ccent)
                # Fallback: if prioritized seed clusters had no valid candidate, allow any cluster.
                if best_for_state is None and first_pick and priority_seed:
                    for gi in list(state["remaining"]):
                        for ir, m, ccent in cluster_candidates[gi]:
                            if any(r['res_id'] in state["used_ids"] for r in m.values()):
                                continue
                            cr = centroid_pair_rmsd(
                                state["selected_centroids"],
                                state["selected_tcentroids"],
                                ccent,
                                Tcluster_centroids[gi],
                            )
                            score = args.cluster_weight_internal * ir + args.cluster_weight_centroid * cr
                            if best_for_state is None or score < best_for_state[0]:
                                best_for_state = (score, gi, m, ccent)
                if best_for_state is None:
                    expanded.append(state)
                    continue
                progressed = True
                score, gi, m, ccent = best_for_state
                new_state = {
                    "score": state["score"] + score,
                    "selected": dict(state["selected"]),
                    "selected_clusters": list(state["selected_clusters"]),
                    "selected_centroids": list(state["selected_centroids"]),
                    "selected_tcentroids": list(state["selected_tcentroids"]),
                    "used_ids": set(state["used_ids"]),
                    "remaining": set(state["remaining"]),
                }
                for ck, r in m.items():
                    new_state["selected"][ck] = r
                    new_state["used_ids"].add(r['res_id'])
                new_state["selected_clusters"].append(gi)
                new_state["selected_centroids"].append(ccent)
                new_state["selected_tcentroids"].append(Tcluster_centroids[gi])
                new_state["remaining"].remove(gi)
                expanded.append(new_state)

            if not progressed:
                beam = expanded
                break
            expanded.sort(key=lambda s: s["score"])
            beam = expanded[:beam_width]

        beam.sort(key=lambda s: s["score"])
        state = beam[0] if beam else None
        if state is None or not state["selected"]:
            continue
        if args.min_match_clusters is not None and len(state["selected_clusters"]) < args.min_match_clusters:
            continue
        if required_clusters and not required_clusters.issubset(set(state["selected_clusters"])):
            continue
        selected = state["selected"]
        selected_clusters = state["selected_clusters"]
        selected_centroids = state["selected_centroids"]
        selected_tcentroids = state["selected_tcentroids"]
        if args.cluster_seed_min_residues is not None and len(selected) < args.cluster_seed_min_residues:
            continue

        # enforce require_residues and require_template
        if require_residues:
            ok_req = True
            for aa_req, resseq_req in require_residues:
                if not any(
                    r["aa"] == aa_req and r["resseq"] == resseq_req
                    for r in selected.values()
                ):
                    ok_req = False
                    break
            if not ok_req:
                continue
        if require_template:
            ok_req = True
            inv_map = {label: ck for ck, label in mapping.items()}
            for label, resseq_req in require_template.items():
                if label not in inv_map:
                    ok_req = False
                    break
                r = selected.get(inv_map[label])
                if r is None or r["aa"] != label or r["resseq"] != resseq_req:
                    ok_req = False
                    break
            if not ok_req:
                continue

        # compute final score using selected clusters
        Cpair = []
        Tpair = []
        for i in range(len(selected_centroids)):
            for j in range(i + 1, len(selected_centroids)):
                Cpair.append(distc(selected_centroids[i], selected_centroids[j]))
                Tpair.append(distc(selected_tcentroids[i], selected_tcentroids[j]))
        rmsd_centroid = rmsd_over_pairs(Cpair, Tpair) if Cpair else 0.0

        Cinternal = []
        Tinternal = []
        for gi in selected_clusters:
            grp = cluster_groups[gi]
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    if a in selected and b in selected:
                        Cinternal.append(distc(selected[a]['xyz'], selected[b]['xyz']))
                        Tinternal.append(distc(Tcoords[a], Tcoords[b]))
        rmsd_internal = rmsd_over_pairs(Cinternal, Tinternal) if Cinternal else 0.0

        score = args.cluster_weight_centroid * rmsd_centroid + args.cluster_weight_internal * rmsd_internal

        # build ordered residues in class_keys order for stable output
        ordered = [selected[ck] for ck in class_keys if ck in selected]
        resid_set = set(r['res_id'] for r in ordered)
        payload = (score, e, uid, resid_set, tuple(), *ordered)
        candidates.append(payload)

    candidates.sort(key=lambda t: t[0])

    # Optional MJ normalization across pockets (peptide-contact MJ on selected residues)
    if args.mj_norm and candidates:
        mj_vals = []
        mj_meta = []
        for payload in candidates:
            score0, e, uid, resid_set, dist_list, *ordered = payload
            motif_points = [r["xyz"] for r in ordered]
            motif_aas = [r["aa"] for r in ordered]
            mj_sum, n_contacts = peptide_contact_mj(motif_points, motif_aas, args, mj)
            mj_metric = (-mj_sum / n_contacts) if n_contacts > 0 else 0.0
            mj_vals.append(mj_metric)
            mj_meta.append((mj_sum, n_contacts, mj_metric))

        mean = sum(mj_vals) / len(mj_vals) if mj_vals else 0.0
        var = sum((x - mean) ** 2 for x in mj_vals) / len(mj_vals) if mj_vals else 0.0
        std = math.sqrt(var) if var > 0 else 1.0

        rescored = []
        for payload, meta in zip(candidates, mj_meta):
            score0, e, uid, resid_set, dist_list, *ordered = payload
            mj_metric = meta[2]
            mj_norm = (mj_metric - mean) / std
            score = score0 + args.weight_mj * mj_norm
            rescored.append((score, e, uid, resid_set, dist_list, *ordered))
        rescored.sort(key=lambda t: t[0])
        candidates = rescored

    return candidates

def rescore_candidates(
    candidates_raw,
    pairs,
    mapping,
    Tdist,
    Tpoints,
    class_keys,
    tmpl_base,
    args,
):
    """
    candidates_raw: list of payloads from build_candidates, each payload includes dist_list
    Returns: list of rescored payloads with updated score (same payload shape)
    """
    rescored = []
    cluster_groups = []
    cluster_idx = {}
    if getattr(args, "cluster_groups", None):
        cluster_groups = parse_cluster_groups(args.cluster_groups, class_keys)
        for i, grp in enumerate(cluster_groups):
            for ck in grp:
                cluster_idx[ck] = i
    tcentroid = None
    tcdist = None
    if args.stage0_centroid_rmsd is not None:
        tcentroid = bbox_center(Tpoints)
        tcdist = [distc(p, tcentroid) for p in Tpoints]

    for payload in candidates_raw:
        # payload layout: (score0, entry, uid, resid_set, dist_list, *ordered)
        score0, e, uid, resid_set, dist_list, *ordered = payload

        # gates that depend on Tdist
        if not cluster_groups:
            if min(dist_list) < args.min_pair or max(dist_list) > args.max_pair:
                continue
        if args.stage1_rmsd_tol is not None:
            points = [r['xyz'] for r in ordered]
            if kabsch_rmsd(points, Tpoints) > args.stage1_rmsd_tol:
                continue
        if tcdist is not None:
            coords = [r['xyz'] for r in ordered]
            c = bbox_center(coords)
            cd = [distc(p, c) for p in coords]
            if rmsd_over_pairs(cd, tcdist) > args.stage0_centroid_rmsd:
                continue
        # score
        if args.use_weighted:
            num = 0.0
            den = 0.0
            for (p, dc) in zip(pairs, dist_list):
                aa1 = mapping[p[0]]
                aa2 = mapping[p[1]]
                w = WEIGHTS.get((aa1, aa2), WEIGHTS.get((aa2, aa1), 1.0))
                num += w * (dc - Tdist[p])**2
                den += w
            score = math.sqrt(num / den) if den > 0 else float('inf')
        else:
            score = math.sqrt(sum((dc - Tdist[p])**2 for (p, dc) in zip(pairs, dist_list)) / len(dist_list))

        if args.blend_centroid:
            if args.centroid_peptide:
                mj = getattr(args, "mj_matrix", None)
                if mj is None:
                    mj = load_mj_matrix()
                motif_aas = [r["aa"] for r in ordered]
                tmpl_aas = [mapping[ck] for ck in class_keys]
                cd_rmsd = centroid_peptide_rmsd(
                    [r["xyz"] for r in ordered],
                    motif_aas,
                    [tmpl_base[mapping[ck]] for ck in class_keys],
                    tmpl_aas,
                    args,
                    mj,
                )
            else:
                # centroid-distance RMSD across positions (alignment-invariant)
                coords = [r['xyz'] for r in ordered]
                cd_rmsd = centroid_dist_rmsd(coords, Tpoints)
            score = args.weight_pair * score + args.weight_centroid * cd_rmsd

        rescored.append((score, e, uid, resid_set, dist_list, *ordered))

    rescored.sort(key=lambda t: t[0])
    return rescored


def nonempty_subsets(chars: list[str]) -> list[tuple[str, ...]]:
    out = []
    for r in range(1, len(chars) + 1):
        out.extend(list(itertools.combinations(chars, r)))
    return out


def enumerate_variants(pattern: str, variant_subsets: bool) -> list[str]:
    tokens = parse_pattern_tokens(pattern)
    per_pos = []
    for t in tokens:
        letters = sorted(t['letters'])
        if t['bracket'] and variant_subsets:
            subsets = nonempty_subsets(letters)
            per_pos.append([''.join(s) for s in subsets])
        else:
            per_pos.append([''.join(letters)])

    variants = []
    for choice in itertools.product(*per_pos):
        parts = []
        for s in choice:
            if len(s) == 1:
                parts.append(s)
            else:
                parts.append(f"[{s}]")
        variants.append(' '.join(parts))
    return variants


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float('-inf')
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def logsumexp(logs: list[float]) -> float:
    m = max(logs)
    if m == float('-inf'):
        return m
    return m + math.log(sum(math.exp(x - m) for x in logs))


def parse_cluster_groups(spec: str, class_keys: list[str]) -> list[list[str]]:
    """
    Parse cluster groups like "(1-2)(3-7)(8-10)(11-12)(13)(14-15)(16-18)"
    into class-key groups based on 1-based pattern positions.
    """
    if not spec:
        return []
    if '(' not in spec or ')' not in spec:
        raise RuntimeError("Cluster groups must be provided in parentheses, e.g. \"(1-2)(3-7)(8-10)\".")
    # reject stray non-space characters outside groups
    cleaned = re.sub(r"\([^)]*\)", "", spec)
    if cleaned.strip():
        raise RuntimeError("Cluster groups must be specified only as parenthesized groups, e.g. \"(1-2)(3-7)\".")

    n = len(class_keys)
    groups = []
    seen = set()
    for grp in re.findall(r"\(([^)]*)\)", spec):
        items = [p.strip() for p in grp.split(',') if p.strip()]
        if not items:
            raise RuntimeError("Empty cluster group found.")
        idxs = []
        for part in items:
            if '-' in part:
                a, b = part.split('-', 1)
                a = int(a)
                b = int(b)
                if a > b:
                    a, b = b, a
                idxs.extend(list(range(a, b + 1)))
            else:
                idxs.append(int(part))
        for i in idxs:
            if i < 1 or i > n:
                raise RuntimeError(f"Cluster index {i} out of range (1..{n}).")
            if i in seen:
                raise RuntimeError(f"Cluster index {i} appears in multiple groups.")
            seen.add(i)
        groups.append([class_keys[i - 1] for i in idxs])
    if len(seen) != n:
        raise RuntimeError("Cluster groups must cover all pattern positions.")
    return groups


def rmsd_over_pairs(d1: list[float], d2: list[float]) -> float:
    if not d1 or not d2 or len(d1) != len(d2):
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(d1, d2)) / len(d1))


def hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    # P[X >= k], X ~ Hypergeom(N, K, n)
    max_x = min(K, n)
    if k > max_x:
        return 0.0
    if k <= 0:
        return 1.0
    logs = []
    for x in range(k, max_x + 1):
        logp = log_comb(K, x) + log_comb(N - K, n - x) - log_comb(N, n)
        logs.append(logp)
    return math.exp(logsumexp(logs))


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    ranked = [pvals[i] for i in order]
    q = [0.0] * m
    prev = 1.0
    for idx in range(m - 1, -1, -1):
        i = idx + 1
        val = ranked[idx] * m / i
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    out = [0.0] * m
    for j, i in enumerate(order):
        out[i] = q[j]
    return out


def fmt_sig(x: float, sig: int = 3) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
    return f"{x:.{sig}g}"


def run_bclamp_2cluster_mode(args, entries, entry_to_uid):
    """
    Two-cluster filter mode:
    1) Find best 4-position cluster-1 pattern match to DPO3B geometry per entry.
    2) Require a connected local triplet from args.bclamp_triplet_set that
       matches DPO3B T/G/L geometry and is near the DHRY cluster.
    """
    def rmsd_vals(a, b):
        if len(a) != len(b) or not a:
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))

    def pairvec4(P):
        return [
            distc(P[0], P[1]),
            distc(P[0], P[2]),
            distc(P[0], P[3]),
            distc(P[1], P[2]),
            distc(P[1], P[3]),
            distc(P[2], P[3]),
        ]

    def pairvec3_sorted(P):
        return sorted([
            distc(P[0], P[1]),
            distc(P[0], P[2]),
            distc(P[1], P[2]),
        ])

    def first_valid_slot_assignment(aas, allowed_sets):
        # Return first residue-index assignment by slot, or None if impossible.
        used = set()
        out = [None] * len(allowed_sets)

        def rec(slot_i):
            if slot_i == len(allowed_sets):
                return True
            for ridx, aa in enumerate(aas):
                if ridx in used:
                    continue
                if aa not in allowed_sets[slot_i]:
                    continue
                used.add(ridx)
                out[slot_i] = ridx
                if rec(slot_i + 1):
                    return True
                used.remove(ridx)
                out[slot_i] = None
            return False

        if rec(0):
            return out
        return None

    # fixed DPO3B references for this mode
    dpo_path = os.path.join(args.ecoli_batch, "DPO3B_ECOLI", "residues.tsv")
    if not os.path.exists(dpo_path):
        raise RuntimeError(f"DPO3B template residues not found: {dpo_path}")
    need = {
        ("D", "173"),
        ("H", "175"),
        ("R", "176"),
        ("Y", "323"),
        ("T", "172"),
        ("G", "174"),
        ("L", "177"),
    }
    tcoords = {}
    with open(dpo_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["aa"], row["resseq"])
            if key in need:
                tcoords[key] = (
                    float(row["sc_x"]),
                    float(row["sc_y"]),
                    float(row["sc_z"]),
                )
    missing = [k for k in need if k not in tcoords]
    if missing:
        raise RuntimeError(f"Missing DPO3B template residues: {missing}")

    T_dhry = [
        tcoords[("D", "173")],
        tcoords[("H", "175")],
        tcoords[("R", "176")],
        tcoords[("Y", "323")],
    ]
    dhry_tokens = parse_pattern_tokens(args.bclamp_dhry_pattern)
    if len(dhry_tokens) != 4:
        raise RuntimeError(
            "--bclamp-dhry-pattern must define exactly 4 positions (e.g., 'D H R Y' or '[DE] H [RK] Y')."
        )
    dhry_allowed = [set(t["letters"]) for t in dhry_tokens]
    if any(not s for s in dhry_allowed):
        raise RuntimeError("--bclamp-dhry-pattern has an empty position class.")

    def format_dhry_label(chosen):
        return ",".join(f"{r['aa']}{r['resseq']}" for r in chosen)

    def parse_seed_residues(res):
        toks = [x.strip() for x in res.split(",") if x.strip()]
        if len(toks) != 4:
            return None
        out = {}
        for t in toks:
            aa = t[0]
            pos = t[1:]
            if not pos.isdigit():
                return None
            matches = [i for i, allowed in enumerate(dhry_allowed) if aa in allowed and i not in out]
            if len(matches) != 1:
                return None
            out[matches[0]] = pos
        if len(out) != 4:
            return None
        return out

    T_dhry_pair = pairvec4(T_dhry)
    T_dhry_pair_map = {(i, j): distc(T_dhry[i], T_dhry[j]) for i in range(4) for j in range(i + 1, 4)}
    dhry_sse_limit = 6.0 * (args.bclamp_dhry_pairdist_cutoff ** 2)

    T_tgl = [
        tcoords[("T", "172")],
        tcoords[("G", "174")],
        tcoords[("L", "177")],
    ]
    T_tgl_pair = pairvec3_sorted(T_tgl)
    triplet_tokens = parse_pattern_tokens(args.bclamp_triplet_pattern)
    if len(triplet_tokens) != 3:
        raise RuntimeError(
            "--bclamp-triplet-pattern must define exactly 3 positions (e.g., 'G T L' or '[GASN] [TANV] [LIMV]')."
        )
    triplet_allowed = [set(t["letters"]) for t in triplet_tokens]
    if any(not s for s in triplet_allowed):
        raise RuntimeError("--bclamp-triplet-pattern has an empty position class.")
    triplet_set = set().union(*triplet_allowed)

    # Optional seed DHRY clusters from pulldown file when columns are present.
    # Expected format like residues="D313,H306,R315,Y309" keyed by entry or uid.
    dhry_seed_by_entry = {}
    dhry_seed_by_uid = {}
    try:
        with open(args.pulldown, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fields = set(reader.fieldnames or [])
            if "residues" in fields:
                for row in reader:
                    res = (row.get("residues") or "").strip()
                    if not res:
                        continue
                    tok = parse_seed_residues(res)
                    if tok is not None:
                        entry = (row.get("entry") or "").strip()
                        uid = (row.get("uniprot_id") or "").strip()
                        if entry:
                            dhry_seed_by_entry[entry] = tok
                        if uid:
                            dhry_seed_by_uid[uid] = tok
    except Exception:
        # Keep mode robust when pulldown file has a different shape.
        pass
    seeded_only = bool(dhry_seed_by_entry or dhry_seed_by_uid)
    rows_out = []

    for e in entries:
        path = os.path.join(args.ecoli_batch, e, "residues.tsv")
        if not os.path.exists(path):
            continue
        uid = entry_to_uid.get(e, "")
        if not uid:
            continue

        residues = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                residues.append({
                    "aa": row["aa"],
                    "resseq": row["resseq"],
                    "res_id": row["res_id"],
                    "xyz": (
                        float(row["sc_x"]),
                        float(row["sc_y"]),
                        float(row["sc_z"]),
                    ),
                })

        byaa = defaultdict(list)
        for r in residues:
            byaa[r["aa"]].append(r)
        slot_lists = [[r for r in residues if r["aa"] in dhry_allowed[i]] for i in range(4)]
        if any(not slot_lists[i] for i in range(4)):
            continue

        # best DHRY cluster under cutoffs.
        # Fast path: use seeded residues from input file when available.
        best_dhry = None
        seed = dhry_seed_by_entry.get(e) or dhry_seed_by_uid.get(uid)
        has_seed = seed is not None
        if seed:
            # Build direct index by (aa, resseq) for constant-time lookup.
            idx = {(x["aa"], x["resseq"]): x for x in residues}
            chosen = []
            for i in range(4):
                cands = [idx[k] for k in idx if k[1] == seed[i] and k[0] in dhry_allowed[i]]
                if len(cands) != 1:
                    chosen = []
                    break
                chosen.append(cands[0])
            if len(chosen) == 4:
                P = [chosen[i]["xyz"] for i in range(4)]
                pd = rmsd_vals(pairvec4(P), T_dhry_pair)
                if pd <= args.bclamp_dhry_pairdist_cutoff:
                    best_dhry = (pd, chosen)
        elif seeded_only:
            # In seeded datasets (e.g., bclamp_overlap.tsv), skip costly exhaustive fallback.
            continue

        # Fallback to exhaustive only for unseeded datasets.
        if seeded_only and has_seed and best_dhry is None:
            continue
        if best_dhry is None:
            for r0 in slot_lists[0]:
                for r1 in slot_lists[1]:
                    if r1["res_id"] == r0["res_id"]:
                        continue
                    dh = distc(r0["xyz"], r1["xyz"])
                    e_dh = dh - T_dhry_pair_map[(0, 1)]
                    sse_dh = e_dh * e_dh
                    if sse_dh > dhry_sse_limit:
                        continue
                    for r2 in slot_lists[2]:
                        if r2["res_id"] in (r0["res_id"], r1["res_id"]):
                            continue
                        dr = distc(r0["xyz"], r2["xyz"])
                        hr = distc(r1["xyz"], r2["xyz"])
                        e_dr = dr - T_dhry_pair_map[(0, 2)]
                        e_hr = hr - T_dhry_pair_map[(1, 2)]
                        sse_dhr = sse_dh + e_dr * e_dr + e_hr * e_hr
                        if sse_dhr > dhry_sse_limit:
                            continue
                        for r3 in slot_lists[3]:
                            if r3["res_id"] in (r0["res_id"], r1["res_id"], r2["res_id"]):
                                continue
                            dy = distc(r0["xyz"], r3["xyz"])
                            hy = distc(r1["xyz"], r3["xyz"])
                            ry = distc(r2["xyz"], r3["xyz"])
                            e_dy = dy - T_dhry_pair_map[(0, 3)]
                            e_hy = hy - T_dhry_pair_map[(1, 3)]
                            e_ry = ry - T_dhry_pair_map[(2, 3)]
                            sse = sse_dhr + e_dy * e_dy + e_hy * e_hy + e_ry * e_ry
                            if sse > dhry_sse_limit:
                                continue
                            pd = math.sqrt(sse / 6.0)
                            rec = (pd, [r0, r1, r2, r3])
                            if best_dhry is None or pd < best_dhry[0]:
                                best_dhry = rec
        if best_dhry is None:
            continue

        _, dhry_chosen = best_dhry
        dhry_ids = {r["res_id"] for r in dhry_chosen}
        dhry_points = [r["xyz"] for r in dhry_chosen]
        dhry_str = format_dhry_label(dhry_chosen)

        cands = [x for x in residues if x["aa"] in triplet_set and x["res_id"] not in dhry_ids]
        n = len(cands)
        if n < 3:
            continue

        # adjacency for local connected triplets
        Dmat = [[0.0] * n for _ in range(n)]
        adj = [set() for _ in range(n)]
        for i in range(n - 1):
            for j in range(i + 1, n):
                dij = distc(cands[i]["xyz"], cands[j]["xyz"])
                Dmat[i][j] = dij
                Dmat[j][i] = dij
                if dij <= args.bclamp_triplet_edge:
                    adj[i].add(j)
                    adj[j].add(i)

        # Precompute min distance to DHRY per candidate for fast near-pruning.
        min_to_dhry = [
            min(distc(cands[i]["xyz"], q) for q in dhry_points)
            for i in range(n)
        ]
        near_ok = [v <= args.bclamp_triplet_near for v in min_to_dhry]

        first_tri = None
        seen_trip = set()
        for j in range(n):
            nj = sorted(adj[j])
            if len(nj) < 2:
                continue
            for a in range(len(nj) - 1):
                i = nj[a]
                for b in range(a + 1, len(nj)):
                    k = nj[b]
                    if not (near_ok[i] or near_ok[j] or near_ok[k]):
                        continue
                    key = tuple(sorted((i, j, k)))
                    if key in seen_trip:
                        continue
                    seen_trip.add(key)
                    near = min(min_to_dhry[i], min_to_dhry[j], min_to_dhry[k])
                    if near > args.bclamp_triplet_near:
                        continue
                    p3 = sorted([Dmat[i][j], Dmat[j][k], Dmat[i][k]])
                    pd3 = rmsd_vals(p3, T_tgl_pair)
                    if pd3 > args.bclamp_triplet_pairdist_cutoff:
                        continue
                    tri = (cands[i], cands[j], cands[k])
                    slot_idx = first_valid_slot_assignment([x["aa"] for x in tri], triplet_allowed)
                    if slot_idx is None:
                        continue
                    tri_slot = (tri[slot_idx[0]], tri[slot_idx[1]], tri[slot_idx[2]])
                    first_tri = (pd3, near, tri_slot)
                    break
                if first_tri is not None:
                    break
            if first_tri is not None:
                break

        if first_tri is None:
            continue

        pd3, near, tri = first_tri
        tri_str = ";".join(f"{x['aa']}{x['resseq']}" for x in tri)
        rows_out.append({
            "uniprot_id": uid,
            "entry": e,
            "DHRY_cluster": dhry_str,
            "DHRY_pairdist_rmsd": f"{best_dhry[0]:.4f}",
            "triplet_set": args.bclamp_triplet_pattern.replace(" ", ""),
            "triplet_cluster": tri_str,
            "triplet_pairdist_rmsd": f"{pd3:.4f}",
            "triplet_min_dist_to_DHRY_A": f"{near:.4f}",
        })

    rows_out.sort(key=lambda r: (float(r["DHRY_pairdist_rmsd"]), float(r["triplet_pairdist_rmsd"])))
    out = args.out or "bclamp_2cluster.tsv"
    with open(out, "w", newline="") as f:
        fn = [
            "rank",
            "uniprot_id",
            "entry",
            "DHRY_cluster",
            "DHRY_pairdist_rmsd",
            "triplet_set",
            "triplet_cluster",
            "triplet_pairdist_rmsd",
            "triplet_min_dist_to_DHRY_A",
        ]
        w = csv.DictWriter(f, fieldnames=fn, delimiter="\t")
        w.writeheader()
        for i, r in enumerate(rows_out, start=1):
            rr = dict(r)
            rr["rank"] = i
            w.writerow(rr)
    print(out)
    print("count", len(rows_out))


# -------------------- main pipeline --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--pattern',
        default=DEFAULT_TEMPLATE_PATTERN,
        help='Pattern like "H [RK] [LIVM] [DE] Y" (default: DPO3B 19-position degenerate pattern).',
    )
    ap.add_argument('--ecoli-batch', default='src/ecoli_batch')
    ap.add_argument('--pulldown', default='pulldown.tsv')
    ap.add_argument('--radius', type=float, default=None,
                    help='Centroid radius cutoff (default: mean + 2*std of template bbox distances).')
    ap.add_argument('--min-radius', type=float, default=None,
                    help='Minimum centroid radius (default: mean - 2*std of template bbox distances).')
    ap.add_argument('--radius-far-slack', type=float, default=1.5,
                    help='Extra slack for distances beyond template mean (default: 1.5 Å).')
    ap.add_argument('--pair-prune', type=float, default=None,
                    help='Neighbor prune distance (default: template max + 1 Å).')
    ap.add_argument('--min-pair', type=float, default=None,
                    help='Minimum allowed pairwise distance (default: template min minus 1 Å).')
    ap.add_argument('--max-pair', type=float, default=None,
                    help='Maximum allowed pairwise distance (default: template max).')
    ap.add_argument('--stage0-centroid-rmsd', type=float, default=10.0,
                    help='Stage 0 prefilter: centroid-distance RMSD cutoff in Å (cluster mode only).')
    ap.add_argument('--stage1-rmsd-tol', type=float, default=None,
                    help='Stage 1 geometric check: Kabsch RMSD cutoff in Å (default: disabled).')
    ap.add_argument('--per-protein-cap', type=int, default=10)
    # Removed per-protein max checks cap (discovery-first).
    ap.add_argument('--jaccard-merge-threshold', type=float, default=0.6,
                    help='If >0, merge same-protein pockets with Jaccard overlap >= threshold (use best score).')
    ap.add_argument('--align-merge-threshold', type=float, default=1.5,
                    help='If >0, merge same-protein pockets with 5-point RMSD after optimal alignment <= threshold.')
    ap.add_argument('--use-weighted', action='store_true')
    ap.add_argument('--blend-centroid', action='store_true', default=True,
                    help='Blend pairwise-distance score with centroid-distance RMSD (alignment-invariant).')
    ap.add_argument('--weight-pair', type=float, default=1.0,
                    help='Weight for pairwise-distance score in blended scoring (default: 1.0).')
    ap.add_argument('--weight-centroid', type=float, default=1.0,
                    help='Weight for centroid-distance RMSD in blended scoring (default: 1.0).')
    ap.add_argument('--centroid-peptide', action='store_true', default=False,
                    help='Use peptide-based centroid RMSD (best MJ contact per motif residue).')
    ap.add_argument('--mj-norm', action='store_true', default=True,
                    help='Add normalized peptide-contact MJ term to score (default: on).')
    ap.add_argument('--no-mj-norm', dest='mj_norm', action='store_false',
                    help='Disable normalized peptide-contact MJ term.')
    ap.add_argument('--weight-mj', type=float, default=0.5,
                    help='Weight for normalized MJ term in blended scoring (default: 0.5).')
    ap.add_argument('--peptide-seq', default=None,
                    help='Peptide sequence for centroid-peptide scoring (e.g., MDRWLVK).')
    ap.add_argument('--peptide-capacity-map', default='W:2',
                    help='One-sided per-residue peptide position capacities for MJ assignment, e.g. "W:2,F:2,Y:2".')
    ap.add_argument('--peptide-cutoff', type=float, default=6.0,
                    help='Peptide contact cutoff in Å (default: 6.0).')
    ap.add_argument('--peptide-rise', type=float, default=1.5,
                    help='Helix rise per residue in Å (default: 1.5).')
    ap.add_argument('--peptide-radius', type=float, default=2.3,
                    help='Helix radius in Å (default: 2.3).')
    ap.add_argument('--peptide-rot', type=float, default=100.0,
                    help='Helix rotation per residue in degrees (default: 100).')
    ap.add_argument('--peptide-ensemble', type=int, default=24,
                    help='Number of sampled peptide helix conformers (default: 24; flexible ensemble).')
    ap.add_argument('--peptide-ensemble-seed', type=int, default=123,
                    help='Random seed for peptide helix ensemble sampling (default: 123).')
    ap.add_argument('--peptide-rise-jitter', type=float, default=0.35,
                    help='Uniform perturbation range for rise in Å (default: ±0.35).')
    ap.add_argument('--peptide-radius-jitter', type=float, default=0.40,
                    help='Uniform perturbation range for helix radius in Å (default: ±0.40).')
    ap.add_argument('--peptide-rot-jitter', type=float, default=25.0,
                    help='Uniform perturbation range for rotation in degrees (default: ±25.0).')
    ap.add_argument('--peptide-tilt-jitter-deg', type=float, default=15.0,
                    help='Uniform perturbation range for axis tilt in degrees (default: ±15.0).')
    ap.add_argument('--peptide-center-shift', type=float, default=1.0,
                    help='Uniform perturbation range for center shift in Å on each axis (default: ±1.0).')
    ap.add_argument('--peptide-clash-radius', type=float, default=2.8,
                    help='Soft clash radius in Å for peptide-pocket contacts (default: 2.8).')
    ap.add_argument('--peptide-clash-weight', type=float, default=1.0,
                    help='Weight for soft clash penalty below clash radius (default: 1.0).')
    ap.add_argument('--peptide-hard-clash-min-dist', type=float, default=2.0,
                    help='Hard minimum peptide-pocket pseudoatom distance in Å; contacts below this are rejected (default: 2.0, <=0 disables).')
    ap.add_argument('--peptide-kink-enable', action='store_true', default=True,
                    help='Enable single-kink peptide conformers in ensemble sampling (default: on).')
    ap.add_argument('--no-peptide-kink-enable', dest='peptide_kink_enable', action='store_false',
                    help='Disable single-kink peptide conformers (rigid/no-kink mode).')
    ap.add_argument('--peptide-kink-max-bend-deg', type=float, default=25.0,
                    help='Maximum kink bend angle in degrees for sampled conformers (default: 25).')
    ap.add_argument('--peptide-kink-min-index', type=int, default=2,
                    help='Minimum 0-based hinge index for peptide kink (default: 2).')
    ap.add_argument('--peptide-kink-max-index', type=int, default=12,
                    help='Maximum 0-based hinge index for peptide kink (default: 12).')
    ap.add_argument('--mode', choices=['metrics','rank-pulldown','variant-search','pocket-search','bclamp-2cluster','pose-scan'], default='metrics')
    ap.add_argument('--out', default=None)
    ap.add_argument('--template-residues', default=None,
                    help='Comma-separated template residues in DPO3B (e.g., V247,P242,P363,M362).')
    ap.add_argument('--template-entry', default='DPO3B_ECOLI',
                    help='Entry name for template residues (default: DPO3B_ECOLI).')
    ap.add_argument('--only-entry', default=None,
                    help='Restrict search to a single entry name (e.g., DPO3B_ECOLI).')
    ap.add_argument('--only-pulldown', action='store_true',
                    help='Restrict search to entries whose UniProt IDs are in pulldown.tsv.')
    ap.add_argument('--stop-after-first-non-template', action='store_true',
                    help='Stop after first hit from a non-template entry (discovery fast path).')
    ap.add_argument('--require-residue', default=None,
                    help='Comma-separated required residues (e.g., Y328,W45). Candidates must include all.')
    ap.add_argument('--require-template', default=None,
                    help='Comma-separated template position requirements (e.g., Y:328,H:175).')
    ap.add_argument('--variant-subsets', action='store_true', default=True,
                    help='Enable subset expansion for bracket tokens (default: on).')
    ap.add_argument('--no-variant-subsets', dest='variant_subsets', action='store_false',
                    help='Disable subset expansion for bracket tokens.')
    ap.add_argument('--variant-M', type=int, default=None,
                    help='Top-M distinct sites to evaluate per variant (default: all sites).')
    ap.add_argument('--cluster-groups', default=None,
                    help='Cluster groups by 1-based pattern positions using parentheses (e.g., "(1-2)(3-7)(8-10)(11-12)(13-15)(16-19)").')
    ap.add_argument('--cluster-weight-centroid', type=float, default=1.0,
                    help='Weight for cluster centroid-pair RMSD (default: 1.0).')
    ap.add_argument('--cluster-weight-internal', type=float, default=1.0,
                    help='Weight for cluster internal-pair RMSD (default: 1.0).')
    ap.add_argument('--cluster-pair-tol', type=float, default=2.0,
                    help='Hard filter: max allowed RMSD of inter-cluster centroid pair distances (default: 2.0 Å).')
    ap.add_argument('--pos-radius', type=float, default=6.0,
                    help='Per-position max distance to template coordinate in Å (default: 6.0). Use <=0 to disable.')
    ap.add_argument('--segment-search', action='store_true',
                    help='Segmented search: match clusters independently and score by centroid RMSD.')
    ap.add_argument('--debug-segment-entry', default=None,
                    help='If set, print top stitched combos and gating for this entry.')
    ap.add_argument('--debug-template-entry', default=None,
                    help='If set, validate exact template combo against filters for this entry.')
    ap.add_argument('--debug-template-combo', action='store_true',
                    help='If set, evaluate exact template combo against segment-search gates.')
    ap.add_argument('--max-score', type=float, default=None,
                    help='Drop candidates with score above this threshold (segment-search score).')
    ap.add_argument('--min-match', type=int, default=None,
                    help='Require at least this many positions to match template AA class (default: all).')
    ap.add_argument('--min-match-clusters', type=int, default=None,
                    help='Cluster-seed only: require at least this many clusters to match (default: disabled).')
    ap.add_argument('--require-clusters', default=None,
                    help='Cluster-seed only: require specific 1-based cluster indices in final match (e.g., "1,2" or "1-2").')
    # Auto-relax removed; min-match is now strict.
    ap.add_argument('--identity-bonus', type=float, default=0.0,
                    help='Bonus (subtract from cost) when a residue matches the template residue at that position (default: 0.0).')
    ap.add_argument('--identity-bonus-degen', type=float, default=0.0,
                    help='Bonus for identity match at degenerate (multi-AA) positions (default: 0.0).')
    ap.set_defaults(template_first=True)
    ap.add_argument('--beam-width', type=int, default=1000,
                    help='Beam width for search (default: 1000). Larger = more coverage, slower.')
    ap.add_argument('--cluster-top-n', type=int, default=0,
                    help='Top-N assignments kept per cluster in segment search (default: 0 = no cap).')
    ap.add_argument('--cluster-seed', action='store_true',
                    help='Use greedy cluster-seed search (pocket-search only).')
    ap.add_argument('--cluster-seed-beam', type=int, default=1,
                    help='Beam width for cluster-seed search (default: 1 = greedy).')
    ap.add_argument('--cluster-seed-priority', default='2',
                    help='Cluster-seed only: comma-separated 1-based cluster indices to prioritize for first seed pick (default: 2; e.g., "1" or "1,2").')
    ap.add_argument('--cluster-seed-min-residues', type=int, default=None,
                    help='Cluster-seed only: require at least this many residues in the final match; when below motif length, allows partially populated clusters.')
    ap.add_argument('--progress-every', type=int, default=100,
                    help='Print progress every N entries (default: 100).')
    ap.add_argument('--null', choices=['permute_dist', 'jitter_dist'], default=None,
                    help='Null model (permute_dist or jitter_dist) to assess enrichment against perturbed template distances.')
    ap.add_argument('--seed', type=int, default=123,
                    help='Random seed for null permutations.')
    ap.add_argument('--n_perms', type=int, default=200,
                    help='Number of permutations for null model.')
    ap.add_argument('--jitter-sigmas', default='0.5,1.0,2.0',
                    help='Comma-separated sigmas for jitter_dist (default: 0.5,1.0,2.0 Å).')
    ap.add_argument('--bclamp-triplet-set', default='GAVLITS',
                    help='AA set for second cluster triplet in bclamp-2cluster mode (default: GAVLITS).')
    ap.add_argument('--bclamp-triplet-pattern', default='[GAVLITS] [GAVLITS] [GAVLITS]',
                    help='3-position pattern for second cluster in bclamp-2cluster mode (default: [GAVLITS] [GAVLITS] [GAVLITS]). Example: \"[GASN] [TANV] [LIMV]\".')
    ap.add_argument('--bclamp-dhry-pattern', default='D H R Y',
                    help='4-position pattern for cluster-1 matching in bclamp-2cluster mode (default: D H R Y). Example: \"[DE] H [RK] Y\".')
    ap.add_argument('--bclamp-triplet-edge', type=float, default=5.0,
                    help='Local-edge threshold in Å for connected triplets in bclamp-2cluster mode (default: 5.0).')
    ap.add_argument('--bclamp-triplet-near', type=float, default=6.0,
                    help='Max distance in Å from triplet to DHRY cluster in bclamp-2cluster mode (default: 6.0).')
    ap.add_argument('--bclamp-triplet-pairdist-cutoff', type=float, default=3.0,
                    help='Max pair-distance RMSD for triplet->TGL mapping in bclamp-2cluster mode (default: 3.0).')
    ap.add_argument('--bclamp-dhry-pairdist-cutoff', type=float, default=3.0,
                    help='Max pair-distance RMSD for DHRY->template mapping in bclamp-2cluster mode (default: 3.0).')
    ap.add_argument('--pose-scan-trials', type=int, default=2000,
                    help='Number of random centers to sample in pose-scan mode (default: 2000).')
    ap.add_argument('--pose-scan-center-radius', type=float, default=6.0,
                    help='Sampling radius in Å around motif centroid in pose-scan mode (default: 6.0).')
    ap.add_argument('--pose-scan-top-n', type=int, default=200,
                    help='Number of top poses to write in pose-scan mode (default: 200).')
    ap.add_argument('--pose-scan-min-contacts', type=int, default=5,
                    help='Minimum contacts required for an accepted pose in pose-scan mode (default: 5).')
    ap.add_argument('--pose-min-favorable', type=int, default=0,
                    help='Minimum number of favorable contacts required in pose-scan mode (default: 0). Favorable uses raw MJ threshold.')
    ap.add_argument('--pose-favorable-threshold', type=float, default=0.0,
                    help='Raw MJ threshold for favorable contacts in pose-scan mode (default: 0.0; favorable if raw MJ < threshold).')
    ap.add_argument('--pose-require-pep-idx', default='',
                    help='Comma-separated peptide indices (1-based) that must appear in accepted contacts, e.g. \"4\".')
    ap.add_argument('--pose-require-pocket-residues', default='',
                    help='Comma-separated pocket residues that must appear in accepted contacts, e.g. \"V247\" or \"V247,M362\".')
    ap.add_argument('--pose-anchor', default='',
                    help='Optional anchor constraints in pose-scan mode, e.g. \"4:V247,M362;3:D173\".')
    ap.add_argument('--pose-anchor-max-dist', type=float, default=6.5,
                    help='Max distance in Å to each anchored target in pose-scan mode (default: 6.5).')
    ap.add_argument('--pose-anchor-balance', type=float, default=2.5,
                    help='If an anchor lists >=2 targets, max allowed spread among those distances (default: 2.5 Å).')
    ap.add_argument('--pose-strict-clash', action='store_true', default=True,
                    help='Enable stricter pose clash filter against all protein proxy points (CA/SC, default: on).')
    ap.add_argument('--no-pose-strict-clash', dest='pose_strict_clash', action='store_false',
                    help='Disable strict pose clash filter (exploratory mode).')
    ap.add_argument('--pose-strict-clash-min-dist', type=float, default=1.4,
                    help='Hard minimum distance in Å for strict pose clash screening (default: 1.4).')
    ap.add_argument('--pose-strict-protein-points', default='ca,sc',
                    help='Comma-separated protein proxy points for strict clashes: ca, sc, or both (default: ca,sc).')
    args = ap.parse_args()
    try:
        args.peptide_capacity_map = parse_peptide_capacity_map(args.peptide_capacity_map)
    except RuntimeError as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        args.pose_strict_protein_points_set = parse_pose_strict_protein_points(
            args.pose_strict_protein_points
        )
    except RuntimeError as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)
    if args.peptide_ensemble < 1:
        print("Query failed: --peptide-ensemble must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.cluster_groups and not args.segment_search:
        print("Query failed: --cluster-groups requires --segment-search", file=sys.stderr)
        sys.exit(1)

    if len(parse_pattern(args.pattern)) < 2:
        print("Query failed: pattern must include at least 2 residues to define distances.", file=sys.stderr)
        sys.exit(1)
    # For min-match, enable segment search and allow K-of-N by default.
    if args.min_match is not None and not args.segment_search:
        args.segment_search = True
        args._auto_segment_search = True
    # If using segment-search with min-match and no cluster groups, default to DPO3B clusters.
    if args.segment_search and args.min_match is not None and not args.cluster_groups:
        args.cluster_groups = "(4,6-7,15)(8,11-13)(16,18-19)(1-2,9)(3,5)(10)(14,17)"
    if args.cluster_seed and not args.cluster_groups:
        args.cluster_groups = "(4,6-7,15)(8,11-13)(16,18-19)(1-2,9)(3,5)(10)(14,17)"
    if args.segment_search and '--beam-width' in sys.argv and not getattr(args, "_auto_segment_search", False):
        print("Query failed: --segment-search cannot be used with --beam-width", file=sys.stderr)
        sys.exit(1)

    # pocket-search is geometry-only by default unless a peptide is provided.
    if args.mode == 'pocket-search' and not args.peptide_seq:
        args.mj_norm = False
    if args.mode == 'bclamp-2cluster':
        args.mj_norm = False
    if args.mode == 'pose-scan':
        args.mj_norm = False

    if args.mode == 'variant-search':
        args.jaccard_merge_threshold = 0.0
        args.align_merge_threshold = 0.0
    if args.centroid_peptide and not args.peptide_seq:
        print("Query failed: --centroid-peptide requires --peptide-seq", file=sys.stderr)
        sys.exit(1)
    if args.mode == 'pose-scan' and not args.peptide_seq:
        print("Query failed: --mode pose-scan requires --peptide-seq", file=sys.stderr)
        sys.exit(1)
    if args.mj_norm and not args.peptide_seq:
        print("Query failed: --mj-norm requires --peptide-seq", file=sys.stderr)
        sys.exit(1)
    if args.centroid_peptide:
        args.mj_matrix = load_mj_matrix()

    if args.mode == 'pose-scan':
        try:
            run_pose_scan_mode(args)
        except RuntimeError as e:
            print(f"Query failed: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # load sets
    try:
        pulldown = load_pulldown(args.pulldown)
    except Exception as e:
        print(f"Query failed: could not read pulldown list from {args.pulldown}: {e}", file=sys.stderr)
        sys.exit(1)
    if not pulldown:
        print(f"Query failed: pulldown list is empty in {args.pulldown}", file=sys.stderr)
        sys.exit(1)
    entry_to_uid = load_entry_to_uid(args.ecoli_batch)
    N = len(set(entry_to_uid.values()))

    # load residues per entry (base allowed set from pattern tokens)
    base_tokens, pattern_info = parse_pattern_tokens(args.pattern, return_info=True)
    base_allowed = set()
    for t in base_tokens:
        base_allowed.update(list(t['letters']))
    entries = [os.path.basename(p) for p in glob.glob(os.path.join(args.ecoli_batch, '*')) if os.path.isdir(p)]
    if args.only_entry:
        entries = [args.only_entry] if args.only_entry in entries else []
    if args.only_pulldown:
        entries = [e for e in entries if entry_to_uid.get(e) in pulldown]
    if args.mode == 'bclamp-2cluster':
        try:
            run_bclamp_2cluster_mode(args, entries, entry_to_uid)
        except RuntimeError as e:
            print(f"Query failed: {e}", file=sys.stderr)
            sys.exit(1)
        return
    res_cache = {}
    for e in entries:
        path = os.path.join(args.ecoli_batch, e, 'residues.tsv')
        if not os.path.exists(path):
            continue
        res_cache[e] = load_residues(path, allowed=base_allowed)

    use_fixed_template = False
    tmpl_meta = None
    if args.template_residues:
        toks = [t.strip() for t in args.template_residues.split(',') if t.strip()]
        residues = []
        for t in toks:
            aa = t[0]
            resseq = t[1:]
            residues.append((aa, resseq))
        tmpl_base, tmpl_meta = template_from_entry_residues(args.ecoli_batch, args.template_entry, residues, return_meta=True)
        use_fixed_template = True
    else:
        tmpl_base, tmpl_meta = template_from_dpo3b(args.ecoli_batch, return_meta=True)

    args.require_residues = []
    raw_require_template = args.require_template
    args.require_template = {}
    if args.require_residue:
        toks = [t.strip() for t in args.require_residue.split(',') if t.strip()]
        for t in toks:
            aa = t[0]
            resseq = t[1:]
            args.require_residues.append((aa, resseq))
    if raw_require_template:
        toks = [t.strip() for t in raw_require_template.split(',') if t.strip()]
        for t in toks:
            if ':' not in t:
                raise SystemExit(f"Invalid --require-template token: {t}")
            label, resseq = t.split(':', 1)
            label = label.strip()
            resseq = resseq.strip()
            args.require_template[label] = resseq

    # Always use all cores for variant search now that variant-jobs is removed.
    args.variant_jobs = os.cpu_count() or 1

    # Fast template-only sanity check for pocket-search when only-entry matches template-entry.
    if (
        args.mode == 'pocket-search'
        and args.only_entry
        and args.template_entry
        and args.template_residues
        and args.only_entry == args.template_entry
        and not args.debug_segment_entry
    ):
        classes = parse_pattern(args.pattern)
        class_keys = [ck for ck, _ in classes]
        # mapping from class key to template residue label
        mapping = resolve_mapping(classes, tmpl_base, tmpl_meta, fixed_mapping=True)

        # load residues for template entry
        path = os.path.join(args.ecoli_batch, args.template_entry, 'residues.tsv')
        res = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                res.append(row)

        # build ordered residues in class_keys order
        ordered = []
        for ck in class_keys:
            label = mapping[ck]
            meta = tmpl_meta[label]
            aa = meta['aa']
            resseq = meta['resseq']
            hit = next((r for r in res if r['aa'] == aa and r['resseq'] == resseq), None)
            if hit is None:
                raise RuntimeError(f'Template residue {aa}{resseq} not found in {args.template_entry}')
            ordered.append({
                'aa': aa,
                'res_id': hit['res_id'],
                'resseq': resseq,
                'x': float(hit['sc_x']),
                'y': float(hit['sc_y']),
                'z': float(hit['sc_z']),
                'xyz': (float(hit['sc_x']), float(hit['sc_y']), float(hit['sc_z'])),
            })

        # compute template distances and centroid-distance RMSD
        Tcoords = {ck: tmpl_base[mapping[ck]] for ck in class_keys}
        Tpoints = [Tcoords[ck] for ck in class_keys]
        pairs = pair_list(class_keys)
        Tdist = build_tdist_from_tcoords(Tcoords, class_keys)

        coords = [r['xyz'] for r in ordered]
        cd_rmsd = centroid_dist_rmsd(coords, Tpoints)

        dist_list = tuple(
            distc(ordered[class_keys.index(a)]['xyz'], ordered[class_keys.index(b)]['xyz'])
            for a, b in pairs
        )
        score = math.sqrt(
            sum((dc - Tdist[p])**2 for p, dc in zip(pairs, dist_list))
            / len(dist_list)
        )

        # override score with cluster score if requested
        if args.cluster_groups:
            cluster_groups = parse_cluster_groups(args.cluster_groups, class_keys)
            Tcluster_centroids = []
            for grp in cluster_groups:
                pts = [Tcoords[ck] for ck in grp]
                Tcluster_centroids.append(centroid(pts))
            Tcluster_pair_dists = []
            for i in range(len(Tcluster_centroids)):
                for j in range(i + 1, len(Tcluster_centroids)):
                    Tcluster_pair_dists.append(distc(Tcluster_centroids[i], Tcluster_centroids[j]))
            Tcluster_internal_dists = []
            for grp in cluster_groups:
                if len(grp) < 2:
                    continue
                for i in range(len(grp)):
                    for j in range(i + 1, len(grp)):
                        Tcluster_internal_dists.append(distc(Tcoords[grp[i]], Tcoords[grp[j]]))
            coords_by_key = {ck: ordered[class_keys.index(ck)]['xyz'] for ck in class_keys}
            score = cluster_score_from_precomputed(
                coords_by_key,
                cluster_groups,
                Tcluster_pair_dists,
                Tcluster_internal_dists,
                args.cluster_weight_centroid,
                args.cluster_weight_internal,
            )

        # write single-result output
        out = args.out or 'pocket_search.tsv'
        with open(out, 'w', newline='') as f:
            f.write('rank\tscore\tcentroid_rmsd\tuniprot_id\tentry\tresidues\n')
            labels = [f"{r['aa']}{r['resseq']}" for r in ordered]
            uid = entry_to_uid.get(args.template_entry, '')
            f.write(f"1\t{fmt_sig(score)}\t{fmt_sig(cd_rmsd)}\t{uid}\t{args.template_entry}\t{','.join(labels)}\n")
        print(out)
        print('count', 1)
        return

    if args.mode == 'pocket-search' and args.cluster_seed:
        candidates = cluster_seed_search(
            args.pattern,
            args,
            res_cache,
            entry_to_uid,
            tmpl_base,
            fixed_mapping=use_fixed_template,
            tmpl_meta=tmpl_meta,
        )
        candidates = dedupe_candidates_by_uid(candidates)
        out = args.out or 'pocket_search.tsv'
        with open(out, 'w', newline='') as f:
            f.write('rank\tscore\tuniprot_id\tentry\tresidues\n')
            for i, (score, entry, uid, _resid_set, _dist_list, *res) in enumerate(candidates, start=1):
                labels = [f"{r['aa']}{r['resseq']}" for r in res]
                f.write(f"{i}\t{fmt_sig(score)}\t{uid}\t{entry}\t{','.join(labels)}\n")
        print(out)
        print('count', len(candidates))
        return

    # Allow graceful shutdown to dump partial results.
    _abort = {"hit": False}

    def _handle_abort(signum, frame):
        _abort["hit"] = True

    signal.signal(signal.SIGINT, _handle_abort)
    signal.signal(signal.SIGTERM, _handle_abort)
    args.abort_flag = _abort

    if args.mode == 'variant-search':
        variants = enumerate_variants(args.pattern, args.variant_subsets)

        shared = {
            "args": args,
            "entry_to_uid": entry_to_uid,
            "pulldown": pulldown,
            "N_univ": N,
            "variant_M": args.variant_M,
            "res_cache": res_cache,
            "tmpl_base": tmpl_base,
            "tmpl_meta": tmpl_meta,
            "use_fixed_template": use_fixed_template,
        }
        if args.centroid_peptide:
            shared["mj_matrix"] = load_mj_matrix()

        rows = []
        pvals = []
        if args.variant_jobs and args.variant_jobs > 1:
            with mp.Pool(processes=args.variant_jobs, initializer=_init_variant_worker, initargs=(shared,)) as pool:
                for r in pool.imap_unordered(_variant_task, variants):
                    if r is None:
                        continue
                    rows.append(r)
                    pvals.append(r['p_value'])
        else:
            _init_variant_worker(shared)
            for motif in variants:
                r = _variant_task(motif)
                if r is None:
                    continue
                rows.append(r)
                pvals.append(r['p_value'])

        if not rows:
            print('No variants produced any sites.')
            return

        qvals = bh_fdr(pvals)
        for r, q in zip(rows, qvals):
            r['q_value'] = q

        # Union enrichment for motifs passing p-value / FDR thresholds
        union_specs = [
            ("p", 0.05),
            ("q", 0.05),
        ]
        union_rows = []
        union_hitsets = {}
        for label, threshold in union_specs:
            union_hits = set()
            for r in rows:
                if r[f"{label}_value"] <= threshold:
                    union_hits.update(r.get('hit_proteins', set()))
            union_hitsets[label] = union_hits
            if union_hits:
                P_union = len(union_hits)
                P_union_pulldown = len(union_hits & pulldown)
                base = len(pulldown) / N if N else float('nan')
                union_enrich = (P_union_pulldown / P_union) / base if P_union else float('nan')
                a = P_union_pulldown
                b = P_union - P_union_pulldown
                c = len(pulldown) - P_union_pulldown
                d = (N - len(pulldown)) - b
                union_or = (a * d) / (b * c) if b > 0 and c > 0 else float('inf')
                union_p = hypergeom_sf(a - 1, N, len(pulldown), P_union)
                union_rows.append({
                    "label": f"{label}<=0.05",
                    "P": P_union,
                    "P_pulldown": P_union_pulldown,
                    "enrichment": union_enrich,
                    "odds_ratio": union_or,
                    "p_value": union_p,
                })
                print(f"union_{label}<=0.05", f"P={P_union}", f"P_pulldown={P_union_pulldown}",
                      f"enrichment={fmt_sig(union_enrich)}", f"odds_ratio={fmt_sig(union_or)}",
                      f"p_value={fmt_sig(union_p)}")
            else:
                print(f"union_{label}<=0.05", "no motifs passed threshold")

        rows.sort(key=lambda r: (r['p_value'], -r['fold_enrichment'], -r['P_pulldown']))
        out = args.out or 'variant_search.tsv'
        out_union = out.replace('.tsv', '_union.tsv') if out.endswith('.tsv') else out + '_union.tsv'
        out_pulldown_p = out.replace('.tsv', '_pulldown_hits_p<=0.05.tsv') if out.endswith('.tsv') else out + '_pulldown_hits_p<=0.05.tsv'
        out_pulldown_q = out.replace('.tsv', '_pulldown_hits_q<=0.05.tsv') if out.endswith('.tsv') else out + '_pulldown_hits_q<=0.05.tsv'
        with open(out, 'w', newline='') as f:
            f.write('motif\tM\tP\tP_pulldown\tfold_enrichment\todds_ratio\tp_value\tq_value\n')
            for r in rows:
                f.write(
                    f"{r['motif']}\t{r['M']}\t{r['P']}\t{r['P_pulldown']}\t"
                    f"{fmt_sig(r['fold_enrichment'])}\t{fmt_sig(r['odds_ratio'])}\t"
                    f"{fmt_sig(r['p_value'])}\t{fmt_sig(r['q_value'])}\n"
                )
        if union_rows:
            with open(out_union, 'w', newline='') as f:
                f.write('threshold\tP\tP_pulldown\tenrichment\todds_ratio\tp_value\n')
                for r in union_rows:
                    f.write(
                        f"{r['label']}\t{r['P']}\t{r['P_pulldown']}\t"
                        f"{fmt_sig(r['enrichment'])}\t{fmt_sig(r['odds_ratio'])}\t"
                        f"{fmt_sig(r['p_value'])}\n"
                    )
        # Write pulldown proteins present in union hits for p<=0.05 and q<=0.05
        for label, out_path in (('p', out_pulldown_p), ('q', out_pulldown_q)):
            hits = union_hitsets.get(label, set())
            pulldown_hits = sorted(hits & pulldown)
            with open(out_path, 'w', newline='') as f:
                f.write('uniprot_id\n')
                for uid in pulldown_hits:
                    f.write(f"{uid}\n")
        if not union_rows:
            with open(out_union, 'w', newline='') as f:
                f.write('threshold\tP\tP_pulldown\tenrichment\todds_ratio\tp_value\n')
                f.write('none\t0\t0\tnan\tnan\tnan\n')
        print(out)
        # Pretty print aligned table to stdout
        header = ['motif','M','P','P_pulldown','fold_enrichment','odds_ratio','p_value','q_value']
        table_rows = []
        for r in rows:
            table_rows.append([
                r['motif'],
                str(r['M']),
                str(r['P']),
                str(r['P_pulldown']),
                fmt_sig(r['fold_enrichment']),
                fmt_sig(r['odds_ratio']),
                fmt_sig(r['p_value']),
                fmt_sig(r['q_value']),
            ])
        widths = [len(h) for h in header]
        for row in table_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
        print(fmt.format(*header))
        for row in table_rows:
            print(fmt.format(*row))
        return

    try:
        if args.min_match is not None:
            args.min_match = min(args.min_match, len(parse_pattern(args.pattern)))
            candidates, candidates_raw, classes, mapping, Tdist = build_candidates(
                args.pattern, args, res_cache, entry_to_uid, tmpl_base,
                fixed_mapping=use_fixed_template, tmpl_meta=tmpl_meta
            )
            print(f"[min-match] {args.min_match}: hits={len(candidates)}", file=sys.stderr)
        else:
            candidates, candidates_raw, classes, mapping, Tdist = build_candidates(
                args.pattern, args, res_cache, entry_to_uid, tmpl_base, fixed_mapping=use_fixed_template, tmpl_meta=tmpl_meta
            )
    except RuntimeError as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        # If interrupted during candidate generation, dump partial results if any.
        if _abort["hit"]:
            out = args.out or 'pocket_search.tsv'
            if 'candidates' in locals() and candidates:
                candidates.sort(key=lambda t: t[0])
                with open(out, 'w', newline='') as f:
                    f.write('rank\tscore\tcentroid_rmsd\tuniprot_id\tentry\tresidues\n')
                    for i, (score, entry, uid, resid_set, dist_list, *res) in enumerate(candidates, start=1):
                        labels = [f"{r['aa']}{r['resseq']}" for r in res]
                        # centroid_rmsd not available here; write blank
                        f.write(f"{i}\t{fmt_sig(score)}\t\t{uid}\t{entry}\t{','.join(labels)}\n")
                print(out)
                print('count', len(candidates))
            else:
                print('Interrupted: no candidates to write yet.')
            sys.exit(1)
        raise
    pairs = pair_list([ck for ck, _ in classes])
    Tpoints = [tmpl_base[mapping[ck]] for ck, _ in classes]

    # Merge pockets within each protein using union-find:
    # connect if Jaccard >= threshold OR alignment RMSD <= threshold
    candidates = merge_pockets_within_uid(candidates, args)
    candidates_raw = merge_pockets_within_uid(candidates_raw, args)

    if args.mode == 'pocket-search':
        if _abort["hit"]:
            out = args.out or 'pocket_search.tsv'
            candidates.sort(key=lambda t: t[0])
            with open(out, 'w', newline='') as f:
                f.write('rank\tscore\tcentroid_rmsd\tuniprot_id\tentry\tresidues\n')
                for i, (score, entry, uid, resid_set, dist_list, *res) in enumerate(candidates, start=1):
                    labels = [f"{r['aa']}{r['resseq']}" for r in res]
                    f.write(f"{i}\t{fmt_sig(score)}\t\t{uid}\t{entry}\t{','.join(labels)}\n")
            print(out)
            print('count', len(candidates))
            return
        out = args.out or 'pocket_search.tsv'
        candidates = dedupe_candidates_by_uid(candidates)
        with open(out, 'w', newline='') as f:
            f.write('rank\tscore\tuniprot_id\tentry\tresidues\n')
            for i, (score, entry, uid, resid_set, dist_list, *res) in enumerate(candidates, start=1):
                labels = [f"{r['aa']}{r['resseq']}" for r in res]
                f.write(f"{i}\t{fmt_sig(score)}\t{uid}\t{entry}\t{','.join(labels)}\n")
        print(out)
        print('count', len(candidates))
        return

    if args.mode == 'rank-pulldown':
        # best per pulldown protein
        best = {}
        for score, entry, uid, resid_set, dist_list, *res in candidates:
            if uid not in pulldown:
                continue
            if uid not in best:
                best[uid] = (score, entry, res)
        ranked = sorted(best.items(), key=lambda t: t[1][0])
        out = args.out or 'pulldown_ranked_by_similarity.tsv'
        with open(out, 'w', newline='') as f:
            f.write('rank\tuniprot_id\tentry\tscore\tresidues\n')
            for i,(uid,(score, entry, res)) in enumerate(ranked, start=1):
                labels = [f"{r['aa']}{r['resseq']}" for r in res]
                f.write(f"{i}\t{uid}\t{entry}\t{fmt_sig(score)}\t{','.join(labels)}\n")
        print(out)
        print('count', len(ranked))
        return

    # metrics mode (top-M defined by distinct pocket sites)
    Ms = [25,50,100,200,400]
    out = args.out or 'pairwise_score_topM.tsv'

    # precompute global enrichment over all distinct sites across proteome
    all_sites = compute_top_sites_for_m(candidates_raw, args, 10**9)
    P_all = set(s['uid'] for s in all_sites)
    P_all_pulldown = P_all & pulldown
    N = len(set(entry_to_uid.values()))
    base_all = len(pulldown)/N if N else float('nan')
    global_all_enrich = (len(P_all_pulldown)/len(P_all))/base_all if P_all else float('nan')

    with open(out, 'w', newline='') as f:
        f.write('M\tP\tP_pulldown\tpulldown_enrichment\todds_ratio\tmean_site_deg\tmax_site_deg\n')
        rows = []
        metrics_map = compute_site_metrics_for_ms(candidates_raw, args, Ms, pulldown)
        for M in Ms:
            met = metrics_map[M]
            Pn = met['P']
            P_pulldownn = met['P_pulldown']
            base = len(pulldown)/N if N else float('nan')
            enrich = (P_pulldownn/Pn)/base if Pn else float('nan')
            a = P_pulldownn
            b = Pn - a
            c = len(pulldown) - a
            d = N - len(pulldown) - b
            oratio = (a*d)/(b*c) if b>0 and c>0 else float('inf')
            row = [
                str(M),
                str(Pn),
                str(P_pulldownn),
                fmt_sig(enrich),
                fmt_sig(oratio),
                fmt_sig(met['mean_deg']),
                str(met['max_deg']),
            ]
            rows.append(row)
            f.write("\t".join(row) + "\n")
    print(f"global_all_sites_enrichment\t{fmt_sig(global_all_enrich)}")
    print(out)

    # Pretty print aligned table to stdout
    header = ['M','P','P_pulldown','pulldown_enrichment','odds_ratio','mean_site_deg','max_site_deg']
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*header))
    for row in rows:
        print(fmt.format(*row))

    # Null model: permute template distances within motif
    if args.null in ('permute_dist', 'jitter_dist'):
        base_Tdist = Tdist
        obs_enrich = {int(r[0]): float(r[3]) for r in rows}
        if args.null == 'permute_dist':
            null_enrich = {m: [] for m in Ms}
            null_rows = []
            for i in range(args.n_perms):
                rng = random.Random(args.seed + i)
                Tdist_perm = permute_tdist(base_Tdist, rng=rng)
                c_raw_perm = rescore_candidates(
                    candidates_raw,
                    pairs,
                    mapping,
                    Tdist_perm,
                    Tpoints,
                    [ck for ck, _ in classes],
                    tmpl_base,
                    args,
                )
                metrics_perm = compute_site_metrics_for_ms(c_raw_perm, args, Ms, pulldown)
                for m in Ms:
                    Pn = metrics_perm[m]['P']
                    P_pulldownn = metrics_perm[m]['P_pulldown']
                    base = len(pulldown)/N if N else float('nan')
                    enr = (P_pulldownn/Pn)/base if Pn else float('nan')
                    null_enrich[m].append(enr)
                    null_rows.append({
                        'perm': i,
                        'M': m,
                        'enrichment': enr,
                        'P': Pn,
                        'P_pulldown': P_pulldownn,
                    })

            # save null distribution
            null_out = (args.out or 'pairwise_score_topM.tsv').replace('.tsv', '_null_perm.tsv')
            with open(null_out, 'w', newline='') as f:
                f.write('perm\tM\tenrichment\tP\tP_pulldown\n')
                for r in null_rows:
                    enr = r['enrichment']
                    enr_s = fmt_sig(enr)
                    f.write(f"{r['perm']}\t{r['M']}\t{enr_s}\t{r['P']}\t{r['P_pulldown']}\n")
            print(null_out)

            print("null_perm_summary")
            for m in Ms:
                obs = obs_enrich[m]
                vals = [v for v in null_enrich[m] if not math.isnan(v)]
                invalid = args.n_perms - len(vals)
                ge = sum(1 for v in vals if v >= obs)
                p_emp = (ge + 1) / (len(vals) + 1) if vals else float('nan')
                mean_null = sum(vals)/len(vals) if vals else float('nan')
                print(f"M={m}\tobs={fmt_sig(obs)}\tmean_null={fmt_sig(mean_null)}\tp_emp={fmt_sig(p_emp)}\tinvalid={invalid}")
        else:
            sigmas = [float(s.strip()) for s in args.jitter_sigmas.split(',') if s.strip()]
            print("null_jitter_summary")
            for sigma in sigmas:
                null_enrich = {m: [] for m in Ms}
                null_rows = []
                for i in range(args.n_perms):
                    rng = random.Random(args.seed + i)
                    Tdist_j = jitter_tdist(base_Tdist, sigma=sigma, rng=rng)
                    c_raw_perm = rescore_candidates(
                        candidates_raw,
                        pairs,
                        mapping,
                        Tdist_j,
                        Tpoints,
                        [ck for ck, _ in classes],
                        tmpl_base,
                        args,
                    )
                    metrics_perm = compute_site_metrics_for_ms(c_raw_perm, args, Ms, pulldown)
                    for m in Ms:
                        Pn = metrics_perm[m]['P']
                        P_pulldownn = metrics_perm[m]['P_pulldown']
                        base = len(pulldown)/N if N else float('nan')
                        enr = (P_pulldownn/Pn)/base if Pn else float('nan')
                        null_enrich[m].append(enr)
                        null_rows.append({
                            'perm': i,
                            'M': m,
                            'sigma': sigma,
                            'enrichment': enr,
                            'P': Pn,
                            'P_pulldown': P_pulldownn,
                        })

                null_out = (args.out or 'pairwise_score_topM.tsv').replace('.tsv', f'_null_jitter_sigma{sigma}.tsv')
                with open(null_out, 'w', newline='') as f:
                    f.write('perm\tM\tsigma\tenrichment\tP\tP_pulldown\n')
                    for r in null_rows:
                        enr = r['enrichment']
                        enr_s = fmt_sig(enr)
                        f.write(f"{r['perm']}\t{r['M']}\t{r['sigma']}\t{enr_s}\t{r['P']}\t{r['P_pulldown']}\n")
                print(null_out)

                for m in Ms:
                    obs = obs_enrich[m]
                    vals = [v for v in null_enrich[m] if not math.isnan(v)]
                    invalid = args.n_perms - len(vals)
                    ge = sum(1 for v in vals if v >= obs)
                    p_emp = (ge + 1) / (len(vals) + 1) if vals else float('nan')
                    mean_null = sum(vals)/len(vals) if vals else float('nan')
                    print(f"sigma={sigma}\tM={m}\tobs={fmt_sig(obs)}\tmean_null={fmt_sig(mean_null)}\tp_emp={fmt_sig(p_emp)}\tinvalid={invalid}")


if __name__ == '__main__':
    main()
