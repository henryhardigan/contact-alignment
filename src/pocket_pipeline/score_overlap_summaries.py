#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
from collections import Counter


DEFAULT_TARGET_BY_ROWS = {
    54: "data/r3_mapped_ac_ge2_minus_r1r2_sanitized.tsv",
    53: "data/r3_mapped_ac_ge2_minus_r1r2_sanitized.tsv",
    576: "data/r3_mapped_ac_ge2_sets_sanitized.tsv",
    577: "data/r3_mapped_ac_ge2_sets_sanitized.tsv",
    644: "data/r3_mapped_ac_ge2_sets.tsv",
    643: "data/r3_mapped_ac_ge2_sets.tsv",
}

VALID_CLASSES = {"APIM_only", "Both", "EYFP_only", "Neither"}


def read_metric_summary(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if "metric" not in (r.fieldnames or []) or "value" not in (r.fieldnames or []):
            return None
        out = {}
        for row in r:
            out[(row.get("metric") or "").strip()] = (row.get("value") or "").strip()
        return out


def gi(metrics, key):
    try:
        return int(float(metrics.get(key, "0") or "0"))
    except Exception:
        return 0


def canonical_class(raw):
    c = (raw or "").strip().replace("\r", "")
    if c == "Control_only":
        c = "EYFP_only"
    if c in VALID_CLASSES:
        return c
    return ""


def pick_first(row, keys):
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def infer_query_tsv_from_summary(summary_tsv):
    # Convert: data/foo_overlap_r3*_summary.tsv -> data/foo.tsv
    stem = summary_tsv[:-4] if summary_tsv.endswith(".tsv") else summary_tsv
    if stem.endswith("_summary"):
        stem = stem[:-8]
    marker = stem.find("_overlap_r3")
    if marker < 0:
        return ""
    q = stem[:marker] + ".tsv"
    if os.path.exists(q):
        return q
    return ""


def query_scope_from_path(query_tsv):
    if not query_tsv:
        return "unknown"
    name = os.path.basename(query_tsv).lower()
    if "proteome" in name:
        return "proteome"
    if "pulldown" in name:
        return "pulldown"
    return "unknown"


def load_target_universe_counts(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    counts = Counter()
    for r in rows:
        c = canonical_class(r.get("class"))
        counts[c or "Neither"] += 1
    return {
        "N": len(rows),
        "K_apim": counts.get("APIM_only", 0),
        "K_both": counts.get("Both", 0),
        "K_eyfp": counts.get("EYFP_only", 0),
        "K_neither": counts.get("Neither", 0),
    }


def load_proteome_universe_counts(class_tsv, ecoli_batch):
    if not class_tsv or not os.path.exists(class_tsv):
        return None
    if not ecoli_batch or not os.path.isdir(ecoli_batch):
        return None

    class_by_uid = {}
    with open(class_tsv, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            uid = pick_first(
                row,
                ("uniprot_id", "uid", "primary_id", "mapped_uniprot_id"),
            ).upper()
            if not uid:
                continue
            c = canonical_class(
                pick_first(
                    row,
                    (
                        "pulldown_region",
                        "class",
                        "r3_class_direct",
                        "class_control_ge2",
                        "class_control_ge1",
                        "region",
                    ),
                )
            )
            if c:
                class_by_uid[uid] = c

    searchable_uids = set()
    for meta_path in glob.glob(os.path.join(ecoli_batch, "*", "meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            uid = os.path.splitext(os.path.basename(meta.get("pdb", "")))[0].strip().upper()
            if uid:
                searchable_uids.add(uid)
        except Exception:
            continue

    if not searchable_uids:
        return None

    counts = Counter()
    for uid in searchable_uids:
        counts[class_by_uid.get(uid, "Neither")] += 1

    return {
        "N": len(searchable_uids),
        "K_apim": counts.get("APIM_only", 0),
        "K_both": counts.get("Both", 0),
        "K_eyfp": counts.get("EYFP_only", 0),
        "K_neither": counts.get("Neither", 0),
        "K_labeled": counts.get("APIM_only", 0) + counts.get("Both", 0) + counts.get("EYFP_only", 0),
        "class_tsv": class_tsv,
        "ecoli_batch": ecoli_batch,
        "_class_by_uid": class_by_uid,
        "_searchable_uids": searchable_uids,
    }


def load_query_class_counts(query_tsv, class_by_uid, searchable_uids):
    if not query_tsv or not os.path.exists(query_tsv):
        return None
    with open(query_tsv, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not (r.fieldnames or []) or "uniprot_id" not in (r.fieldnames or []):
            return None
        ids = set()
        for row in r:
            uid = (row.get("uniprot_id") or "").strip().upper()
            if uid:
                ids.add(uid)
    ids &= searchable_uids
    counts = Counter(class_by_uid.get(uid, "Neither") for uid in ids)
    return {
        "n": len(ids),
        "APIM_only": counts.get("APIM_only", 0),
        "Both": counts.get("Both", 0),
        "EYFP_only": counts.get("EYFP_only", 0),
        "Neither": counts.get("Neither", 0),
    }


def log_choose(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_p_geq(x, N, K, n):
    if x <= 0:
        return 1.0
    hi = min(K, n)
    if x > hi:
        return 0.0
    den = log_choose(N, n)
    s = 0.0
    for k in range(x, hi + 1):
        s += math.exp(log_choose(K, k) + log_choose(N - K, n - k) - den)
    return min(1.0, s)


def apim_or_and_ci(a, b, c, d):
    # Raw OR
    if b == 0 or c == 0:
        or_raw = float("inf")
    elif a == 0 or d == 0:
        or_raw = 0.0
    else:
        or_raw = (a * d) / (b * c)

    # Haldane-Anscombe corrected OR + Wald CI
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (aa * dd) / (bb * cc)
    se = math.sqrt((1.0 / aa) + (1.0 / bb) + (1.0 / cc) + (1.0 / dd))
    z = 1.96
    lo = math.exp(math.log(or_ha) - z * se)
    hi = math.exp(math.log(or_ha) + z * se)
    return or_raw, or_ha, lo, hi


def compute_enrichment_block(apim, eyfp, n, universe):
    N = int(universe.get("N", 0) or 0)
    K_apim = int(universe.get("K_apim", 0) or 0)
    K_eyfp = int(universe.get("K_eyfp", 0) or 0)

    # 2x2 table for APIM_only vs non-APIM
    a = apim
    b = max(0, n - apim)
    c = max(0, K_apim - apim)
    d = max(0, (N - K_apim) - b)

    apim_frac = (apim / n) if n else 0.0
    eyfp_frac = (eyfp / n) if n else 0.0
    apim_bg = (K_apim / N) if N else 0.0
    eyfp_bg = (K_eyfp / N) if N else 0.0

    apim_lift = (apim_frac / apim_bg) if apim_bg > 0 else float("inf")
    eyfp_lift = (eyfp_frac / eyfp_bg) if eyfp_bg > 0 else float("inf")

    p_apim = hypergeom_p_geq(apim, N, K_apim, n)
    p_eyfp = hypergeom_p_geq(eyfp, N, K_eyfp, n)
    or_raw, or_ha, or_lo, or_hi = apim_or_and_ci(a, b, c, d)

    return {
        "APIM_frac": apim_frac,
        "EYFP_frac": eyfp_frac,
        "APIM_bg_frac": apim_bg,
        "EYFP_bg_frac": eyfp_bg,
        "APIM_lift": apim_lift,
        "EYFP_lift": eyfp_lift,
        "APIM_OR_raw": or_raw,
        "APIM_OR_HA": or_ha,
        "APIM_OR_CI95_low": or_lo,
        "APIM_OR_CI95_high": or_hi,
        "p_APIM_enrich": p_apim,
        "p_EYFP_enrich": p_eyfp,
    }


def bh_fdr(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    min_so_far = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i_rank = m - rank + 1
        val = (pvals[idx] * m) / i_rank
        if val < min_so_far:
            min_so_far = val
        q[idx] = min(1.0, min_so_far)
    return q


def set_bh_qvalues(rows, p_key, q_key):
    idx = []
    pvals = []
    for i, row in enumerate(rows):
        p = row.get(p_key, "")
        if isinstance(p, (int, float)) and math.isfinite(float(p)):
            idx.append(i)
            pvals.append(float(p))
    for row in rows:
        row[q_key] = ""
    if not pvals:
        return
    qvals = bh_fdr(pvals)
    for i, q in zip(idx, qvals):
        rows[i][q_key] = q


def resolve_target_tsv(metrics, args):
    if args.target_tsv:
        return args.target_tsv

    from_summary = (metrics.get("target_tsv") or "").strip()
    if from_summary and os.path.exists(from_summary):
        return from_summary

    tr = gi(metrics, "target_rows")
    inferred = DEFAULT_TARGET_BY_ROWS.get(tr)
    if inferred and os.path.exists(inferred):
        return inferred

    return args.default_target_tsv


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Score overlap summary TSVs with APIM/EYFP enrichment under two null models: "
            "pull-down target universe and full searchable proteome."
        )
    )
    ap.add_argument(
        "--summary-glob",
        default="data/*overlap*r3*summary.tsv",
        help='Glob for input summary TSVs (default: "data/*overlap*r3*summary.tsv").',
    )
    ap.add_argument(
        "--target-tsv",
        default="",
        help="Optional fixed target universe TSV for all summaries.",
    )
    ap.add_argument(
        "--default-target-tsv",
        default="data/r3_mapped_ac_ge2_minus_r1r2_sanitized.tsv",
        help="Fallback pull-down universe when summary metadata is missing and row-count inference fails.",
    )
    ap.add_argument(
        "--proteome-class-tsv",
        default="data/apim_control_expanded_ecolibatch.tsv",
        help="Class label TSV used for proteome-background scoring.",
    )
    ap.add_argument(
        "--proteome-batch-dir",
        default="src/ecoli_batch",
        help="Searchable proteome directory (expects per-entry meta.json).",
    )
    ap.add_argument(
        "--no-proteome-model",
        action="store_true",
        help="Disable proteome-background scoring block.",
    )
    ap.add_argument(
        "--proteome-include-nonproteome-queries",
        action="store_true",
        help=(
            "By default, proteome model is only applied when summary filename maps to a query TSV "
            "whose basename contains 'proteome'. Set this flag to apply proteome model to all summaries."
        ),
    )
    ap.add_argument("--out", default="data/overlap_significance.tsv")
    return ap.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(args.summary_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.summary_glob}")

    target_universe_cache = {}
    query_class_cache = {}
    proteome_universe = None
    if not args.no_proteome_model:
        proteome_universe = load_proteome_universe_counts(args.proteome_class_tsv, args.proteome_batch_dir)
        if proteome_universe is None:
            print("warn_proteome_model_disabled missing_or_invalid_proteome_inputs")

    rows = []
    for path in files:
        metrics = read_metric_summary(path)
        if metrics is None:
            continue

        n = gi(metrics, "overlap_targets")
        apim = gi(metrics, "class_APIM_only")
        both = gi(metrics, "class_Both")
        eyfp = gi(metrics, "class_EYFP_only")
        qrows = gi(metrics, "query_rows")
        tr_summary = gi(metrics, "target_rows")

        target_tsv = resolve_target_tsv(metrics, args)
        if not target_tsv or not os.path.exists(target_tsv):
            continue

        if target_tsv not in target_universe_cache:
            target_universe_cache[target_tsv] = load_target_universe_counts(target_tsv)
        target_u = target_universe_cache[target_tsv]
        target_stats = compute_enrichment_block(apim, eyfp, n, target_u)
        inferred_query_tsv = infer_query_tsv_from_summary(path)
        query_scope = query_scope_from_path(inferred_query_tsv)
        allow_proteome = args.proteome_include_nonproteome_queries or (query_scope == "proteome")
        proteome_applied = int((proteome_universe is not None) and allow_proteome)
        proteome_skip_reason = ""
        if proteome_universe is None:
            proteome_skip_reason = "proteome_disabled_or_unavailable"
        elif not allow_proteome:
            proteome_skip_reason = "query_scope_not_proteome"

        row = {
            "summary_tsv": path,
            "query_tsv_inferred": inferred_query_tsv,
            "query_scope": query_scope,
            "query_rows": qrows,
            "target_rows_summary": tr_summary,
            "target_tsv_used": target_tsv,
            "universe_size": target_u["N"],
            "universe_apim_only": target_u["K_apim"],
            "universe_both": target_u["K_both"],
            "universe_eyfp_only": target_u["K_eyfp"],
            "universe_neither": target_u["K_neither"],
            "overlap_targets": n,
            "APIM_only": apim,
            "Both": both,
            "EYFP_only": eyfp,
            "APIM_frac": target_stats["APIM_frac"],
            "EYFP_frac": target_stats["EYFP_frac"],
            "APIM_bg_frac": target_stats["APIM_bg_frac"],
            "EYFP_bg_frac": target_stats["EYFP_bg_frac"],
            "APIM_lift": target_stats["APIM_lift"],
            "EYFP_lift": target_stats["EYFP_lift"],
            "APIM_OR_raw": target_stats["APIM_OR_raw"],
            "APIM_OR_HA": target_stats["APIM_OR_HA"],
            "APIM_OR_CI95_low": target_stats["APIM_OR_CI95_low"],
            "APIM_OR_CI95_high": target_stats["APIM_OR_CI95_high"],
            "p_APIM_enrich": target_stats["p_APIM_enrich"],
            "p_EYFP_enrich": target_stats["p_EYFP_enrich"],
            "proteome_model_enabled": int(proteome_universe is not None),
            "proteome_model_applied": proteome_applied,
            "proteome_model_skip_reason": proteome_skip_reason,
            "proteome_class_tsv_used": "",
            "proteome_batch_dir_used": "",
            "proteome_universe_size": "",
            "proteome_universe_apim_only": "",
            "proteome_universe_both": "",
            "proteome_universe_eyfp_only": "",
            "proteome_universe_neither": "",
            "proteome_universe_labeled": "",
            "proteome_APIM_bg_frac": "",
            "proteome_EYFP_bg_frac": "",
            "proteome_APIM_lift": "",
            "proteome_EYFP_lift": "",
            "proteome_APIM_OR_raw": "",
            "proteome_APIM_OR_HA": "",
            "proteome_APIM_OR_CI95_low": "",
            "proteome_APIM_OR_CI95_high": "",
            "p_APIM_enrich_proteome": "",
            "p_EYFP_enrich_proteome": "",
            "query_hits_searchable": "",
            "query_hits_APIM_only": "",
            "query_hits_Both": "",
            "query_hits_EYFP_only": "",
            "query_hits_Neither": "",
            "p_APIM_enrich_proteome_query": "",
            "p_EYFP_enrich_proteome_query": "",
        }

        if proteome_applied:
            prot_stats = compute_enrichment_block(apim, eyfp, n, proteome_universe)
            row.update(
                {
                    "proteome_class_tsv_used": proteome_universe["class_tsv"],
                    "proteome_batch_dir_used": proteome_universe["ecoli_batch"],
                    "proteome_universe_size": proteome_universe["N"],
                    "proteome_universe_apim_only": proteome_universe["K_apim"],
                    "proteome_universe_both": proteome_universe["K_both"],
                    "proteome_universe_eyfp_only": proteome_universe["K_eyfp"],
                    "proteome_universe_neither": proteome_universe["K_neither"],
                    "proteome_universe_labeled": proteome_universe["K_labeled"],
                    "proteome_APIM_bg_frac": prot_stats["APIM_bg_frac"],
                    "proteome_EYFP_bg_frac": prot_stats["EYFP_bg_frac"],
                    "proteome_APIM_lift": prot_stats["APIM_lift"],
                    "proteome_EYFP_lift": prot_stats["EYFP_lift"],
                    "proteome_APIM_OR_raw": prot_stats["APIM_OR_raw"],
                    "proteome_APIM_OR_HA": prot_stats["APIM_OR_HA"],
                    "proteome_APIM_OR_CI95_low": prot_stats["APIM_OR_CI95_low"],
                    "proteome_APIM_OR_CI95_high": prot_stats["APIM_OR_CI95_high"],
                    "p_APIM_enrich_proteome": prot_stats["p_APIM_enrich"],
                    "p_EYFP_enrich_proteome": prot_stats["p_EYFP_enrich"],
                }
            )

            if inferred_query_tsv not in query_class_cache:
                query_class_cache[inferred_query_tsv] = load_query_class_counts(
                    inferred_query_tsv,
                    proteome_universe.get("_class_by_uid", {}),
                    proteome_universe.get("_searchable_uids", set()),
                )
            qcc = query_class_cache[inferred_query_tsv]
            if qcc is not None:
                query_stats = compute_enrichment_block(
                    qcc["APIM_only"],
                    qcc["EYFP_only"],
                    qcc["n"],
                    proteome_universe,
                )
                row.update(
                    {
                        "query_hits_searchable": qcc["n"],
                        "query_hits_APIM_only": qcc["APIM_only"],
                        "query_hits_Both": qcc["Both"],
                        "query_hits_EYFP_only": qcc["EYFP_only"],
                        "query_hits_Neither": qcc["Neither"],
                        "p_APIM_enrich_proteome_query": query_stats["p_APIM_enrich"],
                        "p_EYFP_enrich_proteome_query": query_stats["p_EYFP_enrich"],
                    }
                )

        rows.append(row)

    if not rows:
        raise SystemExit("No metric-style summary TSVs parsed.")

    set_bh_qvalues(rows, "p_APIM_enrich", "q_APIM_enrich")
    set_bh_qvalues(rows, "p_EYFP_enrich", "q_EYFP_enrich")
    set_bh_qvalues(rows, "p_APIM_enrich_proteome", "q_APIM_enrich_proteome")
    set_bh_qvalues(rows, "p_EYFP_enrich_proteome", "q_EYFP_enrich_proteome")
    set_bh_qvalues(rows, "p_APIM_enrich_proteome_query", "q_APIM_enrich_proteome_query")
    set_bh_qvalues(rows, "p_EYFP_enrich_proteome_query", "q_EYFP_enrich_proteome_query")

    rows.sort(
        key=lambda r: (
            r["p_APIM_enrich"],
            -r["APIM_lift"],
            -r["overlap_targets"],
            r["p_EYFP_enrich"],
        )
    )

    headers = [
        "summary_tsv",
        "query_tsv_inferred",
        "query_scope",
        "query_rows",
        "target_rows_summary",
        "target_tsv_used",
        "universe_size",
        "universe_apim_only",
        "universe_both",
        "universe_eyfp_only",
        "universe_neither",
        "overlap_targets",
        "APIM_only",
        "Both",
        "EYFP_only",
        "APIM_frac",
        "EYFP_frac",
        "APIM_bg_frac",
        "EYFP_bg_frac",
        "APIM_lift",
        "EYFP_lift",
        "APIM_OR_raw",
        "APIM_OR_HA",
        "APIM_OR_CI95_low",
        "APIM_OR_CI95_high",
        "p_APIM_enrich",
        "q_APIM_enrich",
        "p_EYFP_enrich",
        "q_EYFP_enrich",
        "proteome_model_enabled",
        "proteome_model_applied",
        "proteome_model_skip_reason",
        "proteome_class_tsv_used",
        "proteome_batch_dir_used",
        "proteome_universe_size",
        "proteome_universe_apim_only",
        "proteome_universe_both",
        "proteome_universe_eyfp_only",
        "proteome_universe_neither",
        "proteome_universe_labeled",
        "proteome_APIM_bg_frac",
        "proteome_EYFP_bg_frac",
        "proteome_APIM_lift",
        "proteome_EYFP_lift",
        "proteome_APIM_OR_raw",
        "proteome_APIM_OR_HA",
        "proteome_APIM_OR_CI95_low",
        "proteome_APIM_OR_CI95_high",
        "p_APIM_enrich_proteome",
        "q_APIM_enrich_proteome",
        "p_EYFP_enrich_proteome",
        "q_EYFP_enrich_proteome",
        "query_hits_searchable",
        "query_hits_APIM_only",
        "query_hits_Both",
        "query_hits_EYFP_only",
        "query_hits_Neither",
        "p_APIM_enrich_proteome_query",
        "q_APIM_enrich_proteome_query",
        "p_EYFP_enrich_proteome_query",
        "q_EYFP_enrich_proteome_query",
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("wrote", args.out)
    print("count", len(rows))
    if proteome_universe is not None:
        print(
            "proteome_universe",
            proteome_universe["N"],
            proteome_universe["K_apim"],
            proteome_universe["K_both"],
            proteome_universe["K_eyfp"],
            proteome_universe["K_neither"],
        )


if __name__ == "__main__":
    main()
