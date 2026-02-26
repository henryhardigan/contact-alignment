#!/usr/bin/env python3
"""
Build a unified APIM-specificity feature matrix and fit a regularized model
with bootstrap stability scores (NumPy-only; no sklearn dependency).
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np


AA_CLASSES = {
    "aliphatic": set("VILM"),
    "aromatic": set("FWY"),
    "basic": set("RKH"),
    "acidic": set("DE"),
    "polar": set("STNQ"),
    "special": set("PG"),
}


def parse_comma_list(spec):
    return [x.strip() for x in (spec or "").split(",") if x.strip()]


def parse_query_globs(glob_args):
    out = []
    for g in glob_args:
        out.extend(parse_comma_list(g))
    return out


def maybe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def split_multi(s, seps=";,| "):
    if s is None:
        return []
    out = [str(s)]
    for sep in seps:
        nxt = []
        for p in out:
            nxt.extend(p.split(sep))
        out = nxt
    return [x.strip() for x in out if x.strip()]


def parse_pattern_token_sizes(pattern):
    # Example: "[VILMF] [HNQ] [DE]"
    sizes = []
    toks = split_multi(pattern, seps=" ")
    for tok in toks:
        if tok.startswith("[") and tok.endswith("]"):
            sizes.append(max(1, len(tok[1:-1])))
        else:
            sizes.append(1)
    return sizes


def extract_row_features(row):
    features = {}

    # Raw numeric columns (excluding obvious IDs/labels/leakage fields).
    drop_cols = {
        "rank",
        "query_row",
        "uid",
        "uniprot_id",
        "entry",
        "class",
        "mapped_by",
        "target_class",
        "matched_target_id",
        "matched_target_row_id",
        "query_tokens",
    }
    for col, val in row.items():
        lc = col.lower()
        if col in drop_cols:
            continue
        if "rank" in lc:
            continue
        if "apim" in lc or "eyfp" in lc:
            continue
        if lc in {"count_a", "count_c"}:
            continue
        v = maybe_float(val)
        if v is None:
            continue
        features[f"num__{col}"] = v

    # Geometry summaries from cluster-wise RMSD fields.
    pairdist_vals = []
    for col, val in row.items():
        if re.match(r"^cluster\d+_pairdist_rmsd$", col):
            v = maybe_float(val)
            if v is not None:
                pairdist_vals.append(v)
    if pairdist_vals:
        arr = np.array(pairdist_vals, dtype=float)
        features["geo__cluster_pairdist_count"] = float(arr.size)
        features["geo__cluster_pairdist_mean"] = float(arr.mean())
        features["geo__cluster_pairdist_min"] = float(arr.min())
        features["geo__cluster_pairdist_max"] = float(arr.max())
        features["geo__cluster_pairdist_std"] = float(arr.std())

    # Pattern degeneracy summaries from clusterN_pattern columns.
    degen_sizes = []
    for col, val in row.items():
        if re.match(r"^cluster\d+_pattern$", col):
            sizes = parse_pattern_token_sizes(val or "")
            degen_sizes.extend(sizes)
    if degen_sizes:
        arr = np.array(degen_sizes, dtype=float)
        features["pat__token_count"] = float(arr.size)
        features["pat__mean_allowed_size"] = float(arr.mean())
        features["pat__max_allowed_size"] = float(arr.max())
        features["pat__degenerate_frac"] = float(np.mean(arr > 1.0))

    # Chemistry summaries from selected residue columns (cluster1, cluster2, ...).
    aas = []
    residues = []
    for col, val in row.items():
        if not re.match(r"^cluster\d+$", col):
            continue
        for tok in split_multi(val, seps=","):
            m = re.match(r"^([A-Z])(\d+)$", tok)
            if not m:
                continue
            aa = m.group(1)
            residues.append(tok)
            aas.append(aa)
    if aas:
        n = len(aas)
        counts = Counter(aas)
        features["chem__residue_count"] = float(n)
        features["chem__residue_unique_count"] = float(len(set(residues)))
        for cname, aset in AA_CLASSES.items():
            c = sum(counts.get(a, 0) for a in aset)
            features[f"chem__frac_{cname}"] = c / n
            features[f"chem__count_{cname}"] = float(c)
        net = (
            sum(counts.get(a, 0) for a in AA_CLASSES["basic"])
            - sum(counts.get(a, 0) for a in AA_CLASSES["acidic"])
        )
        features["chem__net_basic_minus_acidic_per_res"] = net / n

    return features


def looks_like_summary(fieldnames):
    f = set(fieldnames or [])
    return f == {"metric", "value"} or (("metric" in f) and ("value" in f) and ("uniprot_id" not in f))


def looks_like_overlap_table(fieldnames):
    f = set(fieldnames or [])
    return ("class" in f) and ("mapped_by" in f)


def load_labels(path, id_col, label_col):
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            k = (row.get(id_col) or "").strip()
            if not k:
                continue
            out[k] = (row.get(label_col) or "").strip()
    return out


def build_dataset(files, labels_by_id, args):
    rows = []
    file_stats = []
    neg_set = set(parse_comma_list(args.negative_classes))

    for path in files:
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            fieldnames = r.fieldnames or []
            if looks_like_summary(fieldnames):
                continue
            if looks_like_overlap_table(fieldnames):
                continue
            if args.id_col not in fieldnames:
                continue

            used = 0
            skipped_unlabeled = 0
            skipped_class = 0
            for row in r:
                uid = (row.get(args.id_col) or "").strip()
                if not uid:
                    continue
                lbl = labels_by_id.get(uid, "")
                if not lbl:
                    skipped_unlabeled += 1
                    continue
                if lbl == args.positive_class:
                    y = 1
                elif lbl in neg_set:
                    y = 0
                else:
                    skipped_class += 1
                    continue

                feats = extract_row_features(row)
                if not feats:
                    continue
                rows.append(
                    {
                        "source_tsv": path,
                        "source_tag": os.path.splitext(os.path.basename(path))[0],
                        "uniprot_id": uid,
                        "label_class": lbl,
                        "y": y,
                        "features": feats,
                    }
                )
                used += 1

            file_stats.append(
                {
                    "source_tsv": path,
                    "used_rows": used,
                    "skipped_unlabeled": skipped_unlabeled,
                    "skipped_class": skipped_class,
                }
            )

    if not rows:
        raise RuntimeError("No labeled rows collected from query TSVs.")

    return rows, file_stats


def vectorize_rows(rows, args):
    feat_names = sorted({k for r in rows for k in r["features"].keys()})
    n = len(rows)
    p = len(feat_names)
    idx = {f: j for j, f in enumerate(feat_names)}
    X = np.full((n, p), np.nan, dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    groups = np.array([r["uniprot_id"] for r in rows], dtype=object)

    for i, r in enumerate(rows):
        for f, v in r["features"].items():
            X[i, idx[f]] = float(v)

    # Coverage + variance filtering.
    keep = []
    nonmissing = np.sum(np.isfinite(X), axis=0)
    for j in range(p):
        if nonmissing[j] < args.min_feature_count:
            continue
        if (nonmissing[j] / n) < args.min_feature_frac:
            continue
        col = X[:, j]
        vals = col[np.isfinite(col)]
        if vals.size < 2:
            continue
        if float(np.var(vals)) <= args.min_feature_variance:
            continue
        keep.append(j)

    if not keep:
        raise RuntimeError("No features left after filtering.")

    # Optional cap: choose highest coverage first.
    if args.max_features > 0 and len(keep) > args.max_features:
        keep = sorted(
            keep,
            key=lambda j: (nonmissing[j], float(np.var(X[np.isfinite(X[:, j]), j]))),
            reverse=True,
        )[: args.max_features]
        keep = sorted(keep)

    X = X[:, keep]
    feat_names = [feat_names[j] for j in keep]
    dedup_rows = []
    if args.dedup_enable and X.shape[1] > 1:
        X, feat_names, dedup_rows = deduplicate_features(X, feat_names, args)
    return X, y, groups, feat_names, dedup_rows


def _feature_priority(name, coverage, variance):
    # Prefer high-coverage/high-variance features, and avoid keeping
    # duplicate namespace variants (old_/new_) when equivalent columns exist.
    is_oldnew = int(name.startswith("num__old_") or name.startswith("num__new_"))
    return (coverage, variance, -is_oldnew, -len(name))


def deduplicate_features(X, feat_names, args):
    n, p = X.shape
    if p <= 1:
        return X, feat_names, []

    med, _, _ = fit_preprocessor(X)
    Ximp = np.where(np.isfinite(X), X, med)
    coverage = np.mean(np.isfinite(X), axis=0)
    variance = np.var(Ximp, axis=0)

    order = sorted(
        range(p),
        key=lambda j: _feature_priority(feat_names[j], float(coverage[j]), float(variance[j])),
        reverse=True,
    )

    kept = []
    dropped = {}
    reasons = {}
    corr_to = {}

    # Step 1: exact (or near-exact) duplicate collapse by max absolute diff.
    for j in order:
        if j in dropped:
            continue
        vec_j = Ximp[:, j]
        matched = None
        for k in kept:
            vec_k = Ximp[:, k]
            if vec_j.shape != vec_k.shape:
                continue
            max_abs = float(np.max(np.abs(vec_j - vec_k)))
            if max_abs <= args.dedup_equal_tol:
                matched = k
                break
        if matched is not None:
            dropped[j] = matched
            reasons[j] = "exact_duplicate"
            corr_to[j] = 1.0
        else:
            kept.append(j)

    # Step 2: high-correlation collapse among remaining features.
    kept2 = []
    for j in kept:
        vec_j = Ximp[:, j]
        norm_j = float(np.linalg.norm(vec_j))
        if norm_j <= 1e-15:
            # Constant features should have been filtered already; keep defensively.
            kept2.append(j)
            continue
        matched = None
        matched_corr = None
        for k in kept2:
            vec_k = Ximp[:, k]
            norm_k = float(np.linalg.norm(vec_k))
            if norm_k <= 1e-15:
                continue
            corr = float(np.dot(vec_j, vec_k) / (norm_j * norm_k))
            corr = max(-1.0, min(1.0, corr))
            if abs(corr) >= args.dedup_corr_threshold:
                matched = k
                matched_corr = corr
                break
        if matched is not None:
            dropped[j] = matched
            reasons[j] = "high_corr"
            corr_to[j] = matched_corr
        else:
            kept2.append(j)

    kept2 = sorted(kept2)
    out_X = X[:, kept2]
    out_names = [feat_names[j] for j in kept2]

    dedup_rows = []
    kept_set = set(kept2)
    for j in range(p):
        if j in kept_set:
            dedup_rows.append(
                {
                    "feature": feat_names[j],
                    "status": "kept",
                    "kept_feature": feat_names[j],
                    "reason": "",
                    "abs_corr_to_kept": "",
                    "coverage": float(coverage[j]),
                    "variance_imputed": float(variance[j]),
                }
            )
        else:
            k = dropped.get(j)
            dedup_rows.append(
                {
                    "feature": feat_names[j],
                    "status": "dropped",
                    "kept_feature": feat_names[k] if k is not None else "",
                    "reason": reasons.get(j, "dropped"),
                    "abs_corr_to_kept": abs(float(corr_to.get(j, float("nan"))))
                    if j in corr_to
                    else "",
                    "coverage": float(coverage[j]),
                    "variance_imputed": float(variance[j]),
                }
            )

    dedup_rows.sort(key=lambda r: (r["status"], r["feature"]))
    return out_X, out_names, dedup_rows


def fit_preprocessor(X):
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Ximp = np.where(np.isfinite(X), X, med)
    mu = Ximp.mean(axis=0)
    sd = Ximp.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return med, mu, sd


def transform_with_preprocessor(X, med, mu, sd):
    Ximp = np.where(np.isfinite(X), X, med)
    return (Ximp - mu) / sd


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def fit_logistic_enet(X, y, l1, l2, max_iter=3000, tol=1e-6):
    n, p = X.shape
    y = y.astype(float)
    pos = np.clip(y.mean(), 1e-4, 1.0 - 1e-4)
    b = float(math.log(pos / (1.0 - pos)))
    w = np.zeros(p, dtype=float)

    # Gradient Lipschitz upper bound for logistic + L2.
    try:
        smax = np.linalg.norm(X, ord=2)
        L = 0.25 * (smax ** 2) / max(1, n) + l2 + 1e-12
    except Exception:
        L = 1.0 + l2
    lr = 1.0 / L

    for it in range(1, max_iter + 1):
        z = X @ w + b
        p_hat = sigmoid(z)
        err = p_hat - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(np.mean(err))

        w_new = soft_threshold(w - lr * grad_w, lr * l1)
        b_new = b - lr * grad_b

        dw = np.max(np.abs(w_new - w)) if p else 0.0
        db = abs(b_new - b)
        w, b = w_new, b_new
        if max(dw, db) < tol:
            return w, b, it

    return w, b, max_iter


def predict_scores(X, w, b):
    return X @ w + b


def roc_auc(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n = y.size
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    sorted_s = s[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def make_group_folds(groups, n_splits, seed):
    uniq = sorted(set(groups.tolist()))
    if len(uniq) < 2:
        return []
    n_splits = max(2, min(n_splits, len(uniq)))
    rng = np.random.default_rng(seed)
    perm = uniq[:]
    rng.shuffle(perm)

    fold_groups = [set() for _ in range(n_splits)]
    for i, g in enumerate(perm):
        fold_groups[i % n_splits].add(g)
    return fold_groups


def make_group_fold_masks(groups, n_splits, seed):
    folds = make_group_folds(groups, n_splits, seed)
    masks = []
    for fg in folds:
        mask = np.array([g in fg for g in groups], dtype=bool)
        if np.any(mask):
            masks.append(mask)
    return masks


def make_stratified_row_fold_masks(y, n_splits, seed):
    y = np.asarray(y, dtype=int)
    pos = np.where(y == 1)[0].tolist()
    neg = np.where(y == 0)[0].tolist()
    if not pos or not neg:
        return []
    n_splits = max(2, min(n_splits, len(pos), len(neg)))
    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    fold_indices = [[] for _ in range(n_splits)]
    for i, ix in enumerate(pos):
        fold_indices[i % n_splits].append(ix)
    for i, ix in enumerate(neg):
        fold_indices[i % n_splits].append(ix)

    masks = []
    n = y.size
    for fi in fold_indices:
        if not fi:
            continue
        mask = np.zeros(n, dtype=bool)
        mask[np.array(fi, dtype=int)] = True
        masks.append(mask)
    return masks


def cross_validate_lambda(X, y, lam_total, alpha, fold_masks, args):
    l1 = lam_total * alpha
    l2 = lam_total * (1.0 - alpha)
    aucs = []

    for test_mask in fold_masks:
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
            continue
        y_tr = y[train_mask]
        y_te = y[test_mask]
        if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
            continue

        med, mu, sd = fit_preprocessor(X[train_mask])
        Xtr = transform_with_preprocessor(X[train_mask], med, mu, sd)
        Xte = transform_with_preprocessor(X[test_mask], med, mu, sd)

        w, b, _ = fit_logistic_enet(
            Xtr,
            y_tr,
            l1=l1,
            l2=l2,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        scr = predict_scores(Xte, w, b)
        auc = roc_auc(y_te, scr)
        if math.isfinite(auc):
            aucs.append(auc)

    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def bootstrap_stability(X, y, groups, feat_names, lam_total, alpha, args):
    l1 = lam_total * alpha
    l2 = lam_total * (1.0 - alpha)
    uniq = sorted(set(groups.tolist()))
    g2idx = defaultdict(list)
    for i, g in enumerate(groups.tolist()):
        g2idx[g].append(i)

    rng = np.random.default_rng(args.bootstrap_seed)
    p = X.shape[1]
    nz = np.zeros(p, dtype=int)
    coef_sum = np.zeros(p, dtype=float)
    sign_sum = np.zeros(p, dtype=float)
    oob_aucs = []
    n_fit = 0

    for _ in range(args.bootstrap):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        boot_idx = []
        seen = set()
        for g in draw:
            seen.add(g)
            boot_idx.extend(g2idx[g])
        boot_idx = np.array(boot_idx, dtype=int)
        yb = y[boot_idx]
        if len(set(yb.tolist())) < 2:
            continue

        med, mu, sd = fit_preprocessor(X[boot_idx])
        Xb = transform_with_preprocessor(X[boot_idx], med, mu, sd)
        w, b, _ = fit_logistic_enet(
            Xb,
            yb,
            l1=l1,
            l2=l2,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        n_fit += 1
        keep = np.abs(w) > args.coef_zero_threshold
        nz += keep.astype(int)
        coef_sum += w
        sign_sum += np.sign(w) * keep.astype(float)

        # OOB AUC at group level.
        oob_groups = [g for g in uniq if g not in seen]
        if oob_groups:
            oob_mask = np.array([g in set(oob_groups) for g in groups], dtype=bool)
            if np.any(oob_mask):
                yo = y[oob_mask]
                if len(set(yo.tolist())) >= 2:
                    Xo = transform_with_preprocessor(X[oob_mask], med, mu, sd)
                    s = predict_scores(Xo, w, b)
                    auc = roc_auc(yo, s)
                    if math.isfinite(auc):
                        oob_aucs.append(auc)

    rows = []
    for j, f in enumerate(feat_names):
        sel = nz[j]
        frac = (sel / n_fit) if n_fit else 0.0
        mean_coef = (coef_sum[j] / n_fit) if n_fit else 0.0
        sign_consistency = (abs(sign_sum[j]) / sel) if sel > 0 else 0.0
        rows.append(
            {
                "feature": f,
                "bootstrap_selected_count": int(sel),
                "bootstrap_selected_frac": float(frac),
                "bootstrap_mean_coef_std": float(mean_coef),
                "bootstrap_sign_consistency": float(sign_consistency),
            }
        )

    rows.sort(key=lambda r: (r["bootstrap_selected_frac"], abs(r["bootstrap_mean_coef_std"])), reverse=True)
    oob_auc_mean = float(np.mean(oob_aucs)) if oob_aucs else float("nan")
    return rows, n_fit, oob_auc_mean


def write_tsv(path, rows, headers):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build APIM feature matrix + regularized model + bootstrap stability."
    )
    ap.add_argument(
        "--query-glob",
        action="append",
        default=[],
        help='Input glob(s), can be repeated or comma-separated. Example: --query-glob "data/*candidate7*.tsv,data/hdryvppm*.tsv"',
    )
    ap.add_argument("--label-tsv", default="data/r3_mapped_ac_ge2_minus_r1r2_sanitized.tsv")
    ap.add_argument("--id-col", default="uniprot_id")
    ap.add_argument("--label-col", default="class")
    ap.add_argument("--positive-class", default="APIM_only")
    ap.add_argument("--negative-classes", default="Both,EYFP_only")
    ap.add_argument("--min-feature-count", type=int, default=20)
    ap.add_argument("--min-feature-frac", type=float, default=0.05)
    ap.add_argument("--min-feature-variance", type=float, default=1e-10)
    ap.add_argument("--max-features", type=int, default=300)
    ap.add_argument("--dedup-enable", action="store_true", default=True)
    ap.add_argument("--no-dedup-enable", dest="dedup_enable", action="store_false")
    ap.add_argument(
        "--dedup-corr-threshold",
        type=float,
        default=0.995,
        help="Drop one of two features when |corr| >= threshold after imputation.",
    )
    ap.add_argument(
        "--dedup-equal-tol",
        type=float,
        default=1e-12,
        help="Max abs difference to treat features as exact duplicates after imputation.",
    )
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=123)
    ap.add_argument("--elastic-net-alpha", type=float, default=0.8, help="1.0=l1, 0.0=l2")
    ap.add_argument(
        "--lambda-grid",
        default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1",
        help="Comma-separated total regularization strengths.",
    )
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap-seed", type=int, default=321)
    ap.add_argument("--coef-zero-threshold", type=float, default=1e-4)
    ap.add_argument("--out-prefix", default="data/apim_specificity_model")
    return ap.parse_args()


def main():
    args = parse_args()
    qglobs = parse_query_globs(args.query_glob) or ["data/*candidate*.tsv", "data/hdryvppm*.tsv"]
    files = []
    for pat in qglobs:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No files matched globs: {qglobs}")

    labels_by_id = load_labels(args.label_tsv, args.id_col, args.label_col)
    if not labels_by_id:
        raise SystemExit(f"No labels loaded from {args.label_tsv}")

    rows, file_stats = build_dataset(files, labels_by_id, args)
    X, y, groups, feat_names, dedup_rows = vectorize_rows(rows, args)

    n = len(rows)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print("samples", n, "pos", n_pos, "neg", n_neg, "features", len(feat_names))

    fold_masks = make_group_fold_masks(groups, args.cv_folds, args.cv_seed)
    if not fold_masks:
        fold_masks = make_stratified_row_fold_masks(y, args.cv_folds, args.cv_seed)
        cv_mode = "stratified_row_fallback"
    else:
        cv_mode = "group_by_uniprot_id"

    lam_grid = [float(x) for x in parse_comma_list(args.lambda_grid)]
    cv_rows = []

    def run_cv_rows(masks):
        rows_local = []
        best_local = None
        for lam in lam_grid:
            mean_auc, n_eval = cross_validate_lambda(
                X, y, lam_total=lam, alpha=args.elastic_net_alpha, fold_masks=masks, args=args
            )
            row = {
                "lambda_total": lam,
                "l1": lam * args.elastic_net_alpha,
                "l2": lam * (1.0 - args.elastic_net_alpha),
                "cv_mean_auc": mean_auc,
                "cv_folds_evaluated": n_eval,
            }
            rows_local.append(row)
            key = (
                mean_auc if math.isfinite(mean_auc) else float("-inf"),
                -lam,  # prefer stronger regularization on ties
            )
            if best_local is None or key > best_local[0]:
                best_local = (key, row)
        return rows_local, best_local

    cv_rows, best = run_cv_rows(fold_masks)
    if all(int(r["cv_folds_evaluated"]) == 0 for r in cv_rows):
        fallback_masks = make_stratified_row_fold_masks(y, args.cv_folds, args.cv_seed)
        if fallback_masks:
            cv_rows, best = run_cv_rows(fallback_masks)
            cv_mode = "stratified_row_fallback"

    best_row = best[1]
    best_lam = float(best_row["lambda_total"])
    print("best_lambda", best_lam, "cv_auc", best_row["cv_mean_auc"])

    # Fit final model on full data.
    med, mu, sd = fit_preprocessor(X)
    Xs = transform_with_preprocessor(X, med, mu, sd)
    w, b, n_iter = fit_logistic_enet(
        Xs,
        y,
        l1=best_lam * args.elastic_net_alpha,
        l2=best_lam * (1.0 - args.elastic_net_alpha),
        max_iter=args.max_iter,
        tol=args.tol,
    )
    train_auc = roc_auc(y, predict_scores(Xs, w, b))

    # Bootstrap stability.
    stab_rows, n_boot_fit, oob_auc_mean = bootstrap_stability(
        X, y, groups, feat_names, lam_total=best_lam, alpha=args.elastic_net_alpha, args=args
    )

    # Output paths.
    out_feature = f"{args.out_prefix}_feature_matrix.tsv"
    out_file_stats = f"{args.out_prefix}_file_stats.tsv"
    out_cv = f"{args.out_prefix}_cv.tsv"
    out_coef = f"{args.out_prefix}_coefficients.tsv"
    out_stab = f"{args.out_prefix}_stability.tsv"
    out_dedup = f"{args.out_prefix}_dedup.tsv"
    out_summary = f"{args.out_prefix}_summary.tsv"

    os.makedirs(os.path.dirname(out_feature) or ".", exist_ok=True)

    # Write feature matrix (raw values, blanks for missing).
    feat_headers = ["source_tsv", "source_tag", "uniprot_id", "label_class", "y"] + feat_names
    feat_rows = []
    for i, r in enumerate(rows):
        rr = {
            "source_tsv": r["source_tsv"],
            "source_tag": r["source_tag"],
            "uniprot_id": r["uniprot_id"],
            "label_class": r["label_class"],
            "y": r["y"],
        }
        fdict = r["features"]
        for fn in feat_names:
            v = fdict.get(fn)
            rr[fn] = "" if v is None else v
        feat_rows.append(rr)
    write_tsv(out_feature, feat_rows, feat_headers)

    # File-level stats.
    write_tsv(
        out_file_stats,
        file_stats,
        ["source_tsv", "used_rows", "skipped_unlabeled", "skipped_class"],
    )

    # CV table.
    write_tsv(
        out_cv,
        cv_rows,
        ["lambda_total", "l1", "l2", "cv_mean_auc", "cv_folds_evaluated"],
    )
    if dedup_rows:
        write_tsv(
            out_dedup,
            dedup_rows,
            [
                "feature",
                "status",
                "kept_feature",
                "reason",
                "abs_corr_to_kept",
                "coverage",
                "variance_imputed",
            ],
        )

    # Final coefficients (standardized feature scale).
    coef_rows = []
    for fn, c in zip(feat_names, w):
        coef_rows.append(
            {
                "feature": fn,
                "coef_std": float(c),
                "abs_coef_std": float(abs(c)),
            }
        )
    coef_rows.sort(key=lambda r: r["abs_coef_std"], reverse=True)
    write_tsv(out_coef, coef_rows, ["feature", "coef_std", "abs_coef_std"])

    # Merge stability with final coef.
    coef_map = {r["feature"]: r for r in coef_rows}
    merged_stab = []
    for r in stab_rows:
        c = coef_map.get(r["feature"], {})
        rr = dict(r)
        rr["final_coef_std"] = c.get("coef_std", 0.0)
        rr["final_abs_coef_std"] = c.get("abs_coef_std", 0.0)
        merged_stab.append(rr)
    write_tsv(
        out_stab,
        merged_stab,
        [
            "feature",
            "bootstrap_selected_count",
            "bootstrap_selected_frac",
            "bootstrap_mean_coef_std",
            "bootstrap_sign_consistency",
            "final_coef_std",
            "final_abs_coef_std",
        ],
    )

    summary_rows = [
        {"metric": "n_samples", "value": n},
        {"metric": "n_pos", "value": n_pos},
        {"metric": "n_neg", "value": n_neg},
        {"metric": "n_features", "value": len(feat_names)},
        {"metric": "n_features_dropped_by_dedup", "value": sum(1 for r in dedup_rows if r["status"] == "dropped")},
        {"metric": "n_unique_groups", "value": len(set(groups.tolist()))},
        {"metric": "cv_mode", "value": cv_mode},
        {"metric": "cv_folds_requested", "value": args.cv_folds},
        {"metric": "lambda_best", "value": best_lam},
        {"metric": "alpha", "value": args.elastic_net_alpha},
        {"metric": "cv_auc_best", "value": best_row["cv_mean_auc"]},
        {"metric": "train_auc", "value": train_auc},
        {"metric": "n_iter_final", "value": n_iter},
        {"metric": "bootstrap_requested", "value": args.bootstrap},
        {"metric": "bootstrap_models_fit", "value": n_boot_fit},
        {"metric": "bootstrap_oob_auc_mean", "value": oob_auc_mean},
        {"metric": "feature_matrix_tsv", "value": out_feature},
        {"metric": "file_stats_tsv", "value": out_file_stats},
        {"metric": "cv_tsv", "value": out_cv},
        {"metric": "coefficients_tsv", "value": out_coef},
        {"metric": "stability_tsv", "value": out_stab},
        {"metric": "dedup_tsv", "value": out_dedup if dedup_rows else ""},
    ]
    write_tsv(out_summary, summary_rows, ["metric", "value"])

    print("wrote", out_feature)
    print("wrote", out_file_stats)
    print("wrote", out_cv)
    print("wrote", out_coef)
    print("wrote", out_stab)
    if dedup_rows:
        print("wrote", out_dedup)
    print("wrote", out_summary)


if __name__ == "__main__":
    main()
