#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


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


def load_accession_map(path):
    acc_to_entries = defaultdict(set)
    entry_to_acc = defaultdict(set)
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            acc = (row.get("uniprot_accession") or "").strip()
            if not acc:
                continue
            pe = (row.get("primary_entry") or "").strip()
            entries = []
            if pe:
                entries.append(pe)
            entries.extend(split_multi(row.get("all_entries"), seps=";"))
            for e in entries:
                acc_to_entries[acc].add(e)
                entry_to_acc[e].add(acc)
    return acc_to_entries, entry_to_acc


def preferred_accession(accs):
    if not accs:
        return ""
    six = sorted([a for a in accs if len(a) == 6])
    if six:
        return six[0]
    return sorted(accs)[0]


def canonical_uniprot_id(uid, acc_to_entries, entry_to_acc):
    uid = (uid or "").strip().upper()
    if not uid:
        return ""
    exp = expand_tokens({uid}, acc_to_entries, entry_to_acc, rounds=2)
    accs = {t for t in exp if t in acc_to_entries}
    if accs:
        return preferred_accession(accs)
    if uid in entry_to_acc and entry_to_acc[uid]:
        return preferred_accession({a.strip().upper() for a in entry_to_acc[uid] if a})
    return uid


def expand_tokens(tokens, acc_to_entries, entry_to_acc, rounds=2):
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


def build_target_index(target_rows, target_id_col, acc_to_entries, entry_to_acc, mapping_rows=None):
    token_to_targets = defaultdict(set)
    target_to_tokens = defaultdict(set)

    # Optional R3 mapping index: primary id -> seed candidates
    map_seed = {}
    if mapping_rows is not None:
        for mr in mapping_rows:
            pid = canonical_uniprot_id(mr.get("uniprot_id"), acc_to_entries, entry_to_acc)
            if not pid:
                continue
            seeds = split_multi(mr.get("seed_candidates"), seps="|;,")
            map_seed[pid] = seeds

    for tr in target_rows:
        tid = canonical_uniprot_id(tr.get(target_id_col), acc_to_entries, entry_to_acc)
        if not tid:
            continue
        seeds = {tid}
        seeds |= set(split_multi(tr.get("mapped_by"), seps="|;,"))
        seeds |= set(map_seed.get(tid, []))
        expanded = expand_tokens(seeds, acc_to_entries, entry_to_acc, rounds=2)
        target_to_tokens[tid] |= expanded
        for tok in expanded:
            token_to_targets[tok].add(tid)
    return token_to_targets, target_to_tokens


def parse_args():
    ap = argparse.ArgumentParser(description="Robust overlap with accession/entry alias expansion.")
    ap.add_argument("--query-tsv", required=True)
    ap.add_argument("--target-tsv", required=True)
    ap.add_argument("--accession-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--target-id-col", default="uniprot_id")
    ap.add_argument("--query-id-cols", default="uniprot_id,accession,entry")
    ap.add_argument("--target-class-col", default="class")
    ap.add_argument("--r3-mapping-tsv", default="", help="Optional mapping table with uniprot_id + seed_candidates")
    return ap.parse_args()


def main():
    args = parse_args()
    query_id_cols = [x.strip() for x in args.query_id_cols.split(",") if x.strip()]

    acc_to_entries, entry_to_acc = load_accession_map(args.accession_map)

    with open(args.target_tsv) as f:
        target_rows = list(csv.DictReader(f, delimiter="\t"))
    with open(args.query_tsv) as f:
        query_rows = list(csv.DictReader(f, delimiter="\t"))

    mapping_rows = None
    if args.r3_mapping_tsv:
        with open(args.r3_mapping_tsv) as f:
            mapping_rows = list(csv.DictReader(f, delimiter="\t"))

    token_to_targets, _ = build_target_index(
        target_rows,
        args.target_id_col,
        acc_to_entries,
        entry_to_acc,
        mapping_rows=mapping_rows,
    )

    # Build lookup by both raw and canonical target IDs so canonical match IDs
    # always resolve back to the underlying target row.
    target_by_id = {}
    for r in target_rows:
        raw = (r.get(args.target_id_col) or "").strip()
        if not raw:
            continue
        target_by_id.setdefault(raw, r)
        canon = canonical_uniprot_id(raw, acc_to_entries, entry_to_acc)
        if canon:
            target_by_id.setdefault(canon, r)

    matched_targets = set()
    details = []
    for i, qr in enumerate(query_rows, start=1):
        seeds = set()
        for col in query_id_cols:
            if col in qr:
                seeds |= set(split_multi(qr.get(col), seps=";|,"))
        if not seeds:
            continue
        # Canonicalize query seeds so alias forms converge to the same UID.
        canon = {canonical_uniprot_id(s, acc_to_entries, entry_to_acc) for s in seeds}
        seeds |= {c for c in canon if c}
        expanded = expand_tokens(seeds, acc_to_entries, entry_to_acc, rounds=2)
        hits = set()
        for tok in expanded:
            hits |= token_to_targets.get(tok, set())
        for tid in sorted(hits):
            tr = target_by_id.get(tid, {})
            raw_tid = (tr.get(args.target_id_col) or "").strip() if tr else ""
            matched_targets.add(tid)
            details.append(
                {
                    "query_row": i,
                    "query_tokens": "|".join(sorted(seeds)),
                    "matched_target_id": tid,
                    "matched_target_row_id": raw_tid,
                    "target_class": tr.get(args.target_class_col, ""),
                }
            )

    matched_target_rows = []
    seen_raw_ids = set()
    for tid in sorted(matched_targets):
        tr = target_by_id.get(tid)
        if not tr:
            continue
        raw_id = (tr.get(args.target_id_col) or "").strip()
        if raw_id and raw_id in seen_raw_ids:
            continue
        if raw_id:
            seen_raw_ids.add(raw_id)
        matched_target_rows.append(tr)
    with open(args.out, "w", newline="") as f:
        if matched_target_rows:
            w = csv.DictWriter(f, fieldnames=list(matched_target_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(matched_target_rows)
        else:
            w = csv.writer(f, delimiter="\t")
            w.writerow([args.target_id_col])

    class_counts = Counter((r.get(args.target_class_col) or "") for r in matched_target_rows)
    with open(args.summary_out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["metric", "value"])
        w.writerow(["query_rows", len(query_rows)])
        w.writerow(["target_tsv", args.target_tsv])
        target_base = Path(args.target_tsv).name.lower()
        w.writerow(["target_universe", "sanitized" if "sanitized" in target_base else "custom_or_unsanitized"])
        w.writerow(["target_rows", len(target_rows)])
        w.writerow(["overlap_targets", len(matched_target_rows)])
        for k in sorted(class_counts):
            w.writerow([f"class_{k}", class_counts[k]])

    details_out = args.summary_out.replace(".tsv", "_details.tsv")
    with open(details_out, "w", newline="") as f:
        if details:
            w = csv.DictWriter(f, fieldnames=list(details[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(details)
        else:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query_row", "query_tokens", "matched_target_id", "matched_target_row_id", "target_class"])

    print("overlap_targets", len(matched_target_rows))
    print("class_counts", dict(class_counts))
    print("wrote", args.out)
    print("wrote", args.summary_out)
    print("wrote", details_out)


if __name__ == "__main__":
    main()
