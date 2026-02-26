#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import zipfile
from collections import defaultdict
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


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


def preferred_accession(accs):
    if not accs:
        return ""
    six = sorted([a for a in accs if len(a) == 6])
    if six:
        return six[0]
    return sorted(accs)[0]


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


def load_accession_map(path):
    acc_to_entries = defaultdict(set)
    entry_to_acc = defaultdict(set)
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            acc = (row.get("uniprot_accession") or "").strip().upper()
            if not acc:
                continue
            pe = (row.get("primary_entry") or "").strip().upper()
            entries = []
            if pe:
                entries.append(pe)
            entries.extend([(x or "").strip().upper() for x in split_multi(row.get("all_entries"), seps=";")])
            entries = [e for e in entries if e]
            for e in entries:
                acc_to_entries[acc].add(e)
                entry_to_acc[e].add(acc)
    return acc_to_entries, entry_to_acc


def load_proteome_ids(ecoli_batch):
    uid_to_entries = defaultdict(set)
    for meta_path in glob.glob(os.path.join(ecoli_batch, "*", "meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            uid = os.path.splitext(os.path.basename(meta.get("pdb", "")))[0].strip().upper()
            entry = os.path.basename(os.path.dirname(meta_path)).strip().upper()
            if uid:
                uid_to_entries[uid].add(entry)
        except Exception:
            continue
    return uid_to_entries


def load_shared_strings(zf):
    out = []
    key = "xl/sharedStrings.xml"
    if key not in zf.namelist():
        return out
    root = ET.fromstring(zf.read(key))
    for si in root.findall(f"{{{NS_MAIN}}}si"):
        txt = "".join(t.text or "" for t in si.findall(f".//{{{NS_MAIN}}}t"))
        out.append(txt)
    return out


def col_to_idx(col):
    n = 0
    for ch in col:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def cell_value(cell, shared_strings):
    ctype = cell.attrib.get("t")
    if ctype == "inlineStr":
        is_node = cell.find(f"{{{NS_MAIN}}}is")
        if is_node is None:
            return ""
        return "".join(t.text or "" for t in is_node.findall(f".//{{{NS_MAIN}}}t"))
    v = cell.find(f"{{{NS_MAIN}}}v")
    if v is None:
        return ""
    raw = v.text or ""
    if ctype == "s":
        try:
            return shared_strings[int(raw)]
        except Exception:
            return raw
    return raw


def workbook_sheet_map(zf):
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {}
    for rel in rels.findall(f"{{{NS_REL_PKG}}}Relationship"):
        rid_to_target[rel.attrib.get("Id", "")] = rel.attrib.get("Target", "")
    out = {}
    for s in wb.findall(f".//{{{NS_MAIN}}}sheet"):
        name = s.attrib.get("name", "")
        rid = s.attrib.get(f"{{{NS_REL_DOC}}}id", "")
        tgt = rid_to_target.get(rid, "")
        if not tgt:
            continue
        out[name] = f"xl/{tgt}".replace("//", "/")
    return out


def read_sheet_rows(zf, sheet_path, shared_strings):
    root = ET.fromstring(zf.read(sheet_path))
    out = []
    for row in root.findall(f".//{{{NS_MAIN}}}row"):
        ridx = int(row.attrib.get("r", "0") or 0)
        cells = {}
        for c in row.findall(f"{{{NS_MAIN}}}c"):
            ref = c.attrib.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            if not col:
                continue
            cidx = col_to_idx(col)
            cells[cidx] = cell_value(c, shared_strings)
        out.append((ridx, cells))
    return out


def collect_sheet_tokens(zf, sheet_path, shared_strings, wanted_cols):
    rows = read_sheet_rows(zf, sheet_path, shared_strings)
    if not rows:
        return []
    header_cells = rows[0][1]
    header = {idx: (val or "").strip() for idx, val in header_cells.items()}
    wanted_set = {x.strip() for x in wanted_cols if x.strip()}
    chosen_idx = [idx for idx, name in header.items() if name in wanted_set]
    if not chosen_idx:
        raise RuntimeError(f"No requested ID columns found in sheet {sheet_path}: {sorted(wanted_set)}")

    tokens = []
    for ridx, cells in rows[1:]:
        for idx in chosen_idx:
            val = cells.get(idx, "")
            for tok in split_multi(val, seps=";|,"):
                t = (tok or "").strip().upper()
                if t:
                    tokens.append((ridx, t))
    return tokens


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Build a proteome TSV with all IDs overlapping RAW DATA R1/R2 "
            "from the B-clamp IP workbook removed."
        )
    )
    ap.add_argument("--xlsx", default="IPBC_2_experiments.xlsx")
    ap.add_argument("--sheet-r1", default="RAW DATA R1")
    ap.add_argument("--sheet-r2", default="RAW DATA R2")
    ap.add_argument("--id-cols", default="Protein IDs,Majority protein IDs")
    ap.add_argument("--accession-map", default="data/accession_to_entry_map.tsv")
    ap.add_argument("--proteome-dir", default="src/ecoli_batch")
    ap.add_argument("--out", default="data/proteome_minus_r1r2_bclamp_ip.tsv")
    ap.add_argument("--removed-out", default="data/proteome_removed_by_r1r2_bclamp_ip.tsv")
    ap.add_argument("--summary-out", default="data/proteome_minus_r1r2_bclamp_ip_summary.tsv")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.xlsx):
        raise SystemExit(f"xlsx not found: {args.xlsx}")
    if not os.path.exists(args.accession_map):
        raise SystemExit(f"accession map not found: {args.accession_map}")
    if not os.path.isdir(args.proteome_dir):
        raise SystemExit(f"proteome dir not found: {args.proteome_dir}")

    id_cols = [x.strip() for x in args.id_cols.split(",") if x.strip()]
    acc_to_entries, entry_to_acc = load_accession_map(args.accession_map)
    uid_to_entries = load_proteome_ids(args.proteome_dir)
    proteome_ids = set(uid_to_entries.keys())

    with zipfile.ZipFile(args.xlsx) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_map = workbook_sheet_map(zf)
        if args.sheet_r1 not in sheet_map:
            raise SystemExit(f"sheet not found: {args.sheet_r1}")
        if args.sheet_r2 not in sheet_map:
            raise SystemExit(f"sheet not found: {args.sheet_r2}")

        r1_tokens = collect_sheet_tokens(zf, sheet_map[args.sheet_r1], shared_strings, id_cols)
        r2_tokens = collect_sheet_tokens(zf, sheet_map[args.sheet_r2], shared_strings, id_cols)

    r1_uids = set()
    r2_uids = set()
    uid_source = defaultdict(lambda: {"R1": 0, "R2": 0})

    for _row, tok in r1_tokens:
        uid = canonical_uniprot_id(tok, acc_to_entries, entry_to_acc)
        if uid:
            r1_uids.add(uid)
            uid_source[uid]["R1"] += 1

    for _row, tok in r2_tokens:
        uid = canonical_uniprot_id(tok, acc_to_entries, entry_to_acc)
        if uid:
            r2_uids.add(uid)
            uid_source[uid]["R2"] += 1

    r1r2_union = r1_uids | r2_uids
    removed = sorted(r1r2_union & proteome_ids)
    filtered = sorted(proteome_ids - set(removed))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["uniprot_id", "entry_candidates"])
        for uid in filtered:
            w.writerow([uid, ";".join(sorted(uid_to_entries.get(uid, set())))])

    with open(args.removed_out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["uniprot_id", "entry_candidates", "in_R1", "in_R2", "source_count_R1", "source_count_R2"])
        for uid in removed:
            w.writerow(
                [
                    uid,
                    ";".join(sorted(uid_to_entries.get(uid, set()))),
                    1 if uid in r1_uids else 0,
                    1 if uid in r2_uids else 0,
                    uid_source[uid]["R1"],
                    uid_source[uid]["R2"],
                ]
            )

    with open(args.summary_out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["metric", "value"])
        w.writerow(["xlsx", args.xlsx])
        w.writerow(["sheet_r1", args.sheet_r1])
        w.writerow(["sheet_r2", args.sheet_r2])
        w.writerow(["proteome_total", len(proteome_ids)])
        w.writerow(["r1_unique_ids", len(r1_uids)])
        w.writerow(["r2_unique_ids", len(r2_uids)])
        w.writerow(["r1r2_union_unique_ids", len(r1r2_union)])
        w.writerow(["overlap_with_proteome", len(removed)])
        w.writerow(["overlap_r1_only", len((r1_uids - r2_uids) & proteome_ids)])
        w.writerow(["overlap_r2_only", len((r2_uids - r1_uids) & proteome_ids)])
        w.writerow(["overlap_r1_and_r2", len((r1_uids & r2_uids) & proteome_ids)])
        w.writerow(["proteome_minus_r1r2_total", len(filtered)])

    print("wrote", args.out)
    print("wrote", args.removed_out)
    print("wrote", args.summary_out)
    print("proteome_total", len(proteome_ids))
    print("removed", len(removed))
    print("remaining", len(filtered))


if __name__ == "__main__":
    main()
