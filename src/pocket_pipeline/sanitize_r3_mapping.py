#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

ECO_TOKENS = (
    "_ECOLI",
    "_ECOLX",
    "ESCHERICHIA COLI",
    "ORGANISM_TAXID: 562",
)

RABBIT_TOKENS = (
    "_RABIT",
    "_RABBIT",
    "ORYCTOLAGUS CUNICULUS",
    "ORGANISM_TAXID: 9986",
)


def load_accession_map(path: Path):
    out = {}
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            acc = (row.get("uniprot_accession") or "").strip()
            if not acc:
                continue
            entries = []
            pe = (row.get("primary_entry") or "").strip()
            if pe:
                entries.append(pe)
            for x in (row.get("all_entries") or "").split(";"):
                x = x.strip()
                if x:
                    entries.append(x)
            # stable unique order
            seen = set()
            uniq = []
            for e in entries:
                if e not in seen:
                    uniq.append(e)
                    seen.add(e)
            out[acc] = uniq
    return out


def load_mapped_tokens(path: Path):
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            uid = (row.get("uniprot_id") or "").strip()
            tok = (row.get("mapped_token") or "").strip()
            if uid and tok:
                out[uid] = tok
    return out


def read_pdb_header_text(pdb_path: Path, max_lines: int = 180):
    if not pdb_path.exists() or not pdb_path.is_file():
        return ""
    lines = []
    try:
        with pdb_path.open("r", errors="ignore") as f:
            for i, line in enumerate(f):
                lines.append(line.strip())
                if i + 1 >= max_lines:
                    break
    except Exception:
        return ""
    return "\n".join(lines).upper()


def classify_structure_species(pdb_header_upper: str):
    has_eco = any(tok in pdb_header_upper for tok in ECO_TOKENS)
    has_rabbit = any(tok in pdb_header_upper for tok in RABBIT_TOKENS)
    if has_eco and not has_rabbit:
        return "ecoli"
    if has_rabbit and not has_eco:
        return "rabbit"
    if has_eco and has_rabbit:
        return "mixed"
    return "unknown"


def choose_entry_for_accession(acc: str, entry_candidates, ecoli_batch_dir: Path):
    best = None
    best_rank = -1
    best_meta = {}

    for entry in entry_candidates:
        d = ecoli_batch_dir / entry
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        pdb = Path(meta.get("pdb", "")) if meta.get("pdb") else Path("")
        header = read_pdb_header_text(pdb)
        species = classify_structure_species(header)

        # Rank: ecoli > mixed > unknown > rabbit
        rank = {
            "ecoli": 4,
            "mixed": 3,
            "unknown": 2,
            "rabbit": 1,
        }.get(species, 0)

        if rank > best_rank:
            best_rank = rank
            best = entry
            best_meta = {
                "pdb": str(pdb),
                "species": species,
                "entry": entry,
                "header_has_eco": int(any(tok in header for tok in ECO_TOKENS)),
                "header_has_rabbit": int(any(tok in header for tok in RABBIT_TOKENS)),
            }

    if best is None:
        return None, {"species": "no_structure"}

    return best, best_meta


def main():
    ap = argparse.ArgumentParser(description="Sanitize R3 mapping by validating backing structure species.")
    ap.add_argument("--in-tsv", default="data/r3_mapped_ac_ge2_sets.tsv")
    ap.add_argument("--accession-map", default="data/accession_to_entry_map.tsv")
    ap.add_argument("--ecoli-batch", default="src/ecoli_batch")
    ap.add_argument("--mapping-audit", default="data/r3_mapped_ac_ge2_mapping_audit_full.tsv")
    ap.add_argument("--out-tsv", default="data/r3_mapped_ac_ge2_sets_sanitized.tsv")
    ap.add_argument("--audit-tsv", default="data/r3_mapped_ac_ge2_sets_sanitized_audit.tsv")
    args = ap.parse_args()

    in_tsv = Path(args.in_tsv)
    accession_map_path = Path(args.accession_map)
    ecoli_batch_dir = Path(args.ecoli_batch)
    mapping_audit = Path(args.mapping_audit)
    out_tsv = Path(args.out_tsv)
    audit_tsv = Path(args.audit_tsv)

    acc_to_entries = load_accession_map(accession_map_path)
    uid_to_mapped_token = load_mapped_tokens(mapping_audit)

    with in_tsv.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    kept = []
    audit = []
    for row in rows:
        acc = (row.get("uniprot_id") or "").strip()
        entries = []
        mapped_tok = uid_to_mapped_token.get(acc, "")
        if mapped_tok:
            entries.append(mapped_tok)
        entries.extend(acc_to_entries.get(acc, []))
        seen = set()
        entries = [x for x in entries if not (x in seen or seen.add(x))]
        chosen_entry, meta = choose_entry_for_accession(acc, entries, ecoli_batch_dir)
        species = meta.get("species", "no_structure")

        # Keep by default; only drop when backing structure explicitly points to rabbit.
        status = "drop" if species == "rabbit" else "keep"
        rec = dict(row)
        rec["sanitized_species"] = species
        rec["sanitized_entry"] = chosen_entry or ""
        rec["sanitized_pdb"] = meta.get("pdb", "")
        rec["sanitized_status"] = status

        audit.append(rec)
        if status == "keep":
            kept.append(row)

    if rows:
        out_fields = list(rows[0].keys())
        with out_tsv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t")
            w.writeheader()
            w.writerows(kept)

    if audit:
        with audit_tsv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(audit)

    print("input_rows", len(rows))
    print("kept_rows", len(kept))
    print("dropped_rows", len(rows) - len(kept))
    print("wrote", out_tsv)
    print("wrote", audit_tsv)


if __name__ == "__main__":
    main()
