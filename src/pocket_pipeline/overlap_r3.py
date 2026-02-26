#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Alias-aware overlap wrapper against the R3 mapped pulldown universe."
    )
    ap.add_argument("--query-tsv", required=True, help="Input query TSV (must contain uniprot_id/accession/entry tokens).")
    ap.add_argument("--out", default=None, help="Output matched-target TSV. Default: data/<query>_overlap_r3_robust.tsv")
    ap.add_argument("--summary-out", default=None, help="Output summary TSV. Default: data/<query>_overlap_r3_robust_summary.tsv")
    ap.add_argument("--target-tsv", default="data/r3_mapped_ac_ge2_minus_r1r2_sanitized.tsv")
    ap.add_argument("--accession-map", default="data/accession_to_entry_map.tsv")
    ap.add_argument("--r3-mapping-tsv", default="data/r3_full_mapping_expanded_to_ecolibatch.tsv")
    ap.add_argument("--target-id-col", default="uniprot_id")
    ap.add_argument("--query-id-cols", default="uniprot_id,accession,entry,Entry,Entry Name")
    ap.add_argument("--target-class-col", default="class")
    args = ap.parse_args()

    q = Path(args.query_tsv)
    stem = q.stem
    out = args.out or f"data/{stem}_overlap_r3_robust.tsv"
    summary = args.summary_out or f"data/{stem}_overlap_r3_robust_summary.tsv"

    cmd = [
        "python3", "src/pocket_pipeline/robust_overlap.py",
        "--query-tsv", args.query_tsv,
        "--target-tsv", args.target_tsv,
        "--accession-map", args.accession_map,
        "--r3-mapping-tsv", args.r3_mapping_tsv,
        "--out", out,
        "--summary-out", summary,
        "--target-id-col", args.target_id_col,
        "--query-id-cols", args.query_id_cols,
        "--target-class-col", args.target_class_col,
    ]
    r = subprocess.run(cmd)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
