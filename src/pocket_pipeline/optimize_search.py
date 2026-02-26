#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import subprocess
import tempfile
from pathlib import Path


def parse_values(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return []
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid range spec: {spec}")
        start, stop, step = map(float, parts)
        vals = []
        x = start
        if step <= 0:
            raise ValueError("step must be > 0")
        while x <= stop + (step * 1e-9):
            vals.append(round(x, 6))
            x += step
        return vals
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def load_list_arg(single_value: str, file_path: str):
    vals = []
    if single_value:
        vals.append(single_value.strip())
    if file_path:
        with open(file_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                vals.append(s)
    return vals


def read_summary(path: Path):
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            out[row["metric"]] = row["value"]
    return out


def gi(d, key):
    try:
        return int(float(d.get(key, 0) or 0))
    except Exception:
        return 0


def fmt(x):
    return f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser(description="Grid search optimizer for pairdist_hardcut motif queries.")
    ap.add_argument("--ecoli-batch", default="src/ecoli_batch")
    ap.add_argument("--template-entry", default="DPO3B_ECOLI")
    ap.add_argument("--clusters-template", default=None,
                    help='Single template spec, e.g. "H175 | D173 | R176 | Y323"')
    ap.add_argument("--clusters-template-file", default=None,
                    help="File with one --clusters-template per line")
    ap.add_argument("--clusters-pattern", default=None,
                    help='Single pattern spec, e.g. "[HNQ] | [DE] | [RK] | [FY]"')
    ap.add_argument("--clusters-pattern-file", default=None,
                    help="File with one --clusters-pattern per line")
    ap.add_argument("--rmsd-values", default="0.5:4.0:0.5",
                    help='Either "start:stop:step" or comma list')
    ap.add_argument("--tol-values", default="0.5:10.0:0.5",
                    help='Either "start:stop:step" or comma list')
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max-candidates-per-cluster", type=int, default=500)
    ap.add_argument("--only-pulldown", action="store_true")
    ap.add_argument("--pulldown", default="data/r3_mapped_ac_ge2_minus_r1r2_sanitized.tsv")
    ap.add_argument("--pulldown-mapping-tsv", default="data/r3_full_mapping_expanded_to_ecolibatch.tsv")
    ap.add_argument("--accession-map", default="data/accession_to_entry_map.tsv")
    ap.add_argument("--min-overlap-targets", type=int, default=20)
    ap.add_argument("--min-apim-only", type=int, default=0)
    ap.add_argument("--eyfp-weight", type=float, default=2.0)
    ap.add_argument("--both-weight", type=float, default=0.0,
                    help="Optional positive weight for Both fraction")
    ap.add_argument("--out", default="data/optimize_search_ranked.tsv")
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    templates = load_list_arg(args.clusters_template, args.clusters_template_file)
    patterns = load_list_arg(args.clusters_pattern, args.clusters_pattern_file)
    if not templates:
        raise SystemExit("Provide --clusters-template or --clusters-template-file")
    if not patterns:
        raise SystemExit("Provide --clusters-pattern or --clusters-pattern-file")
    rmsd_vals = parse_values(args.rmsd_values)
    tol_vals = parse_values(args.tol_values)
    if not rmsd_vals or not tol_vals:
        raise SystemExit("No rmsd/tol values parsed")

    run_dir = Path(tempfile.mkdtemp(prefix="optimize_search_", dir="/tmp"))
    if args.keep_intermediate:
        run_dir = Path("data/optimize_runs")
        run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rid = 0
    combos = list(itertools.product(range(len(templates)), range(len(patterns)), rmsd_vals, tol_vals))
    total = len(combos)
    for ti, pi, rmsd, tol in combos:
        rid += 1
        tmpl = templates[ti]
        patt = patterns[pi]
        run_tag = f"t{ti+1}_p{pi+1}_r{rmsd:g}_tol{tol:g}"
        query_out = run_dir / f"{run_tag}.tsv"
        overlap_out = run_dir / f"{run_tag}_overlap.tsv"
        summary_out = run_dir / f"{run_tag}_summary.tsv"

        cmd = [
            "python3", "src/pocket_pipeline/pairdist_hardcut.py",
            "--ecoli-batch", args.ecoli_batch,
            "--template-entry", args.template_entry,
            "--clusters-template", tmpl,
            "--clusters-pattern", patt,
            "--inter-cluster-rmsd-cutoff", str(rmsd),
            "--inter-pair-tol", str(tol),
            "--max-candidates-per-cluster", str(args.max_candidates_per_cluster),
            "--jobs", str(args.jobs),
            "--pulldown", args.pulldown,
            "--pulldown-mapping-tsv", args.pulldown_mapping_tsv,
            "--accession-map", args.accession_map,
            "--out", str(query_out),
        ]
        if args.only_pulldown:
            cmd.append("--only-pulldown")
        pr = subprocess.run(cmd, capture_output=True, text=True)
        if pr.returncode != 0:
            rows.append({
                "run_id": rid, "status": "pairdist_error", "template_idx": ti + 1, "pattern_idx": pi + 1,
                "rmsd": fmt(rmsd), "tol": fmt(tol), "error": (pr.stderr or pr.stdout).strip()[:500],
            })
            continue

        cmd2 = [
            "python3", "src/pocket_pipeline/robust_overlap.py",
            "--query-tsv", str(query_out),
            "--target-tsv", args.pulldown,
            "--accession-map", args.accession_map,
            "--r3-mapping-tsv", args.pulldown_mapping_tsv,
            "--out", str(overlap_out),
            "--summary-out", str(summary_out),
        ]
        pr2 = subprocess.run(cmd2, capture_output=True, text=True)
        if pr2.returncode != 0:
            rows.append({
                "run_id": rid, "status": "overlap_error", "template_idx": ti + 1, "pattern_idx": pi + 1,
                "rmsd": fmt(rmsd), "tol": fmt(tol), "error": (pr2.stderr or pr2.stdout).strip()[:500],
            })
            continue

        sm = read_summary(summary_out)
        ov = gi(sm, "overlap_targets")
        apim = gi(sm, "class_APIM_only")
        both = gi(sm, "class_Both")
        eyfp = gi(sm, "class_EYFP_only")
        apim_frac = (apim / ov) if ov else 0.0
        both_frac = (both / ov) if ov else 0.0
        eyfp_frac = (eyfp / ov) if ov else 0.0
        score = apim_frac + (args.both_weight * both_frac) - (args.eyfp_weight * eyfp_frac)
        pass_filters = int(ov >= args.min_overlap_targets and apim >= args.min_apim_only)

        rows.append({
            "run_id": rid,
            "status": "ok",
            "template_idx": ti + 1,
            "pattern_idx": pi + 1,
            "template": tmpl,
            "pattern": patt,
            "rmsd": fmt(rmsd),
            "tol": fmt(tol),
            "query_rows": gi(sm, "query_rows"),
            "overlap_targets": ov,
            "APIM_only": apim,
            "Both": both,
            "EYFP_only": eyfp,
            "APIM_only_frac": fmt(apim_frac),
            "Both_frac": fmt(both_frac),
            "EYFP_only_frac": fmt(eyfp_frac),
            "score": fmt(score),
            "passes_filters": pass_filters,
            "query_tsv": str(query_out),
            "overlap_tsv": str(overlap_out),
            "summary_tsv": str(summary_out),
        })
        print(f"[{rid}/{total}] t{ti+1} p{pi+1} rmsd={rmsd:g} tol={tol:g} "
              f"ov={ov} apim={apim} both={both} eyfp={eyfp} score={score:.4f}")

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: (
        int(r.get("passes_filters", 0)),
        float(r.get("score", -1e9)),
        int(r.get("APIM_only", 0)),
        -int(r.get("EYFP_only", 0)),
        int(r.get("overlap_targets", 0)),
    ), reverse=True)
    errors = [r for r in rows if r.get("status") != "ok"]
    out_rows = ok + errors

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = []
    for r in out_rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("wrote", out_path)
    if ok:
        b = ok[0]
        print("best",
              f"template_idx={b['template_idx']}",
              f"pattern_idx={b['pattern_idx']}",
              f"rmsd={b['rmsd']}",
              f"tol={b['tol']}",
              f"score={b['score']}",
              f"APIM_only={b['APIM_only']}",
              f"EYFP_only={b['EYFP_only']}",
              f"overlap={b['overlap_targets']}")


if __name__ == "__main__":
    main()
