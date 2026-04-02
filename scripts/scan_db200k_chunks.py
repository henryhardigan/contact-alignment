import argparse
import heapq
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import _db200k_cli
except ModuleNotFoundError:
    from examples import _db200k_cli

from contact_alignment import db200k_scan


def get_parser():
    parser = argparse.ArgumentParser(
        description="Scan multiple FASTA chunk files with a DB200K-derived query profile."
    )
    _db200k_cli.add_profile_args(parser)
    parser.add_argument(
        "--chunks-dir",
        type=str,
        required=True,
        help="Directory containing FASTA chunk files to scan.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="sprot_*.fasta",
        help="Glob pattern for FASTA chunk files inside --chunks-dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the merged hit report.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of chunk files to scan in parallel.",
    )
    parser.add_argument(
        "--progress-every-windows",
        type=int,
        default=500000,
        help="Emit a worker progress line every N valid windows scanned within a chunk. Set to 0 to disable.",
    )
    _db200k_cli.add_threshold_arg(parser, required=True)
    _db200k_cli.add_alignment_args(parser)
    return parser


def _format_hit_row(hit: dict[str, object], include_alignment: bool) -> str:
    fields = [
        f"{float(hit['score']):.4f}",
        str(hit["header"]),
        str(hit["start"]),
        str(hit["end"]),
        str(hit["window"]),
    ]
    if include_alignment:
        alignment = hit.get("alignment")
        if alignment is None:
            fields.extend(["", "", "", "", "", ""])
        else:
            fields.extend(
                [
                    str(alignment.query_start),
                    str(alignment.query_end),
                    str(alignment.target_start),
                    str(alignment.target_end),
                    "" if alignment.skipped_target_index is None else str(alignment.skipped_target_index),
                    alignment.aligned_window,
                ]
            )
    return "\t".join(fields)


def _iter_sorted_hit_rows(temp_paths: list[Path]):
    streams = [path.open() for path in temp_paths]
    try:
        heap = []
        for idx, fh in enumerate(streams):
            line = fh.readline()
            if line:
                heap.append((float(line.split("\t", 1)[0]), idx, line))
        heapq.heapify(heap)
        while heap:
            _, idx, line = heapq.heappop(heap)
            yield line
            next_line = streams[idx].readline()
            if next_line:
                heapq.heappush(heap, (float(next_line.split("\t", 1)[0]), idx, next_line))
    finally:
        for fh in streams:
            fh.close()


def scan_one_chunk(job):
    chunk_path, profiles, scan_kwargs, temp_dir, progress_every_windows = job
    stats: dict[str, int] = {}
    hits, windows_scanned = db200k_scan.scan_records(
        db200k_scan.iter_fasta_records(chunk_path),
        profiles,
        top_k=None,
        stats=stats,
        progress_every_windows=progress_every_windows,
        progress_label=chunk_path.name,
        **scan_kwargs,
    )
    temp_path = Path(temp_dir) / f"{chunk_path.stem}.hits.tsv"
    include_alignment = scan_kwargs["alignment_mode"] != "rigid"
    with temp_path.open("w") as fh:
        for hit in hits:
            fh.write(_format_hit_row(hit, include_alignment))
            fh.write("\n")
    return (
        chunk_path.name,
        windows_scanned,
        stats.get("prefilter_passed", 0),
        stats.get("scored_windows", 0),
        stats.get("threshold_passed", len(hits)),
        len(hits),
        temp_path,
    )


def main():
    parser = get_parser()
    args = parser.parse_args()

    profiles = _db200k_cli.build_profiles_from_args(args)
    scan_kwargs = _db200k_cli.scan_kwargs_from_args(args)

    chunk_paths = sorted(Path(args.chunks_dir).glob(args.glob))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk FASTA files matched {args.glob!r} in {args.chunks_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_alignment = args.alignment_mode != "rigid"
    total_windows = 0
    total_prefilter_passed = 0
    total_scored_windows = 0
    total_threshold_passed = 0
    total_hits = 0
    temp_paths = []

    with tempfile.TemporaryDirectory(prefix="db200k_chunk_hits_", dir=output_path.parent) as temp_dir:
        progress_every_windows = None if args.progress_every_windows <= 0 else args.progress_every_windows
        jobs = [
            (chunk_path, profiles, scan_kwargs, temp_dir, progress_every_windows)
            for chunk_path in chunk_paths
        ]
        if args.jobs == 1:
            results_iter = map(scan_one_chunk, jobs)
            pool = None
        else:
            start_method = "spawn" if sys.platform == "win32" else "fork"
            ctx = mp.get_context(start_method)
            pool = ctx.Pool(processes=args.jobs)
            results_iter = pool.imap_unordered(scan_one_chunk, jobs)

        try:
            for idx, (
                chunk_name,
                chunk_windows,
                chunk_prefilter_passed,
                chunk_scored_windows,
                chunk_threshold_passed,
                hit_count,
                temp_path,
            ) in enumerate(results_iter, start=1):
                total_windows += chunk_windows
                total_prefilter_passed += chunk_prefilter_passed
                total_scored_windows += chunk_scored_windows
                total_threshold_passed += chunk_threshold_passed
                total_hits += hit_count
                temp_paths.append(temp_path)
                print(
                    f"[{idx}/{len(chunk_paths)}] {chunk_name}: "
                    f"{chunk_windows} windows, "
                    f"{chunk_prefilter_passed} passed prefilter, "
                    f"{chunk_threshold_passed} passed threshold, "
                    f"{hit_count} hits <= {args.score_threshold:.4f}",
                    flush=True,
                )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        with output_path.open("w") as fh:
            fh.write(f"# query_seq\t{args.query_seq}\n")
            fh.write(f"# score_threshold\t{args.score_threshold:.4f}\n")
            fh.write(
                "# prefilter_score_threshold\t"
                f"{'' if args.prefilter_score_threshold is None else f'{args.prefilter_score_threshold:.4f}'}\n"
            )
            fh.write(f"# prefilter_score_mode\t{args.prefilter_score_mode}\n")
            fh.write(f"# profile_strategy\t{args.profile_strategy}\n")
            fh.write(f"# one_by_one_mode\t{args.one_by_one_mode}\n")
            fh.write(f"# one_by_one_matrix_tsv\t{args.one_by_one_matrix_tsv or ''}\n")
            fh.write(f"# alignment_mode\t{args.alignment_mode}\n")
            fh.write(f"# chunks_scanned\t{len(chunk_paths)}\n")
            fh.write(f"# windows_scanned\t{total_windows}\n")
            fh.write(f"# prefilter_passed_windows\t{total_prefilter_passed}\n")
            fh.write(f"# rescored_windows\t{total_scored_windows}\n")
            fh.write(f"# threshold_passed_windows\t{total_threshold_passed}\n")
            fh.write(f"# hits\t{total_hits}\n")
            if include_alignment:
                fh.write(
                    "# score\theader\tstart\tend\twindow\t"
                    "query_start\tquery_end\ttarget_start\ttarget_end\t"
                    "skipped_target_index\taligned_window\n"
                )
            else:
                fh.write("# score\theader\tstart\tend\twindow\n")
            for line in _iter_sorted_hit_rows(temp_paths):
                fh.write(line)

    print(f"Wrote {total_hits} hits to {output_path}")


if __name__ == "__main__":
    main()
