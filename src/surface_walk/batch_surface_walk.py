#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import threading
import ssl
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from subprocess import run, CalledProcessError
from concurrent.futures import ThreadPoolExecutor, as_completed


FASTA_RE = re.compile(r"^>(\\S+)")
SSL_CONTEXT = None


def open_url(req, timeout=30):
    if SSL_CONTEXT is not None:
        return urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT)
    return urllib.request.urlopen(req, timeout=timeout)


def parse_fasta_headers(path: Path):
    """Yield (accession, entry_name, header) from a UniProt-style FASTA."""
    with path.open() as f:
        for line in f:
            if not line.startswith(">"):
                continue
            header = line.strip()
            # UniProt: >sp|Q6GJC6|RPOB_STAAR ...
            parts = header[1:].split("|")
            acc = None
            entry = None
            if len(parts) >= 3:
                acc = parts[1].strip()
                entry = parts[2].split()[0].strip()
            else:
                m = FASTA_RE.match(header)
                entry = m.group(1) if m else None
            yield acc, entry, header


def urlretrieve(url: str, dest: Path, timeout=30, verbose=False) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "batch_surface_walk/1.0"})
        with open_url(req, timeout=timeout) as r, dest.open("wb") as w:
            w.write(r.read())
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        if verbose:
            print(f"[download failed] {url} -> {e}", file=sys.stderr)
        return False


def fetch_alphafold(acc: str, cache_dir: Path, verbose=False):
    if not acc:
        return None
    # Try AlphaFold API first (gives exact URLs when available)
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "batch_surface_walk/1.0"})
        with open_url(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        if isinstance(data, list) and data:
            entry = data[0]
            for key in ("pdbUrl", "cifUrl", "pdbUrl" if "pdbUrl" in entry else None):
                if not key:
                    continue
                url = entry.get(key)
                if not url:
                    continue
                ext = Path(url).suffix
                dest = cache_dir / "alphafold" / f"{acc}{ext}"
                if dest.exists():
                    return dest
                if urlretrieve(url, dest, verbose=verbose):
                    return dest
    except Exception as e:
        if verbose:
            print(f"[alphafold api failed] {acc} -> {e}", file=sys.stderr)
    # Try common AlphaFold filenames (v4, v3, v2), PDB and CIF.
    base = f"AF-{acc}-F1-model"
    candidates = [
        f"https://alphafold.ebi.ac.uk/files/{base}_v4.pdb",
        f"https://alphafold.ebi.ac.uk/files/{base}_v4.cif",
        f"https://alphafold.ebi.ac.uk/files/{base}_v3.pdb",
        f"https://alphafold.ebi.ac.uk/files/{base}_v3.cif",
        f"https://alphafold.ebi.ac.uk/files/{base}_v2.pdb",
        f"https://alphafold.ebi.ac.uk/files/{base}_v2.cif",
    ]
    for url in candidates:
        ext = Path(url).suffix
        dest = cache_dir / "alphafold" / f"{base}{ext}"
        if dest.exists():
            return dest
        if urlretrieve(url, dest, verbose=verbose):
            return dest
    return None


def fetch_rcsb_pdb(acc: str, cache_dir: Path, verbose=False):
    if not acc:
        return None
    # Query RCSB search API for UniProt accession mapping.
    # RCSB search API (recommended structure): exact match on UniProt accession.
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": acc,
            },
        },
        "return_type": "entry",
        "request_options": {
            "pager": {"start": 0, "rows": 1},
            "results_content_type": ["experimental"],
        },
    }
    try:
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        data = json.dumps(query).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": "batch_surface_walk/1.0",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with open_url(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        if verbose:
            print(f"[rcsb query failed] {acc} -> {e}", file=sys.stderr)
        return None
    result = data.get("result_set", [])
    if not result:
        return None
    pdb_id = result[0].get("identifier")
    if not pdb_id:
        return None
    # Download PDB (fallback to CIF)
    pdb_id = pdb_id.lower()
    dest_pdb = cache_dir / "rcsb" / f"{pdb_id}.pdb"
    if dest_pdb.exists():
        return dest_pdb
    url_pdb = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    if urlretrieve(url_pdb, dest_pdb, verbose=verbose):
        return dest_pdb
    dest_cif = cache_dir / "rcsb" / f"{pdb_id}.cif"
    url_cif = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    if urlretrieve(url_cif, dest_cif, verbose=verbose):
        return dest_cif
    return None


def fetch_uniprot_pdb(acc: str, cache_dir: Path, verbose=False):
    if not acc:
        return None
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "batch_surface_walk/1.0"})
        with open_url(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        if verbose:
            print(f"[uniprot query failed] {acc} -> {e}", file=sys.stderr)
        return None
    xrefs = data.get("uniProtKBCrossReferences", [])
    pdb_ids = []
    for x in xrefs:
        if x.get("database") == "PDB":
            pid = x.get("id")
            if pid:
                pdb_ids.append(pid)
    if not pdb_ids:
        return None
    # Try first PDB id
    pdb_id = pdb_ids[0].upper()
    dest_pdb = cache_dir / "rcsb" / f"{pdb_id.lower()}.pdb"
    if dest_pdb.exists():
        return dest_pdb
    url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    if urlretrieve(url_pdb, dest_pdb, verbose=verbose):
        return dest_pdb
    dest_cif = cache_dir / "rcsb" / f"{pdb_id.lower()}.cif"
    url_cif = f"https://files.rcsb.org/download/{pdb_id}.cif"
    if urlretrieve(url_cif, dest_cif, verbose=verbose):
        return dest_cif
    return None


def run_surface_walk(pdb_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "surface_walk.py"),
        "--pdb",
        str(pdb_path),
        "--outdir",
        str(outdir),
    ]
    try:
        run(cmd, check=True)
        return True
    except CalledProcessError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="UniProt FASTA (e.g., staph.fasta)")
    ap.add_argument("--outdir", default="/Users/henryhardigan/surface_walk_batch")
    ap.add_argument("--cache", default="/Users/henryhardigan/structure_cache")
    ap.add_argument("--max", type=int, default=0, help="Limit to first N entries (0=all)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between downloads")
    ap.add_argument("--verbose", action="store_true", help="Verbose download logging")
    ap.add_argument("--jobs", type=int, default=4, help="Parallel workers (default: 4)")
    ap.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for downloads")
    args = ap.parse_args()
    global SSL_CONTEXT
    if args.insecure:
        SSL_CONTEXT = ssl._create_unverified_context()

    fasta = Path(args.fasta)
    outdir = Path(args.outdir)
    cache = Path(args.cache)

    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "batch_log.tsv"
    # materialize entries to enable parallel processing
    entries = []
    n = 0
    for acc, entry, header in parse_fasta_headers(fasta):
        if args.max and n >= args.max:
            break
        n += 1
        entry = entry or acc or f"entry_{n}"
        entries.append((acc, entry))

    log_lock = threading.Lock()

    def log_line(log_f, line: str):
        with log_lock:
            log_f.write(line)
            log_f.flush()

    def process_one(acc, entry, log_f):
        out = outdir / entry
        if (out / "walks.json").exists():
            log_line(log_f, f"{entry}\t{acc}\tskipped\tcached\t{out}\n")
            return "skipped"

        pdb_path = fetch_alphafold(acc, cache, verbose=args.verbose)
        source = "alphafold" if pdb_path else ""
        if not pdb_path:
            pdb_path = fetch_rcsb_pdb(acc, cache, verbose=args.verbose)
            source = "rcsb" if pdb_path else ""
        if not pdb_path:
            pdb_path = fetch_uniprot_pdb(acc, cache, verbose=args.verbose)
            source = "uniprot_pdb" if pdb_path else ""

        if not pdb_path:
            log_line(log_f, f"{entry}\t{acc}\tno_structure\t\t\n")
            return "no_structure"

        ok = run_surface_walk(pdb_path, out)
        status = "ok" if ok else "failed"
        log_line(log_f, f"{entry}\t{acc}\t{status}\t{source}\t{pdb_path}\n")
        time.sleep(args.sleep)
        return status

    with log_path.open("w") as log:
        log.write("entry\taccession\tstatus\tsource\tpdb_path\n")
        # run in parallel
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = [ex.submit(process_one, acc, entry, log) for acc, entry in entries]
            for _ in as_completed(futs):
                pass

    print(f"Done. Log: {log_path}")


if __name__ == "__main__":
    main()
