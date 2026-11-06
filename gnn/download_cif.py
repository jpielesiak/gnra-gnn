#!/usr/bin/env python3
"""
Collect PDB IDs from two JSON files and download their .cif.gz files.

File 1: dict keyed by PDB ID -> list of motifs (unit_ids also carry pdb_id)
File 2: list of {"pdb_id": ..., "regions": {...}}

Usage:
    python download_cif.py gnra_motifs_by_pdb.json negative_regions.json -o cif_files -j 8
"""
import argparse
import gzip
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL_TEMPLATES = [
    "https://files.rcsb.org/download/{ID}.cif.gz",
    "https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{mid}/{id}.cif.gz",
]


def ids_from_file1(data):
    """Dict keyed by PDB ID; also pick up pdb_id inside unit_ids just in case."""
    ids = set(data.keys())
    for motifs in data.values():
        for motif in motifs or []:
            for unit in motif.get("unit_ids", []):
                if unit.get("pdb_id"):
                    ids.add(unit["pdb_id"])
    return ids


def ids_from_file2(data):
    """List of entries, each with a pdb_id field."""
    return {entry["pdb_id"] for entry in data if entry.get("pdb_id")}


def collect_ids(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        ids = ids_from_file1(data)
    elif isinstance(data, list):
        ids = ids_from_file2(data)
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")
    # Normalise: 4-character IDs, lowercase
    return {i.strip().lower() for i in ids if len(i.strip()) == 4}


def is_valid_gzip(path):
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


def download(pdb_id, out_dir, retries=3):
    dest = out_dir / f"{pdb_id}.cif.gz"
    if dest.exists() and is_valid_gzip(dest):
        return pdb_id, "skipped"

    tmp = dest.with_suffix(".gz.part")
    for template in URL_TEMPLATES:
        url = template.format(ID=pdb_id.upper(), id=pdb_id, mid=pdb_id[1:3])
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "cif-downloader/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                if is_valid_gzip(tmp):
                    tmp.replace(dest)
                    return pdb_id, "ok"
                raise OSError("corrupt gzip")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break  # not on this mirror, try the next one
                time.sleep(2 * attempt)
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    return pdb_id, "failed"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_files", nargs="+", help="JSON files containing PDB IDs")
    p.add_argument("-o", "--out-dir", default="cif_files", help="Output directory")
    p.add_argument("-j", "--jobs", type=int, default=8, help="Parallel downloads")
    args = p.parse_args()

    all_ids = set()
    for path in args.json_files:
        ids = collect_ids(path)
        print(f"{path}: {len(ids)} PDB IDs")
        all_ids |= ids
    print(f"Total unique PDB IDs: {len(all_ids)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(download, i, out_dir) for i in sorted(all_ids)]
        for n, fut in enumerate(as_completed(futures), 1):
            pdb_id, status = fut.result()
            counts[status] += 1
            if status == "failed":
                failed.append(pdb_id)
            print(f"[{n}/{len(futures)}] {pdb_id}: {status}")

    print(f"\nDone. downloaded={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}")
    if failed:
        fail_file = out_dir / "failed_ids.txt"
        fail_file.write_text("\n".join(sorted(failed)) + "\n")
        print(f"Failed IDs written to {fail_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()