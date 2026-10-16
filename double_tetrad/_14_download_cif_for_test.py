#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the full mmCIF file for every structure listed in an answers JSON.

Each top-level key of the JSON (e.g. "139d-assembly1" or "1abc") is fetched
from the RCSB PDB as a gzipped mmCIF and saved under exactly that name
(e.g. 139d-assembly1.cif.gz), so test_models.py can match it back to the JSON.

Example:
    python download_structures.py motifs.json -output_folder structures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RCSB_URL = "https://files.rcsb.org/download/{name}.cif.gz"
GZIP_MAGIC = b"\x1f\x8b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("answers", type=Path, help="JSON file whose keys are structure names")
    p.add_argument(
        "-output_folder", "--output-folder",
        dest="output_folder",
        type=Path,
        default=Path("structures"),
        help="Folder to save the .cif.gz files into (default: %(default)s)",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel downloads (default: %(default)s; keep it small to be polite to RCSB)",
    )
    p.add_argument(
        "--retries", type=int, default=3,
        help="Attempts per file before giving up (default: %(default)s)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-download files that already exist",
    )
    return p.parse_args()


def rcsb_name(key: str) -> str:
    """'139d-assembly1' -> '139D-assembly1', '1abc' -> '1ABC'."""
    pdb_id, sep, rest = key.partition("-")
    return pdb_id.upper() + sep + rest


def download(key: str, out_dir: Path, retries: int, overwrite: bool) -> tuple[str, str]:
    target = out_dir / f"{key}.cif.gz"
    if target.exists() and not overwrite:
        return key, "exists"

    url = RCSB_URL.format(name=rcsb_name(key))
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if not data.startswith(GZIP_MAGIC):
                raise ValueError("response is not a gzip file")
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)  # atomic: no half-written files on interruption
            return key, "ok"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 404:  # not going to appear on retry
                break
        except Exception as exc:  # noqa: BLE001 — network hiccups, timeouts, etc.
            last_error = str(exc)
        time.sleep(2 * attempt)

    return key, f"failed ({last_error}) {url}"


def main() -> None:
    args = parse_args()
    with open(args.answers) as handle:
        keys = sorted(json.load(handle).keys())
    if not keys:
        raise SystemExit(f"No structures listed in {args.answers}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    print(f"{len(keys)} structures -> {args.output_folder.resolve()}")

    counts = {"ok": 0, "exists": 0}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(download, k, args.output_folder, args.retries, args.overwrite)
            for k in keys
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            key, status = future.result()
            if status in counts:
                counts[status] += 1
            else:
                failures.append(f"{key}\t{status}")
                print(f"  [{i}/{len(keys)}] {key}: {status}", file=sys.stderr)
            if i % 25 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)} done")

    print(
        f"\nDownloaded {counts['ok']}, already present {counts['exists']}, "
        f"failed {len(failures)}"
    )
    if failures:
        failed_path = args.output_folder / "failed_downloads.txt"
        failed_path.write_text("\n".join(failures) + "\n")
        print(f"Failed entries listed in {failed_path}")


if __name__ == "__main__":
    main()