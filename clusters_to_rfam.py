#!/usr/bin/env python3
"""
Map clustered RNA motif files (e.g. HL_3NDB_002.cif) to Rfam families.

Produces, next to the input JSON:
    <jsonname>_rfam.csv                  - every motif (representatives + members)
    <jsonname>_representatives_rfam.csv  - only cluster representatives

Rfam data comes from Rfam's own PDB mapping (pdb_full_region.txt.gz) and
family table (family.txt.gz), downloaded once and cached locally.

If the motif .cif files are available on disk, the script reads which chain(s)
each motif lies on and keeps only Rfam families mapped to those chains. This
matters for big structures (e.g. ribosomes), where one PDB entry contains many
Rfam families. If a .cif file can't be found, all families of the PDB entry
are reported and match_level says "entry".

Usage:
    python clusters_to_rfam.py clusters.json
    python clusters_to_rfam.py clusters.json --motif-dir /path/to/files
    python clusters_to_rfam.py clusters.json --no-chain-filter
    uv run clusters_to_rfam.py approximate-affinity-propagation.json --motif-dir ./motif_cif_files ./negative_cif_files
"""
import argparse
import csv
import gzip
import json
import re
import sys
import urllib.request
from collections import OrderedDict
from pathlib import Path

RFAM_BASE = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/"
PDB_MAP_FILE = "pdb_full_region.txt.gz"
FAMILY_FILE = "family.txt.gz"

PDB_ID_RE = re.compile(r"(?:^|_)([0-9][A-Za-z0-9]{3})(?:_|$)")

FIELDS = [
    "cluster_index", "cluster_representative", "motif_file", "motif_id",
    "pdb_id", "is_representative", "motif_chains", "rfam_acc", "rfam_id",
    "rfam_description", "rfam_chains", "rfam_regions", "match_level",
]


# ---------------------------------------------------------------- Rfam data
def download(cache_dir: Path, name: str, refresh: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    if refresh or not path.exists():
        print(f"Downloading {RFAM_BASE + name} ...", file=sys.stderr)
        urllib.request.urlretrieve(RFAM_BASE + name, path)
    return path


def load_pdb_map(path: Path, significant_only: bool) -> dict:
    """pdb_id (lower) -> list of dicts with rfam_acc, chain, start, end.

    Columns: rfam_acc, pdb_id, chain, pdb_start, pdb_end, bit_score,
             evalue_score, cm_start, cm_end, hex_colour, is_significant
    """
    mapping = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            if significant_only and len(cols) >= 11 and cols[10].strip() != "1":
                continue
            mapping.setdefault(cols[1].lower(), []).append({
                "rfam_acc": cols[0],
                "chain": cols[2],
                "start": cols[3],
                "end": cols[4],
            })
    return mapping


def load_families(path: Path) -> dict:
    """rfam_acc -> (rfam_id, description). Columns: rfam_acc, rfam_id, auto_wiki, description, ..."""
    fams = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 4 and cols[0].startswith("RF"):
                fams[cols[0]] = (cols[1], cols[3])
    return fams


# ---------------------------------------------------------------- motif files
def pdb_id_from_path(motif_path: str):
    m = PDB_ID_RE.search(Path(motif_path).stem)
    return m.group(1).lower() if m else None


def find_motif_file(motif_path: str, search_dirs) -> Path | None:
    p = Path(motif_path)
    if p.is_absolute():
        return p if p.exists() else None
    for d in search_dirs:
        for candidate in (d / p, d / p.name):
            if candidate.exists():
                return candidate
    return None


def motif_chains(cif_path: Path) -> set | None:
    """Read chain IDs from the _atom_site loop (auth_asym_id, else label_asym_id)."""
    headers, chains, in_loop = [], set(), False
    try:
        with open(cif_path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if line.startswith("_atom_site."):
                    headers.append(line.split()[0][len("_atom_site."):])
                    in_loop = True
                    continue
                if not in_loop:
                    continue
                if not line or line.startswith(("#", "loop_", "_")):
                    if chains:
                        break
                    if line.startswith(("loop_", "_")):
                        headers, in_loop = [], False
                    continue
                key = "auth_asym_id" if "auth_asym_id" in headers else "label_asym_id"
                if key not in headers:
                    return None
                tokens = line.split()
                idx = headers.index(key)
                if len(tokens) > idx:
                    chains.add(tokens[idx])
    except OSError:
        return None
    return chains or None


# ---------------------------------------------------------------- main logic
def iter_motifs(data: dict):
    clusters = data.get("clustering", data).get("clusters", [])
    for i, cluster in enumerate(clusters, start=1):
        rep = cluster.get("representative")
        if rep:
            yield i, rep, rep, True
        for member in cluster.get("members", []):
            if member != rep:
                yield i, rep, member, False


def rows_for_motif(cluster_idx, rep, motif, is_rep, pdb_map, families,
                   search_dirs, chain_filter):
    pdb_id = pdb_id_from_path(motif)
    base = {
        "cluster_index": cluster_idx,
        "cluster_representative": rep,
        "motif_file": motif,
        "motif_id": Path(motif).stem,
        "pdb_id": pdb_id.upper() if pdb_id else "",
        "is_representative": is_rep,
    }

    chains = None
    if chain_filter:
        f = find_motif_file(motif, search_dirs)
        if f:
            chains = motif_chains(f)
    base["motif_chains"] = ";".join(sorted(chains)) if chains else ""

    hits = pdb_map.get(pdb_id, []) if pdb_id else []
    if chains is not None:
        hits = [h for h in hits if h["chain"] in chains]
        level = "chain"
    else:
        level = "entry"

    if not hits:
        note = "no PDB id in filename" if not pdb_id else f"no Rfam match ({level})"
        return [{**base, "rfam_acc": "", "rfam_id": "", "rfam_description": "",
                 "rfam_chains": "", "rfam_regions": "", "match_level": note}]

    by_family = OrderedDict()
    for h in hits:
        by_family.setdefault(h["rfam_acc"], []).append(h)

    rows = []
    for acc, fam_hits in by_family.items():
        rfam_id, desc = families.get(acc, ("", ""))
        rows.append({
            **base,
            "rfam_acc": acc,
            "rfam_id": rfam_id,
            "rfam_description": desc,
            "rfam_chains": ";".join(sorted({h["chain"] for h in fam_hits})),
            "rfam_regions": ";".join(f'{h["chain"]}:{h["start"]}-{h["end"]}' for h in fam_hits),
            "match_level": level,
        })
    return rows


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--motif-dir", type=Path, nargs="+", action="extend", default=[],
                    help="One or more directories to look for motif .cif files, "
                         "searched in the order given. "
                         "The JSON's directory and the current directory are always searched.")
    ap.add_argument("--no-chain-filter", action="store_true",
                    help="Report every Rfam family in the PDB entry, ignoring motif chains.")
    ap.add_argument("--include-nonsignificant", action="store_true",
                    help="Keep Rfam hits flagged as not significant.")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "rfam_pdb",
                    help="Where to cache Rfam files (default: ~/.cache/rfam_pdb)")
    ap.add_argument("--refresh", action="store_true", help="Re-download the Rfam files.")
    args = ap.parse_args()

    with open(args.json_file, encoding="utf-8") as fh:
        data = json.load(fh)

    pdb_map = load_pdb_map(download(args.cache_dir, PDB_MAP_FILE, args.refresh),
                           significant_only=not args.include_nonsignificant)
    families = load_families(download(args.cache_dir, FAMILY_FILE, args.refresh))

    search_dirs = args.motif_dir + [args.json_file.resolve().parent, Path.cwd()]

    all_rows = []
    for cluster_idx, rep, motif, is_rep in iter_motifs(data):
        all_rows.extend(rows_for_motif(cluster_idx, rep, motif, is_rep, pdb_map,
                                       families, search_dirs,
                                       chain_filter=not args.no_chain_filter))

    out_dir = args.json_file.resolve().parent
    stem = args.json_file.stem
    write_csv(out_dir / f"{stem}_rfam.csv", all_rows)
    write_csv(out_dir / f"{stem}_representatives_rfam.csv",
              [r for r in all_rows if r["is_representative"]])


if __name__ == "__main__":
    main()