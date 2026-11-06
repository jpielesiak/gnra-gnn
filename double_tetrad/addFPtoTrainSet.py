#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create one mmCIF per candidate row in a test-results CSV.

For every row of the CSV (columns: structure, candidate_index, residue_ids,
...), this script:

1. Locates the source assembly mmCIF under the mirror folder, using the same
   layout as the negative-sample generator: ``<pdb_id[1:3]>/<stem>.cif.gz``
   with ``pdb_id = stem.split("-")[0]``.
2. Parses it and collects exactly the residues listed in ``residue_ids``
   (``chain:residue_number``, optional insertion code appended).
3. Writes a new mmCIF containing only those residues, named
   ``<stem>_<candidate_index>.cif``.

Example:
    uv run addFPtoTrainSet.py -csv ./FPtoAdd.csv \
        -mirror ./test_cif \
        -out candidate_cif_files
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, write_cif
from rnapolis.tertiary_v2 import Structure

ResidueKey = Tuple[str, int, str]  # (chain_id, residue_number, insertion_code)


# def mmcif_path(stem: str, mirror_dir: str) -> str:
#     """Same mirror layout as the negative-sample generator."""
#     pdb_id = stem.split("-")[0]
#     mid = pdb_id[1:3]
#     return os.path.join(mirror_dir, mid, f"{stem}.cif.gz")
def mmcif_path(stem: str, mirror_dir: str) -> str:
    """Flat mirror: every assembly mmCIF is directly in the folder."""
    return os.path.join(mirror_dir, f"{stem}.cif.gz")

def parse_residue_id(rid: str) -> ResidueKey:
    """'A:3' or 'A:3B' -> ('A', 3, 'B')."""
    chain, rest = rid.split(":", 1)
    m = re.match(r"(-?\d+)(.*)$", rest)
    if not m:
        raise ValueError(f"Cannot parse residue id {rid!r}")
    return chain, int(m.group(1)), m.group(2)


def clean_text(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    return str(v)


def collect_by_key(structure: Structure) -> dict[ResidueKey, Any]:
    """Map (chain, resnum, icode) -> residue, using the same cleaning as the
    test harness (residue_key in _13_test_all_models.py)."""
    out = {}
    for r in structure.residues:
        icode = clean_text(r.insertion_code).strip()
        if icode in {"?", "."}:
            icode = ""
        key = (clean_text(r.chain_id).strip(), int(r.residue_number), icode)
        out[key] = r
    return out


def build_candidate_cif(
    stem: str,
    residue_ids: List[str],
    mirror_dir: str,
    output_dir: str,
    out_name: str,
) -> Tuple[int, Optional[str]]:
    path = mmcif_path(stem, mirror_dir)
    if not os.path.exists(path):
        return 0, f"source not found: {path}"

    try:
        with gzip.open(path, "rt") as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)
        by_key = collect_by_key(structure)
    except Exception as e:
        return 0, f"parse error: {e}"

    missing = []
    selected = []
    for rid in residue_ids:
        key = parse_residue_id(rid)
        r = by_key.get(key)
        if r is None:
            missing.append(rid)
        else:
            selected.append(r)

    if missing:
        return 0, f"missing residues: {', '.join(missing)}"

    out_file = os.path.join(output_dir, out_name)
    try:
        atoms = pd.concat(r.atoms for r in selected)
        with open(out_file, "w") as f:
            write_cif(atoms, f)
    except Exception as e:
        return 0, f"write error: {e}"

    return len(selected), None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-csv", "--csv", type=Path, required=True,
                   help="Test-results CSV with structure, candidate_index, residue_ids")
    p.add_argument("-mirror", "--mirror", type=Path, required=True,
                   help="Root of the assemblies mmCIF mirror (divided layout)")
    p.add_argument("-out", "--output-dir", type=Path, default=Path("candidate_cif_files"))
    p.add_argument("--prefix", default="",
                   help="Optional prefix for output files (e.g. 'NEG_')")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    required = {"structure", "candidate_index", "residue_ids"}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        sys.exit(f"CSV is missing required columns: {', '.join(sorted(missing_cols))}")

    n_ok = n_skip = 0
    failures: List[str] = []
    for _i, row in df.iterrows():
        stem = str(row["structure"])
        cand = int(row["candidate_index"])
        residue_ids = [r.strip() for r in str(row["residue_ids"]).split(",") if r.strip()]

        out_name = f"{args.prefix}{stem}_{cand:04d}.cif"
        n_found, err = build_candidate_cif(
            stem, residue_ids, str(args.mirror), str(args.output_dir), out_name
        )
        if err is not None:
            n_skip += 1
            failures.append(f"{out_name}: {err}")
            continue
        n_ok += 1

    print(f"Done. Wrote {n_ok} candidate mmCIF files, skipped {n_skip}.")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
