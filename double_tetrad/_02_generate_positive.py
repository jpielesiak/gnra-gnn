#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 02 — extract positive examples (double G-tetrads) as individual CIF files.

Reads ``double_tetrad_motifs_by_pdb.json`` (produced by step 01), opens each
assembly mmCIF, locates the eight guanine residues of every annotated
consecutive tetrad pair, canonicalises their order via
``canonical_order.canonicalize`` and writes a single CIF file per pair into
``double_tetrad/motif_cif_files/``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, write_cif
from rnapolis.tertiary_v2 import Residue, Structure

from canonical_order import canonicalize

DEFAULT_MIRROR = "/mnt/data-ssd/tzok/onquadro-main/mirror/data/assemblies/mmCIF/divided"
DOUBLE_TETRAD_SIZE = 8


def mmcif_path(stem: str, mirror_dir: str) -> str:
    pdb_id = stem.split("-")[0]
    mid = pdb_id[1:3]
    return os.path.join(mirror_dir, mid, f"{stem}.cif.gz")


def find_residues(residues: List[Residue], unit_ids: List[Dict[str, Any]]) -> Optional[List[Residue]]:
    found: List[Residue] = []
    for uid in unit_ids:
        chain_id = uid["chain_id"]
        resnum = uid["residue_number"]
        icode = uid.get("insertion_code") or ""
        match = None
        for r in residues:
            r_icode = r.insertion_code or ""
            if r.chain_id == chain_id and r.residue_number == resnum and r_icode == icode:
                match = r
                break
        if match is None:
            return None
        found.append(match)
    return found


def c1_prime_coords(residues: List[Residue]) -> Optional[List[Tuple[float, float, float]]]:
    coords: List[Tuple[float, float, float]] = []
    for r in residues:
        col_atom = r._col("atom_name")
        c1 = r.atoms[r.atoms[col_atom] == "C1'"]
        if c1.empty:
            return None
        row = c1.iloc[0]
        coords.append((float(row[r._col("x")]), float(row[r._col("y")]), float(row[r._col("z")])))
    return coords


def process_one_stem(
    stem: str, motifs: List[Dict[str, Any]], mirror_dir: str, output_dir: str
) -> Tuple[str, int, int]:
    path = mmcif_path(stem, mirror_dir)
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping {stem}")
        return stem, 0, len(motifs)

    try:
        with gzip.open(path, "rt") as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)
        residues = [r for r in structure.residues if r.is_nucleotide]
    except Exception as e:
        print(f"  Error parsing {stem}: {e}")
        return stem, 0, len(motifs)

    n_saved = 0
    n_skipped = 0

    for motif in motifs:
        motif_key = motif["motif_key"]
        output_file = os.path.join(output_dir, f"{motif_key}.cif")
        if os.path.exists(output_file):
            n_skipped += 1
            continue

        unit_ids = motif["unit_ids"]
        if len(unit_ids) != DOUBLE_TETRAD_SIZE:
            print(f"    Warning: {motif_key} has {len(unit_ids)} unit_ids")
            n_skipped += 1
            continue

        matched = find_residues(residues, unit_ids)
        if matched is None:
            print(f"    Warning: {motif_key} — could not locate all 8 residues")
            n_skipped += 1
            continue

        coords = c1_prime_coords(matched)
        if coords is None:
            print(f"    Warning: {motif_key} — missing C1'")
            n_skipped += 1
            continue

        perm, _ = canonicalize(coords)
        ordered_residues = [matched[i] for i in perm]

        try:
            atoms_df = pd.concat(r.atoms for r in ordered_residues)
            with open(output_file, "w") as f:
                write_cif(atoms_df, f)
            n_saved += 1
        except Exception as e:
            print(f"    Error saving {motif_key}: {e}")
            n_skipped += 1

    return stem, n_saved, n_skipped


def process_wrapper(args):
    return process_one_stem(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent / "double_tetrad_motifs_by_pdb.json"),
    )
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "motif_cif_files"),
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.input) as f:
        motifs_by_pdb: Dict[str, List[Dict[str, Any]]] = json.load(f)

    total_motifs = sum(len(v) for v in motifs_by_pdb.values())
    print(f"Loaded {total_motifs} double tetrads from {len(motifs_by_pdb)} files")

    pending: Dict[str, List[Dict[str, Any]]] = {}
    skipped_stems = 0
    for stem, motifs in motifs_by_pdb.items():
        if all(os.path.exists(os.path.join(output_dir, f"{m['motif_key']}.cif")) for m in motifs):
            skipped_stems += 1
        else:
            pending[stem] = motifs

    print(f"Skipping {skipped_stems} already-processed stems")
    print(f"Will process {len(pending)} stems with {sum(len(v) for v in pending.values())} double tetrads")

    if not pending:
        print("Nothing to do.")
        return

    tasks = [(stem, motifs, args.mirror, output_dir) for stem, motifs in pending.items()]
    total_saved = 0
    total_skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_wrapper, t): t[0] for t in tasks}
        for future in as_completed(futures):
            stem = futures[future]
            try:
                _, saved, skipped = future.result()
                total_saved += saved
                total_skipped += skipped
                if saved:
                    print(f"  {stem}: saved {saved}, skipped {skipped}")
            except Exception as e:
                print(f"  Error processing {stem}: {e}")

    print(f"\nDone. Saved {total_saved} double tetrads, skipped {total_skipped}.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
