#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 03 — generate negative examples (non-double-tetrad 8-tuples) via KD-tree.

For every (pdb, assembly) stem that contributed positive double tetrads:

1. Parse the assembly mmCIF and collect every nucleotide (RNA or DNA) with C1'.
2. Build a ``cKDTree`` over the C1' coordinates.
3. Enumerate all 8-tuples within a radius (no coplanarity pruning — 8 points
   from two stacked tetrads span two parallel planes, not one).
4. Apply **tuple-level exclusion**: ignore candidates sharing more than
   ``max_shared_with_tetrad`` residues with any annotated double tetrad.
5. Sample up to ``K × #positives`` per file, canonicalise, write CIF.

Defaults: ``radius = 22 Å``, ``max_distance = 22 Å``, ``K = 5``,
``max_shared_with_tetrad = 6`` (a double tetrad has 8 residues, so 7/8 or 8/8
shared → ignore; 0–6/8 shared → negative).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, write_cif
from rnapolis.tertiary_v2 import Residue, Structure

from neighborhood import enumerate_tuples

DEFAULT_MIRROR = "/mnt/data-ssd/tzok/onquadro-main/mirror/data/assemblies/mmCIF/divided"
DOUBLE_TETRAD_SIZE = 8
DEFAULT_RADIUS = 22.0
DEFAULT_MAX_DISTANCE = 22.0
DEFAULT_K = 5
DEFAULT_SEED = 42
DEFAULT_MAX_SHARED = 6


def mmcif_path(stem: str, mirror_dir: str) -> str:
    pdb_id = stem.split("-")[0]
    mid = pdb_id[1:3]
    return os.path.join(mirror_dir, mid, f"{stem}.cif.gz")


def collect_nucleotides(
    structure: Structure,
) -> Tuple[List[Residue], List[Tuple[float, float, float]], List[Tuple[str, int, str]]]:
    residues: List[Residue] = []
    coords: List[Tuple[float, float, float]] = []
    ids: List[Tuple[str, int, str]] = []
    for r in structure.residues:
        if not r.is_nucleotide:
            continue
        col_atom = r._col("atom_name")
        c1 = r.atoms[r.atoms[col_atom] == "C1'"]
        if c1.empty:
            continue
        row = c1.iloc[0]
        residues.append(r)
        coords.append(
            (float(row[r._col("x")]), float(row[r._col("y")]), float(row[r._col("z")]))
        )
        ids.append((r.chain_id, r.residue_number, r.insertion_code or ""))
    return residues, coords, ids


def build_tuple_exclusion(motifs: List[Dict[str, Any]]) -> List[frozenset]:
    excluded: List[frozenset] = []
    for motif in motifs:
        unit_ids = motif.get("unit_ids", [])
        if len(unit_ids) != DOUBLE_TETRAD_SIZE:
            continue
        rid_set = frozenset(
            (u["chain_id"], u["residue_number"], u.get("insertion_code") or "")
            for u in unit_ids
        )
        excluded.append(rid_set)
    return excluded


def process_one_stem(
    stem: str,
    tuple_exclusion: List[frozenset],
    positive_count: int,
    mirror_dir: str,
    output_dir: str,
    radius: float,
    max_distance: Optional[float],
    k: int,
    seed: int,
    max_shared_with_tetrad: int,
) -> Tuple[str, int, int]:
    path = mmcif_path(stem, mirror_dir)
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping {stem}")
        return stem, 0, 0

    try:
        with gzip.open(path, "rt") as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)
    except Exception as e:
        print(f"  Error parsing {stem}: {e}")
        return stem, 0, 0

    residues, coords, ids = collect_nucleotides(structure)
    if len(residues) < DOUBLE_TETRAD_SIZE:
        return stem, 0, 0

    candidates = enumerate_tuples(
        coords,
        n=DOUBLE_TETRAD_SIZE,
        radius=radius,
        residue_ids=ids,
        max_distance=max_distance,
        dedup=True,
        tuple_exclusion=tuple_exclusion,
        max_shared_with_tetrad=max_shared_with_tetrad,
    )

    if not candidates:
        return stem, 0, 0

    target = k * positive_count
    rng = np.random.default_rng(seed)
    if len(candidates) > target:
        idx = rng.choice(len(candidates), size=target, replace=False)
        sampled = [candidates[i] for i in idx]
    else:
        sampled = candidates

    n_saved = 0
    n_skipped = 0
    for j, (indices, _ordered) in enumerate(sampled):
        out_file = os.path.join(output_dir, f"NEG_{stem}_{j:04d}.cif")
        if os.path.exists(out_file):
            n_skipped += 1
            continue
        ordered_residues = [residues[i] for i in indices]
        try:
            atoms_df = pd.concat(r.atoms for r in ordered_residues)
            with open(out_file, "w") as f:
                write_cif(atoms_df, f)
            n_saved += 1
        except Exception as e:
            print(f"    Error saving NEG_{stem}_{j:04d}: {e}")
            n_skipped += 1

    return stem, n_saved, n_skipped


def process_wrapper(args):
    return process_one_stem(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motifs",
        default=str(Path(__file__).parent / "double_tetrad_motifs_by_pdb.json"),
    )
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "negative_cif_files"),
    )
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-shared-with-tetrad",
        type=int,
        default=DEFAULT_MAX_SHARED,
        help="Max residues shared with any annotated double tetrad before "
        "being ignored (default: 6 → 7/8 and 8/8 ignored, 0-6/8 kept)",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.motifs) as f:
        motifs_by_pdb: Dict[str, List[Dict[str, Any]]] = json.load(f)

    print(f"Loaded {sum(len(v) for v in motifs_by_pdb.values())} positives across {len(motifs_by_pdb)} stems")
    print(
        f"Params: radius={args.radius} Å, max_distance={args.max_distance} Å, "
        f"K={args.k}, seed={args.seed}, max_shared={args.max_shared_with_tetrad}"
    )

    tasks = []
    for stem, motifs in motifs_by_pdb.items():
        tuple_excl = build_tuple_exclusion(motifs)
        tasks.append(
            (
                stem,
                tuple_excl,
                len(motifs),
                args.mirror,
                output_dir,
                args.radius,
                args.max_distance,
                args.k,
                args.seed + hash(stem) % (2**31),
                args.max_shared_with_tetrad,
            )
        )

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
                    print(f"  {stem}: saved {saved} negatives, skipped {skipped}")
            except Exception as e:
                print(f"  Error processing {stem}: {e}")

    print(f"\nDone. Saved {total_saved} negatives, skipped {total_skipped}.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
