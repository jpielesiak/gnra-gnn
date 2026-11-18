#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 03 — generate negative examples (non-double-tetrad 8-tuples) via KD-tree.

For every (pdb, assembly) stem that contributed positive double tetrads:

1. Parse the assembly mmCIF and collect every **guanine** with C1' (RNA G,
   DNA DG, modified G) — the same residues the inference scripts enumerate,
   so the negatives come from the same distribution as the candidates the
   models will score. ``--all-nucleotides`` restores the old behaviour.
2. Build a ``cKDTree`` over the C1' coordinates.
3. Enumerate all 8-tuples within a radius (no coplanarity pruning — 8 points
   from two stacked tetrads span two parallel planes, not one).
4. Apply **tuple-level exclusion**: ignore candidates sharing more than
   ``max_shared_with_tetrad`` residues with any annotated double tetrad.
5. Split the remaining candidates into
   * **hard** negatives: share >= ``hard_min_shared`` residues with an
     annotated double tetrad (one guanine swapped, wrong pair of tetrads,
     slices through a stack) — exactly the near-copies that inference
     produces around every real motif;
   * **easy** negatives: everything else.
6. Sample up to ``hard_k × #positives`` hard and ``K × #positives`` easy
   negatives per file, canonicalise, write CIF.
   Hard negatives are written as ``NEG_<stem>_hNNNN.cif``, easy ones as
   ``NEG_<stem>_NNNN.cif``.

Defaults: ``radius = 22 Å``, ``max_distance = 22 Å``, ``K = 5``,
``hard_k = 5``, ``hard_min_shared = 3``, ``max_shared_with_tetrad = 7``
(only the motif itself, 8/8, is ignored; a 7/8 candidate has a guanine from
the wrong tetrad and is a negative).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import zlib
from functools import lru_cache
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
DEFAULT_HARD_K = 5
DEFAULT_HARD_MIN_SHARED = 3
DEFAULT_SEED = 42
DEFAULT_MAX_SHARED = 7

GUANINE_NAMES = {"G", "DG"}
STANDARD_NUCLEOTIDES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU"}


def is_guanine(residue: Residue) -> bool:
    """Same rule as _06_run_inference.py / _11_run_inference.py."""
    if not residue.is_nucleotide:
        return False
    name = (residue.residue_name or "").upper()
    if name in GUANINE_NAMES:
        return True
    if name in STANDARD_NUCLEOTIDES:
        return False
    return (residue.one_letter_name or "").upper() == "G"


@lru_cache(maxsize=None)
def _folder_index(folder: str) -> Dict[str, str]:
    """lower-case file name -> path, for every file directly in folder."""
    if not os.path.isdir(folder):
        return {}
    return {name.lower(): os.path.join(folder, name) for name in os.listdir(folder)}


def mmcif_path(stem: str, mirror_dir: str) -> Optional[str]:
    """Find <stem>.cif(.gz) (case-insensitive) directly in <mirror>/ or in the
    PDB 'divided' layout <mirror>/<middle two chars>/. None if not found."""
    pdb_id = stem.split("-")[0].lower()
    for folder in (mirror_dir, os.path.join(mirror_dir, pdb_id[1:3])):
        index = _folder_index(folder)
        for name in (f"{stem}.cif.gz", f"{stem}.cif"):
            path = index.get(name.lower())
            if path is not None:
                return path
    return None


def open_structure(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def collect_nucleotides(
    structure: Structure, guanines_only: bool = True,
) -> Tuple[List[Residue], List[Tuple[float, float, float]], List[Tuple[str, int, str]]]:
    residues: List[Residue] = []
    coords: List[Tuple[float, float, float]] = []
    ids: List[Tuple[str, int, str]] = []
    for r in structure.residues:
        if not r.is_nucleotide:
            continue
        if guanines_only and not is_guanine(r):
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
    hard_k: int,
    hard_min_shared: int,
    guanines_only: bool,
) -> Tuple[str, int, int]:
    path = mmcif_path(stem, mirror_dir)
    if path is None:
        print(f"  Warning: no {stem}.cif.gz / {stem}.cif in {mirror_dir}, skipping")
        return stem, -1, 0

    try:
        with open_structure(path) as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)
    except Exception as e:
        print(f"  Error parsing {stem}: {e}")
        return stem, 0, 0

    residues, coords, ids = collect_nucleotides(structure, guanines_only)
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

    # Split into hard (overlapping an annotated motif) and easy negatives.
    def max_shared(indices) -> int:
        key = frozenset(ids[i] for i in indices)
        return max((len(key & motif) for motif in tuple_exclusion), default=0)

    hard, easy = [], []
    for cand in candidates:
        (hard if max_shared(cand[0]) >= hard_min_shared else easy).append(cand)

    rng = np.random.default_rng(seed)

    def pick(pool, count):
        if len(pool) <= count:
            return list(pool)
        idx = rng.choice(len(pool), size=count, replace=False)
        return [pool[i] for i in idx]

    sampled = [("h", c) for c in pick(hard, hard_k * positive_count)]
    sampled += [("", c) for c in pick(easy, k * positive_count)]

    n_saved = 0
    n_skipped = 0
    counters = {"h": 0, "": 0}
    for prefix, (indices, _ordered) in sampled:
        j = counters[prefix]
        counters[prefix] += 1
        out_file = os.path.join(output_dir, f"NEG_{stem}_{prefix}{j:04d}.cif")
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
            print(f"    Error saving NEG_{stem}_{prefix}{j:04d}: {e}")
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
    parser.add_argument(
        "--mirror", default=DEFAULT_MIRROR,
        help="Folder with the assembly files <pdb>-assembly<N>.cif(.gz), either flat "
             "or in the PDB 'divided' layout (default: %(default)s)",
    )
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
        "being ignored (default: 7 → only 8/8, the motif itself, is ignored)",
    )
    parser.add_argument("--hard-k", type=int, default=DEFAULT_HARD_K,
                        help="Hard negatives per positive (default: %(default)s)")
    parser.add_argument("--hard-min-shared", type=int, default=DEFAULT_HARD_MIN_SHARED,
                        help="Residues shared with a motif that make a negative 'hard' "
                             "(default: %(default)s)")
    parser.add_argument("--all-nucleotides", action="store_true",
                        help="Enumerate all nucleotides, not only guanines (old behaviour)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.endswith(".cif")]
    if existing:
        raise SystemExit(
            f"{output_dir} already contains {len(existing)} .cif files. Existing files "
            "are never overwritten, so old negatives would silently stay. "
            "Delete the folder or pass a new --output-dir."
        )

    with open(args.motifs) as f:
        motifs_by_pdb: Dict[str, List[Dict[str, Any]]] = json.load(f)

    print(f"Loaded {sum(len(v) for v in motifs_by_pdb.values())} positives across {len(motifs_by_pdb)} stems")
    print(
        f"Params: radius={args.radius} Å, max_distance={args.max_distance} Å, "
        f"K={args.k}, hard_K={args.hard_k}, hard_min_shared={args.hard_min_shared}, "
        f"seed={args.seed}, max_shared={args.max_shared_with_tetrad}, "
        f"guanines_only={not args.all_nucleotides}"
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
                # zlib.crc32 is stable across runs; hash() is randomised per process
                args.seed + zlib.crc32(stem.encode()),
                args.max_shared_with_tetrad,
                args.hard_k,
                args.hard_min_shared,
                not args.all_nucleotides,
            )
        )

    total_saved = 0
    total_skipped = 0
    missing: List[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_wrapper, t): t[0] for t in tasks}
        for future in as_completed(futures):
            stem = futures[future]
            try:
                _, saved, skipped = future.result()
                if saved < 0:
                    missing.append(stem)
                    continue
                total_saved += saved
                total_skipped += skipped
                if saved:
                    print(f"  {stem}: saved {saved} negatives, skipped {skipped}")
            except Exception as e:
                print(f"  Error processing {stem}: {e}")

    print(f"\nDone. Saved {total_saved} negatives, skipped {total_skipped}.")
    if missing:
        print(f"WARNING: {len(missing)} of {len(tasks)} structures had no mmCIF file in "
              f"{args.mirror}; their positives have no negatives. Missing list -> "
              f"missing_structures.txt")
        Path("missing_structures.txt").write_text("\n".join(sorted(missing)) + "\n")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()