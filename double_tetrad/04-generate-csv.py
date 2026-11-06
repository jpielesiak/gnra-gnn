#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 04 — compute geometric features for every positive / negative CIF.

Reads per-double-tetrad CIF files from ``motif_cif_files/`` (positives, label
``double_tetrad = True``) and ``negative_cif_files/`` (negatives, label
``double_tetrad = False``), extracts the eight C1' atoms from each, computes
the 280 geometric features via ``features.calculate_geometric_features``, and
writes ``geometric_features.csv``.

**Phase augmentation** (on by default).  Positives are stored topologically
ordered (step 02), but at inference candidates arrive in the order produced by
the geometric canonizer (``canonical_order.canonicalize``), whose cyclic phase
depends on the crystal frame.  To make the trained model phase-tolerant, every
positive is additionally emitted in every cyclic phase the canonizer can
produce: the C1' points are randomly rotated (rigid motion), re-canonicalized,
and the resulting ordering is recorded as an extra *positive* row.  Sampling
stops when no new phase appears, so each motif yields its full phase family
(typically 10-14 variants).

**Shuffled hard negatives** (1 per positive by default).  A random permutation
of a true double tetrad has plausible geometry but a non-canonical index
structure; it teaches the model that the canonical layout itself matters.
Permutations that coincide with a valid phase are rejected.

All rows carry two bookkeeping columns:

* ``example_type`` — ``positive`` | ``phase_positive`` | ``geometric_negative``
  | ``shuffled_negative``;
* ``base_source_file`` — the motif key the row derives from.  Augmented rows
  inherit the base motif's ``source_file`` so that the cluster-based holdout in
  the training notebook keeps every phase/shuffle variant of a motif in the
  same split as the motif itself.
"""

from __future__ import annotations

import argparse
import glob
import os
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms

from canonical_order import canonicalize
from features import calculate_geometric_features
from topological_order import column_sequence_from_perm

N = 8
DEFAULT_SEED = 42
DEFAULT_MAX_ROTATION_SAMPLES = 400
DEFAULT_ROTATION_PATIENCE = 150


def c1_prime_coords_from_cif(path: str) -> Optional[List[Tuple[float, float, float]]]:
    try:
        with open(path) as f:
            atoms_df = parse_cif_atoms(f)
    except Exception as e:
        print(f"  Error parsing {path}: {e}")
        return None

    c1 = atoms_df[atoms_df["auth_atom_id"] == "C1'"]
    c1 = c1.drop_duplicates(
        subset=["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"], keep="first"
    )
    if len(c1) != N:
        print(f"  {os.path.basename(path)}: {len(c1)} C1' atoms (expected {N})")
        return None

    coords: List[Tuple[float, float, float]] = []
    for _, row in c1.iterrows():
        coords.append((float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])))
    return coords


def featurise_directory(
    directory: str, label: bool, example_type: str
) -> Tuple[List[pd.DataFrame], Dict[str, List[Tuple[float, float, float]]]]:
    files = sorted(glob.glob(os.path.join(directory, "*.cif")))
    print(f"Found {len(files)} .cif files in {directory}")
    rows: List[pd.DataFrame] = []
    coords_by_file: Dict[str, List[Tuple[float, float, float]]] = {}
    for fp in files:
        coords = c1_prime_coords_from_cif(fp)
        if coords is None:
            continue
        feats = calculate_geometric_features(coords, n=N)
        name = Path(fp).stem
        feats["source_file"] = name
        feats["base_source_file"] = name
        feats["example_type"] = example_type
        feats["double_tetrad"] = label
        rows.append(pd.DataFrame([feats]))
        coords_by_file[name] = coords
    return rows, coords_by_file


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random proper rotation (det = +1) as a 3x3 matrix."""
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def augment_positive(
    base_name: str,
    coords: List[Tuple[float, float, float]],
    shuffle_multiplier: int,
    max_rotation_samples: int,
    rotation_patience: int,
    seed: int,
) -> Tuple[List[pd.DataFrame], int]:
    """Generate phase-variant positives and shuffled hard negatives for one motif.

    ``coords`` are the 8 C1' points in the topological order stored by step 02.
    Returns (rows, n_phases); rows are feature dicts plus the bookkeeping
    columns.  Feature geometry is rigid-motion invariant, so features are
    computed from the original points reordered by each canonizer output.
    """
    pts = np.asarray(coords, dtype=float)
    rng = np.random.default_rng(seed)

    # --- Phase variants: rigidly rotate, canonicalize, record the ordering ---
    phase_perms: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    stale = 0
    for _ in range(max_rotation_samples):
        rotated = pts @ _random_rotation(rng).T
        perm, _ = canonicalize([tuple(p) for p in rotated])
        seq = column_sequence_from_perm(perm)
        if seq in phase_perms:
            stale += 1
            if stale >= rotation_patience:
                break
            continue
        stale = 0
        phase_perms[seq] = tuple(perm)

    rows: List[pd.DataFrame] = []
    identity = tuple(range(N))
    for seq, perm in phase_perms.items():
        if seq == column_sequence_from_perm(identity):
            continue  # topological order itself — already featurised as the base row
        feats = calculate_geometric_features([tuple(pts[i]) for i in perm], n=N)
        feats["source_file"] = base_name
        feats["base_source_file"] = base_name
        feats["example_type"] = "phase_positive"
        feats["double_tetrad"] = True
        rows.append(pd.DataFrame([feats]))

    # --- Shuffled hard negatives: random non-phase permutations ---
    phase_seqs = set(phase_perms) | {column_sequence_from_perm(identity)}
    n_shuffled = 0
    attempts = 0
    while n_shuffled < shuffle_multiplier and attempts < 1000 * max(1, shuffle_multiplier):
        attempts += 1
        perm = tuple(int(i) for i in rng.permutation(N))
        if column_sequence_from_perm(perm) in phase_seqs:
            continue
        feats = calculate_geometric_features([tuple(pts[i]) for i in perm], n=N)
        feats["source_file"] = base_name
        feats["base_source_file"] = base_name
        feats["example_type"] = "shuffled_negative"
        feats["double_tetrad"] = False
        rows.append(pd.DataFrame([feats]))
        n_shuffled += 1

    return rows, len(phase_perms) - (1 if column_sequence_from_perm(identity) in phase_perms else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-dir",
        default=str(Path(__file__).parent / "motif_cif_files"),
    )
    parser.add_argument(
        "--negative-dir",
        default=str(Path(__file__).parent / "negative_cif_files"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "geometric_features.csv"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shuffle-per-positive", type=int, default=1,
                        help="Shuffled hard negatives per positive (default: 1)")
    parser.add_argument("--max-rotation-samples", type=int, default=DEFAULT_MAX_ROTATION_SAMPLES)
    parser.add_argument("--rotation-patience", type=int, default=DEFAULT_ROTATION_PATIENCE)
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip phase augmentation and shuffled negatives")
    args = parser.parse_args()

    print("Processing positive examples...")
    pos_rows, pos_coords = featurise_directory(args.positive_dir, label=True, example_type="positive")
    print("\nProcessing negative examples...")
    neg_rows, _ = featurise_directory(args.negative_dir, label=False, example_type="geometric_negative")

    all_rows = pos_rows + neg_rows
    n_phases_total = 0
    n_shuffled_total = 0

    if not args.no_augment and pos_coords:
        print(f"\nAugmenting {len(pos_coords)} positives (phase variants + shuffled negatives)...")
        for i, (name, coords) in enumerate(sorted(pos_coords.items()), start=1):
            # Deterministic per-motif seed regardless of processing order.
            motif_seed = args.seed + zlib.crc32(name.encode())
            aug_rows, n_phases = augment_positive(
                name,
                coords,
                shuffle_multiplier=args.shuffle_per_positive,
                max_rotation_samples=args.max_rotation_samples,
                rotation_patience=args.rotation_patience,
                seed=motif_seed,
            )
            all_rows.extend(aug_rows)
            n_phases_total += n_phases
            n_shuffled_total += sum(
                1 for r in aug_rows if r["example_type"].iloc[0] == "shuffled_negative"
            )
            if i % 200 == 0:
                print(f"  augmented {i}/{len(pos_coords)} motifs")

    if not all_rows:
        print("No valid samples found.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} samples to {args.output}")
    print(f"  base positives:        {len(pos_rows)}")
    print(f"  geometric negatives:   {len(neg_rows)}")
    if not args.no_augment:
        print(f"  phase positives:       {len(df) - len(pos_rows) - len(neg_rows) - n_shuffled_total}")
        print(f"  shuffled negatives:    {n_shuffled_total}")
        if pos_coords != 0:
            print(f"  (mean phase variants per motif: {n_phases_total / len(pos_coords):.1f})")
    print(f"Total columns: {len(df.columns)} "
          f"(4 bookkeeping + {len(df.columns) - 4} features)")


if __name__ == "__main__":
    main()
