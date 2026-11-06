#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 06 — run a trained double-tetrad model on an arbitrary structure.

Given a structure file and a bundled model pickle, this script:

1. Parses the structure and collects every **guanine** (RNA ``G`` or DNA
   ``DG``) with a C1' atom.
2. Builds a ``cKDTree`` over the guanine C1' coordinates and enumerates all
   8-tuples within a radius (no coplanarity pruning — two stacked tetrads span
   two parallel planes).
3. Canonicalises each candidate's order, computes the 280 geometric features,
   and applies the bundled scaler + classifier.
4. Runs non-maximum suppression: overlapping candidates (the true motif and
   its near-copies) are reduced to the best-scoring one, so each double tetrad
   is reported once. ``--no-nms`` turns this off.

The CSV holds the reported motifs (``--all-candidates`` writes every
candidate, with ``prediction`` = above threshold and ``final_prediction`` =
reported after NMS).

Predictions are derived from the positive-class probability at a 0.5 threshold
(rather than ``clf.predict``) so that the label and the reported probability
are always consistent — see the ``tetrad/`` README for the SVM Platt-scaling
motivation.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Residue, Structure

from features import calculate_geometric_features
from neighborhood import enumerate_tuples

GUANINE_NAMES = {"G", "DG"}
STANDARD_NUCLEOTIDES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU"}


def is_guanine(residue: Residue) -> bool:
    """RNA G, DNA DG, or a modified guanine that rnapolis maps to G.
    Shared rule with _11_run_inference.py so both pipelines see the same guanines."""
    if not residue.is_nucleotide:
        return False
    name = (residue.residue_name or "").upper()
    if name in GUANINE_NAMES:
        return True
    if name in STANDARD_NUCLEOTIDES:
        return False
    # Modified residue: trust rnapolis' atom-based mapping (lowercase g = DNA).
    return (residue.one_letter_name or "").upper() == "G"
DEFAULT_RADIUS = 22.0
DEFAULT_MAX_DISTANCE = 22.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=Path, help="Path to a .cif, .cif.gz or .pdb file")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-csv", type=Path)
    p.add_argument("--output-plot", type=Path)
    p.add_argument("--structure-model", type=int)
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    p.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    add_nms_arguments(p)
    return p.parse_args()


def load_model_bundle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as h:
        payload = pickle.load(h)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a bundled model artifact.")
    required = {"classifier", "classifier_name", "scaler", "feature_columns", "window_size"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{path} missing bundle keys: {', '.join(missing)}")
    n = payload["window_size"]
    if n != 8:
        raise ValueError(f"{path} window_size={n}, this script expects 8.")
    cols = payload["feature_columns"]
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"{path} has invalid feature_columns metadata.")
    clf = payload["classifier"]
    scaler = payload["scaler"]
    expected = len(cols)
    clf_n = getattr(clf, "n_features_in_", None)
    if clf_n is not None and int(clf_n) != expected:
        raise ValueError(f"{path}: classifier expects {clf_n} features, bundle lists {expected}.")
    scaler_n = getattr(scaler, "n_features_in_", None)
    if scaler_n is not None and int(scaler_n) != expected:
        raise ValueError(f"{path}: scaler expects {scaler_n} features, bundle lists {expected}.")
    return payload


def parse_structure_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        with gzip.open(path, "rt") as h:
            inner = Path(path.stem).suffix.lower()
            if inner == ".cif":
                return parse_cif_atoms(h)
            if inner == ".pdb":
                return parse_pdb_atoms(h)
            raise ValueError(f"Unsupported gzipped format: {path.name}")
    if suffix == ".cif":
        with open(path) as h:
            return parse_cif_atoms(h)
    if suffix == ".pdb":
        with open(path) as h:
            return parse_pdb_atoms(h)
    raise ValueError(f"Unsupported structure format '{suffix}'.")


def select_structure_model(atoms_df: pd.DataFrame, requested: Optional[int]) -> tuple[pd.DataFrame, Optional[int]]:
    fmt = atoms_df.attrs.get("format")
    col = "pdbx_PDB_model_num" if fmt == "mmCIF" else ("model" if fmt == "PDB" else None)
    if col is None or col not in atoms_df.columns:
        return atoms_df, None
    models = list(dict.fromkeys(int(v) for v in atoms_df[col].dropna().tolist() if not pd.isna(v)))
    if not models:
        return atoms_df, None
    sel = requested if requested is not None else models[0]
    if sel not in models:
        raise ValueError(f"Model {sel} not found. Available: {models}")
    out = atoms_df[atoms_df[col] == sel].copy()
    out.attrs = atoms_df.attrs.copy()
    return out, sel


def clean_text(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    return str(v)


def residue_id_str(r: Residue) -> str:
    return f"{clean_text(r.chain_id)}:{r.residue_number}{clean_text(r.insertion_code)}"


def collect_guanines(structure: Structure) -> tuple[list[Residue], list[tuple[float, float, float]], list[tuple[str, int, str]]]:
    residues: list[Residue] = []
    coords: list[tuple[float, float, float]] = []
    ids: list[tuple[str, int, str]] = []
    for r in structure.residues:
        if not is_guanine(r):
            continue
        col_atom = r._col("atom_name")
        c1 = r.atoms[r.atoms[col_atom] == "C1'"]
        if c1.empty:
            continue
        row = c1.iloc[0]
        residues.append(r)
        coords.append((float(row[r._col("x")]), float(row[r._col("y")]), float(row[r._col("z")])))
        ids.append((r.chain_id, r.residue_number, r.insertion_code or ""))
    return residues, coords, ids


def positive_class_probabilities(clf: Any, X: np.ndarray, positive_label: Any):
    if not hasattr(clf, "predict_proba"):
        return None
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", []))
    if positive_label in classes:
        idx = classes.index(positive_label)
    elif len(classes) == 2:
        idx = 1
    else:
        return None
    return proba[:, idx]


# ─────────────────────────────────────────────────────────────────────────────
# Non-maximum suppression
# Overlapping candidates (the true double tetrad plus near-copies that swap one
# or two guanines, or pair the wrong tetrads) all score high. Keep the best
# candidate, drop the ones that overlap it, repeat.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.5
# Two real double tetrads of one stack (tetrads 1+2 and 2+3) share one tetrad,
# i.e. 4 residues, so picks may share up to 4 residues.
DEFAULT_NMS_MAX_SHARED = 4


def select_non_overlapping(
    residue_sets: list,
    scores,
    threshold: float = DEFAULT_THRESHOLD,
    max_shared: int = DEFAULT_NMS_MAX_SHARED,
    suppress_covered: bool = False,
) -> list[int]:
    """Greedy NMS. Returns the indices of the kept candidates, best first.

    A candidate is kept only if its score is >= threshold and it shares at
    most ``max_shared`` residues with every candidate kept before it. With
    ``suppress_covered`` it is also dropped when all of its residues already
    belong to kept candidates (e.g. the tetrad 1 + tetrad 3 chimera once both
    real pairs 1+2 and 2+3 have been kept).
    """
    scores = np.asarray(scores, dtype=float)
    kept: list[int] = []
    covered: set = set()
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < threshold:
            break
        candidate = residue_sets[i]
        if any(len(candidate & residue_sets[j]) > max_shared for j in kept):
            continue
        if suppress_covered and candidate <= covered:
            continue
        kept.append(int(i))
        covered |= candidate
    return kept


def apply_nms(results_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Add 'final_prediction' (and 'nms_rank') to the per-candidate table."""
    if "probability" not in results_df:
        print("Classifier has no predict_proba; NMS skipped, final_prediction = prediction.")
        results_df["final_prediction"] = results_df["prediction"].astype(bool)
        return results_df
    if args.no_nms:
        results_df["final_prediction"] = results_df["probability"] >= args.threshold
        return results_df

    residue_sets = [frozenset(ids.split(",")) for ids in results_df["residue_ids"]]
    kept = select_non_overlapping(
        residue_sets,
        results_df["probability"].to_numpy(),
        threshold=args.threshold,
        max_shared=args.nms_max_shared,
        suppress_covered=args.suppress_covered,
    )
    results_df["final_prediction"] = False
    results_df["nms_rank"] = pd.array([pd.NA] * len(results_df), dtype="Int64")
    for rank, i in enumerate(kept, start=1):
        results_df.loc[results_df.index[i], "final_prediction"] = True
        results_df.loc[results_df.index[i], "nms_rank"] = rank
    return results_df


def add_nms_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Probability threshold for a positive candidate (default: %(default)s)",
    )
    parser.add_argument(
        "--nms-max-shared", type=int, default=DEFAULT_NMS_MAX_SHARED,
        help="Residues two reported motifs may share (default: %(default)s = one tetrad)",
    )
    parser.add_argument(
        "--suppress-covered", action="store_true",
        help="Also drop candidates whose residues are all covered by motifs already reported",
    )
    parser.add_argument(
        "--no-nms", action="store_true",
        help="Report every candidate above the threshold (no suppression)",
    )
    parser.add_argument(
        "--all-candidates", action="store_true",
        help="Write every candidate to the CSV, not only the reported motifs",
    )


def report_motifs(results_df: pd.DataFrame) -> None:
    final = results_df[results_df["final_prediction"]]
    if "probability" in final:
        final = final.sort_values("probability", ascending=False)
    print(f"Reported {len(final)} double tetrad(s):")
    for _, row in final.iterrows():
        p = f"  p={row['probability']:.3f}" if "probability" in row else ""
        print(f"  {row['residue_ids']}{p}")


def save_probability_plot(df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(df["candidate_index"], df["probability"], marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Candidate index")
    plt.ylabel("Positive-class probability")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model)
    n = bundle["window_size"]
    feature_columns = bundle["feature_columns"]
    clf = bundle["classifier"]
    scaler = bundle["scaler"]
    positive_label = bundle.get("positive_label", True)

    atoms_df = parse_structure_file(args.structure)
    atoms_df, selected_model = select_structure_model(atoms_df, args.structure_model)
    structure = Structure(atoms_df)
    residues, coords, ids = collect_guanines(structure)

    if len(residues) < n:
        raise ValueError(f"Only {len(residues)} guanines with C1' in {args.structure.name}; need at least {n}.")

    candidates = enumerate_tuples(
        coords,
        n=n,
        radius=args.radius,
        residue_ids=ids,
        max_distance=args.max_distance,
        dedup=True,
    )

    if not candidates:
        raise ValueError(f"No candidate {n}-tuples found (try relaxing --radius / --max-distance).")

    feature_rows = []
    metadata_rows = []
    for k, (indices, _ordered) in enumerate(candidates, start=1):
        sub_coords = [coords[i] for i in indices]
        feats = calculate_geometric_features(sub_coords, n=n)
        feature_rows.append(feats)
        sel_residues = [residues[i] for i in indices]
        metadata_rows.append(
            {
                "candidate_index": k,
                "structure": args.structure.name,
                "structure_model": selected_model,
                "classifier_name": bundle["classifier_name"],
                "model_path": str(args.model),
                "residue_ids": ",".join(residue_id_str(r) for r in sel_residues),
                "sequence": "".join((r.one_letter_name or "?") for r in sel_residues),
                "centroid_x": float(np.mean([c[0] for c in sub_coords])),
                "centroid_y": float(np.mean([c[1] for c in sub_coords])),
                "centroid_z": float(np.mean([c[2] for c in sub_coords])),
            }
        )

    features_df = pd.DataFrame(feature_rows)
    drop_cols = [c for c in features_df.columns if c not in feature_columns]
    if drop_cols:
        features_df = features_df.drop(columns=drop_cols)
    feature_matrix = features_df.reindex(columns=feature_columns)
    if feature_matrix.isna().any().any():
        raise ValueError("Feature matrix contains NaN values after reindexing; refusing to run inference.")

    transformed = scaler.transform(feature_matrix)
    probabilities = positive_class_probabilities(clf, transformed, positive_label)
    if probabilities is not None:
        predictions = probabilities >= args.threshold
    else:
        predictions = np.asarray(clf.predict(transformed)) == positive_label

    results = pd.DataFrame(metadata_rows)
    results["prediction"] = predictions
    if probabilities is not None:
        results["probability"] = probabilities

    results = apply_nms(results, args)
    to_write = results if args.all_candidates else results[results["final_prediction"]]

    out_csv = args.output_csv if args.output_csv is not None else args.structure.with_name(
        f"{args.structure.stem}-{args.model.stem}-predictions.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    to_write.to_csv(out_csv, index=False)

    n_pos = int(np.asarray(predictions, dtype=bool).sum())
    print(f"Parsed {len(residues)} guanines; generated {len(results)} candidates "
          f"({n_pos} above threshold).")
    report_motifs(results)
    print(f"Saved {'all candidates' if args.all_candidates else 'reported motifs'} to {out_csv}")

    if args.output_plot is not None:
        if probabilities is None:
            print("Classifier does not expose predict_proba; skipping probability plot.")
        else:
            args.output_plot.parent.mkdir(parents=True, exist_ok=True)
            save_probability_plot(results, args.output_plot)
            print(f"Saved probability plot to {args.output_plot}")


if __name__ == "__main__":
    main()