#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-evaluate every bundled model (*.pkl) on a folder of mmCIF structures.

Models whose file name contains "-gat" are run through run_model_graph (GATv2);
all other models are run through run_model (classical sklearn).

For every model, a CSV is written to the result folder with one row per
candidate 8-guanine tuple: which residues it contains, whether the model
predicts the motif, and the positive-class probability.

If an answers JSON is given, a ground_truth column is added, and a
<model>_quality.txt report with TP / FP / TN / FN counts is written as well.

Place this file next to run_model.py and run_model_graph.py (and the
features.py / neighborhood.py modules they import).

Example:
    uv run _13_test_all_models.py -structures test_cif_small/  -result_folder models_test/  -answers  double_tetrad_motifs_by_pdb.json 
_13_test_all_models.py: error: the following arguments are required: structures
    """

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from rnapolis.tertiary_v2 import Structure

import _06_run_inference as run_model
import _11_run_inference as run_model_graph

DEFAULT_RADIUS = 22.0
DEFAULT_MAX_DISTANCE = 22.0
DEFAULT_RESULT_FOLDER = Path("models_test")
STRUCTURE_SUFFIXES = (".cif", ".cif.gz")

ResidueKey = tuple[str, int, str]  # (chain_id, residue_number, insertion_code)
MotifKey = frozenset  # frozenset[ResidueKey], order-independent


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-structures",
        type=Path,
        help="Folder containing the .cif / .cif.gz files to test against",
    )
    p.add_argument(
        "-model_folder", "--model-folder",
        dest="model_folder",
        type=Path,
        default=None,
        help="Folder with the .pkl model bundles (default: current working directory)",
    )
    p.add_argument(
        "-result_folder", "-resultfolder", "--result-folder",
        dest="result_folder",
        type=Path,
        default=DEFAULT_RESULT_FOLDER,
        help="Folder for the output CSVs and quality reports (default: %(default)s)",
    )
    p.add_argument(
        "-answers", "--answers",
        dest="answers",
        type=Path,
        default=None,
        help="JSON file with the known motif locations (enables ground_truth + quality reports)",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS,
        help="KD-tree neighbourhood radius in Angstroms (default: %(default)s)",
    )
    p.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help=(
            "Maximum pairwise C1' distance within a candidate, in Angstroms "
            "(default: %(default)s). Applied to both pipelines so their "
            "candidate sets are comparable."
        ),
    )
    p.add_argument(
        "--structure-model",
        type=int,
        default=None,
        help="1-based model number for multi-model files (default: first model)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StructureStats:
    structure: str
    candidates: int
    tp: int
    fp: int
    tn: int
    fn: int
    known_motifs: int
    missed_by_generation: int


@dataclass
class LoadedModel:
    name: str
    path: Path
    pipeline: str  # "classical" or "graph"
    bundle: dict[str, Any]
    window_size: int
    csv_path: Path
    quality_path: Path
    rows_written: int = 0
    has_probability: Optional[bool] = None
    stats: list[StructureStats] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


def peek_window_size(path: Path) -> int:
    """Read window_size from a bundle so run_model_graph's loader accepts it."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "window_size" in payload:
        return int(payload["window_size"])
    return run_model_graph.DEFAULT_NT_NUMBER


def load_models(model_folder: Path, result_folder: Path) -> list[LoadedModel]:
    pkl_files = sorted(model_folder.glob("*.pkl"))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found in {model_folder.resolve()}")

    models: list[LoadedModel] = []
    for path in pkl_files:
        is_graph = "-gat" in path.name.lower()
        pipeline = "graph" if is_graph else "classical"
        try:
            if is_graph:
                bundle = run_model_graph.load_model_bundle(path, peek_window_size(path))
            else:
                bundle = run_model.load_model_bundle(path)
        except Exception as exc:  # noqa: BLE001 — report and keep going
            print(f"[skip model] {path.name}: {exc}", file=sys.stderr)
            continue

        models.append(
            LoadedModel(
                name=path.stem,
                path=path,
                pipeline=pipeline,
                bundle=bundle,
                window_size=int(bundle["window_size"]),
                csv_path=result_folder / f"{path.stem}.csv",
                quality_path=result_folder / f"{path.stem}_quality.txt",
            )
        )
        print(f"Loaded {path.name} -> {pipeline} pipeline")

    if not models:
        raise SystemExit("None of the .pkl files could be loaded.")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Structures and ground truth
# ─────────────────────────────────────────────────────────────────────────────

def find_structures(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    files = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.name.lower().endswith(STRUCTURE_SUFFIXES)
    ]
    if not files:
        raise SystemExit(f"No .cif / .cif.gz files found in {folder.resolve()}")
    return files


def structure_name(path: Path) -> str:
    """'139d-assembly1.cif.gz' -> '139d-assembly1' (matches the answers JSON keys)."""
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    for suffix in (".cif", ".mmcif", ".pdb"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_structure(path: Path, requested_model: Optional[int]) -> Structure:
    # run_model's parser also handles .cif.gz, so it is used for both pipelines.
    atoms_df = run_model.parse_structure_file(path)
    atoms_df, _ = run_model.select_structure_model(atoms_df, requested_model)
    return Structure(atoms_df)


def residue_key(chain_id: Any, residue_number: Any, insertion_code: Any) -> ResidueKey:
    icode = run_model.clean_text(insertion_code).strip()
    if icode in {"?", "."}:
        icode = ""
    return (run_model.clean_text(chain_id).strip(), int(residue_number), icode)


def load_answers(path: Path) -> dict[str, set[MotifKey]]:
    with open(path) as handle:
        data = json.load(handle)

    answers: dict[str, set[MotifKey]] = {}
    for struct, motifs in data.items():
        keys: set[MotifKey] = set()
        for motif in motifs or []:
            keys.add(
                frozenset(
                    residue_key(u["chain_id"], u["residue_number"], u.get("insertion_code"))
                    for u in motif["unit_ids"]
                )
            )
        answers[struct.lower()] = keys

    n_motifs = sum(len(v) for v in answers.values())
    print(f"Loaded {n_motifs} known motifs for {len(answers)} structures from {path.name}")
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation — mirrors each script's own pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateSet:
    residue_ids: list[str]  # "A:3,D:3,..." for the CSV
    keys: list[MotifKey]  # for ground-truth matching
    features: pd.DataFrame


def build_classical_candidates(
    structure: Structure, n: int, radius: float, max_distance: Optional[float]
) -> Optional[CandidateSet]:
    """Same steps as run_model.main(): guanines -> KD-tree tuples -> features."""
    residues, coords, ids = run_model.collect_guanines(structure)
    if len(residues) < n:
        return None

    candidates = list(
        run_model.enumerate_tuples(
            coords, n=n, radius=radius, residue_ids=ids,
            max_distance=max_distance, dedup=True,
        )
    )
    if not candidates:
        return None

    residue_ids, keys, feature_rows = [], [], []
    for indices, _ordered in candidates:
        selected = [residues[i] for i in indices]
        residue_ids.append(",".join(run_model.residue_id_str(r) for r in selected))
        keys.append(
            frozenset(residue_key(r.chain_id, r.residue_number, r.insertion_code) for r in selected)
        )
        # Same coordinate order as run_model.py (indices, not _ordered).
        feature_rows.append(
            run_model.calculate_geometric_features([coords[i] for i in indices], n=n)
        )

    return CandidateSet(residue_ids, keys, pd.DataFrame(feature_rows))


def build_graph_candidates(
    structure: Structure, n: int, radius: float, max_distance: Optional[float]
) -> Optional[CandidateSet]:
    """Same steps as run_model_graph.build_candidate_rows()."""
    guanines = run_model_graph.get_guanine_residues(structure)

    coords, ids, valid = [], [], []
    for residue in guanines:
        c1 = run_model_graph.first_c1_prime_coordinates(residue)
        if c1 is None:
            continue
        coords.append(c1)
        ids.append(
            (
                run_model_graph.clean_text(residue.chain_id),
                residue.residue_number,
                run_model_graph.clean_text(residue.insertion_code),
            )
        )
        valid.append(residue)

    if len(valid) < n:
        return None

    candidates = list(
        run_model_graph.enumerate_tuples(
            coords, n=n, radius=radius, residue_ids=ids,
            max_distance=max_distance, dedup=True,
        )
    )
    if not candidates:
        return None

    residue_ids, keys, feature_rows = [], [], []
    for indices, ordered_coords in candidates:
        selected = [valid[i] for i in indices]
        residue_ids.append(",".join(run_model_graph.format_residue_id(r) for r in selected))
        keys.append(
            frozenset(residue_key(r.chain_id, r.residue_number, r.insertion_code) for r in selected)
        )
        # Same coordinate order as run_model_graph.py (ordered_coords).
        feature_rows.append(
            run_model_graph.calculate_window_features(list(ordered_coords), n)
        )

    return CandidateSet(residue_ids, keys, pd.DataFrame(feature_rows))


BUILDERS: dict[str, Callable[..., Optional[CandidateSet]]] = {
    "classical": build_classical_candidates,
    "graph": build_graph_candidates,
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score(model: LoadedModel, cands: CandidateSet) -> tuple[np.ndarray, Optional[np.ndarray]]:
    bundle = model.bundle
    feature_columns = bundle["feature_columns"]

    missing = [c for c in feature_columns if c not in cands.features.columns]
    if missing:
        raise ValueError(
            f"generated features lack {len(missing)} columns the model expects "
            f"(e.g. {', '.join(missing[:5])})"
        )

    feature_matrix = cands.features.reindex(columns=feature_columns)
    if feature_matrix.isna().any().any():
        raise ValueError("feature matrix contains NaN values")

    transformed = bundle["scaler"].transform(feature_matrix)
    classifier = bundle["classifier"]
    positive_label = bundle.get("positive_label", True)

    probabilities = run_model.positive_class_probabilities(
        classifier, transformed, positive_label
    )
    if probabilities is not None:
        # Thresholding the probability keeps label and probability consistent
        # (and runs the GAT once instead of twice).
        predictions = np.asarray(probabilities) >= 0.5
    else:
        predictions = np.asarray(classifier.predict(transformed)) == positive_label

    return predictions.astype(bool), probabilities


def append_csv(model: LoadedModel, df: pd.DataFrame) -> None:
    first = model.rows_written == 0
    df.to_csv(model.csv_path, mode="w" if first else "a", header=first, index=False)
    model.rows_written += len(df)


# ─────────────────────────────────────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────────────────────────────────────

def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def write_quality_report(
    model: LoadedModel, answers_path: Path, not_in_answers: list[str]
) -> tuple[int, int, int, int]:
    s = model.stats
    tp = sum(x.tp for x in s)
    fp = sum(x.fp for x in s)
    tn = sum(x.tn for x in s)
    fn = sum(x.fn for x in s)
    total = tp + fp + tn + fn
    known = sum(x.known_motifs for x in s)
    missed = sum(x.missed_by_generation for x in s)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = _ratio(tp + tn, total)
    specificity = _ratio(tn, tn + fp)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den else float("nan")

    if model.has_probability is False:
        rule = "classifier.predict() (no predict_proba available)"
    else:
        rule = "positive if P(motif) >= 0.5"

    lines = [
        f"Model:              {model.name}",
        f"Model file:         {model.path}",
        f"Pipeline:           {'run_model_graph (GAT)' if model.pipeline == 'graph' else 'run_model (classical)'}",
        f"Classifier:         {model.bundle.get('classifier_name', '?')}",
        f"Answers file:       {answers_path}",
        f"Decision rule:      {rule}",
        "",
        f"Structures scored:  {len(s)}",
        f"Candidates scored:  {total}",
        "",
        "Confusion matrix (over generated candidates)",
        f"  True positives  (TP): {tp}",
        f"  False positives (FP): {fp}",
        f"  True negatives  (TN): {tn}",
        f"  False negatives (FN): {fn}",
        "",
        f"  Precision:    {precision:.4f}",
        f"  Recall:       {recall:.4f}",
        f"  Specificity:  {specificity:.4f}",
        f"  F1:           {f1:.4f}",
        f"  Accuracy:     {accuracy:.4f}",
        f"  MCC:          {mcc:.4f}",
        "",
        "Candidate generation",
        f"  Known motifs in scored structures:  {known}",
        f"  Present among candidates (TP + FN): {tp + fn}",
        f"  Never generated as candidates:      {missed}",
        f"  End-to-end recall (TP / known):     {_ratio(tp, known):.4f}",
        "  (Motifs never generated are not part of the confusion matrix: they were",
        "   lost at the KD-tree step — radius / max-distance — before the model saw them.)",
    ]

    if s:
        lines += ["", "Per-structure breakdown", pd.DataFrame([asdict(x) for x in s]).to_string(index=False)]

    if not_in_answers:
        lines += [
            "",
            f"Structures absent from the answers file ({len(not_in_answers)}), "
            "treated as containing no motifs:",
        ] + [f"  {name}" for name in not_in_answers]

    if model.failures:
        lines += ["", f"Structures that failed for this model ({len(model.failures)}):"]
        lines += [f"  {name}: {err}" for name, err in model.failures]

    model.quality_path.write_text("\n".join(lines) + "\n")
    return tp, fp, tn, fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    model_folder = args.model_folder if args.model_folder is not None else Path.cwd()
    result_folder: Path = args.result_folder
    result_folder.mkdir(parents=True, exist_ok=True)

    models = load_models(model_folder, result_folder)
    structures = find_structures(args.structures)
    answers = load_answers(args.answers) if args.answers is not None else None
    not_in_answers: list[str] = []

    print(f"\nTesting {len(models)} model(s) on {len(structures)} structure(s)\n")

    for s_idx, path in enumerate(structures, start=1):
        name = structure_name(path)
        print(f"[{s_idx}/{len(structures)}] {path.name}")

        try:
            structure = load_structure(path, args.structure_model)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not parse: {exc}", file=sys.stderr)
            for model in models:
                model.failures.append((name, f"parse error: {exc}"))
            continue

        known: set[MotifKey] = set()
        if answers is not None:
            if name.lower() in answers:
                known = answers[name.lower()]
            else:
                not_in_answers.append(name)

        # Candidates depend only on the pipeline and tuple size, so build them
        # once per structure and reuse them for every model of that kind.
        cache: dict[tuple[str, int], Any] = {}

        for model in models:
            cache_key = (model.pipeline, model.window_size)
            if cache_key not in cache:
                try:
                    cache[cache_key] = BUILDERS[model.pipeline](
                        structure, model.window_size, args.radius, args.max_distance
                    )
                except Exception as exc:  # noqa: BLE001
                    cache[cache_key] = exc
            cands = cache[cache_key]

            if isinstance(cands, Exception):
                print(f"  {model.name}: candidate generation failed: {cands}", file=sys.stderr)
                model.failures.append((name, f"candidate generation: {cands}"))
                continue

            if cands is None:
                print(f"  {model.name}: no candidates")
                if answers is not None:
                    model.stats.append(StructureStats(name, 0, 0, 0, 0, 0, len(known), len(known)))
                continue

            try:
                predictions, probabilities = score(model, cands)
            except Exception as exc:  # noqa: BLE001
                print(f"  {model.name}: scoring failed: {exc}", file=sys.stderr)
                model.failures.append((name, f"scoring: {exc}"))
                continue

            model.has_probability = probabilities is not None

            df = pd.DataFrame(
                {
                    "structure": name,
                    "candidate_index": np.arange(1, len(cands.residue_ids) + 1),
                    "residue_ids": cands.residue_ids,
                    "prediction": predictions,
                }
            )
            if probabilities is not None:
                df["probability"] = np.asarray(probabilities, dtype=float)

            n_pos = int(predictions.sum())
            msg = f"  {model.name}: {len(df)} candidates, {n_pos} predicted positive"

            if answers is not None:
                truth = np.array([k in known for k in cands.keys], dtype=bool)
                df["ground_truth"] = truth
                stats = StructureStats(
                    structure=name,
                    candidates=len(df),
                    tp=int((predictions & truth).sum()),
                    fp=int((predictions & ~truth).sum()),
                    tn=int((~predictions & ~truth).sum()),
                    fn=int((~predictions & truth).sum()),
                    known_motifs=len(known),
                    missed_by_generation=len(known - set(cands.keys)),
                )
                model.stats.append(stats)
                msg += f", {int(truth.sum())}/{len(known)} known motifs among candidates"
                if known and not truth.any():
                    msg += "  <-- check chain IDs / residue numbering / radius"

            print(msg)
            append_csv(model, df)

    # Finalise outputs
    print()
    for model in models:
        if model.rows_written == 0:
            columns = ["structure", "candidate_index", "residue_ids", "prediction", "probability"]
            if answers is not None:
                columns.append("ground_truth")
            pd.DataFrame(columns=columns).to_csv(model.csv_path, index=False)

        line = f"{model.name}: {model.rows_written} rows -> {model.csv_path}"
        if answers is not None:
            tp, fp, tn, fn = write_quality_report(model, args.answers, not_in_answers)
            line += f"  | TP={tp} FP={fp} TN={tn} FN={fn} -> {model.quality_path.name}"
        if model.failures:
            line += f"  ({len(model.failures)} structure(s) failed)"
        print(line)


if __name__ == "__main__":
    main()