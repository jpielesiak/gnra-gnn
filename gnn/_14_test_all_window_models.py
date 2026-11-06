#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-evaluate every bundled sliding-window model (*.pkl) on a folder of structures.

All *.pkl files in the model folder are run through the sliding-window runner
(the script that cuts every chain into consecutive N-nt windows). Classical
sklearn bundles and GATv2 bundles are both supported; the bundle type is
detected from the pickle contents, not from the file name.

For every model a CSV is written to the result folder with one row per window:
which residues it covers, whether the model predicts the motif, and the
positive-class probability.

If an answers JSON is given, a ground_truth column is added, a
<model>_quality.txt report (TP / FP / TN / FN, precision, recall, MCC, ...)
is written for each model, and summary.csv compares all models side by side.

Place this file next to the sliding-window runner script (or pass -runner).

Examples:
    uv run _14_test_all_window_models.py -structures cif_files/ -answers gnra_motifs_by_pdb.json \
        -runner 05-run-inference.py -result_folder window_test/ -match centered
    uv run _14_test_all_window_models.py -structures test_cif_small/ -answers motifs_by_pdb.json \
        -runner run_model_window.py -model_folder models/ -result_folder window_test/ -match centered
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import math
import pickle
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure

# File name of the sliding-window inference script (the one with
# build_window_rows / get_nucleotide_residues_by_chain). Override with -runner.
DEFAULT_RUNNER = "run_model_window.py"
DEFAULT_RESULT_FOLDER = Path("models_window_test")
STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".mmcif", ".mmcif.gz", ".pdb", ".pdb.gz")

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
        "-structures", "--structures",
        dest="structures",
        type=Path,
        required=True,
        help="Folder containing the .cif / .cif.gz files to test against",
    )
    p.add_argument(
        "-answers", "--answers",
        dest="answers",
        type=Path,
        default=None,
        help="JSON file with the known motif locations (enables ground_truth + quality reports)",
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
        "-runner", "--runner",
        dest="runner",
        type=Path,
        default=None,
        help=f"Path to the sliding-window inference script (default: {DEFAULT_RUNNER} next to this file)",
    )
    p.add_argument(
        "--match",
        choices=("exact", "contains", "centered", "overlap"),
        default="exact",
        help=(
            "When a window counts as a true motif: 'exact' = same residue set; "
            "'contains' = window covers every motif residue; 'overlap' = window "
            "covers at least --min-overlap of the motif's residues (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--min-overlap",
        type=float,
        default=0.5,
        help="Fraction of motif residues a window must cover with --match overlap (default: %(default)s)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for a positive prediction (default: %(default)s)",
    )
    p.add_argument(
        "--structure-model",
        type=int,
        default=None,
        help="1-based model number for multi-model files (default: first model)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Runner import and pickle helpers
# ─────────────────────────────────────────────────────────────────────────────

def import_runner(path: Optional[Path]) -> ModuleType:
    """Import the sliding-window runner script so its GAT definition, feature
    code and window generation are reused verbatim."""
    if path is None:
        path = Path(__file__).resolve().parent / DEFAULT_RUNNER
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(
            f"Runner script not found: {path}\n"
            "Pass the sliding-window inference script with  -runner path/to/script.py"
        )

    sys.path.insert(0, str(path.parent))
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    needed = (
        "load_model_bundle",
        "get_nucleotide_residues_by_chain",
        "build_window_rows",
        "positive_class_probabilities",
    )
    missing = [name for name in needed if not hasattr(module, name)]
    if missing:
        raise SystemExit(
            f"{path.name} does not look like the sliding-window runner "
            f"(missing: {', '.join(missing)})"
        )
    print(f"Using runner {path.name}")
    return module


def allow_cuda_pickles_on_cpu() -> None:
    """GAT bundles pickled on a GPU machine contain CUDA tensors, and a plain
    pickle.load of them fails on a CPU-only machine. Map them to the CPU."""
    try:
        import torch
        import torch.storage
    except ImportError:
        return
    if torch.cuda.is_available() or not hasattr(torch.storage, "_load_from_bytes"):
        return

    def _load_on_cpu(data: bytes):
        try:
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        except TypeError:  # torch < 1.13 has no weights_only argument
            return torch.load(io.BytesIO(data), map_location="cpu")

    torch.storage._load_from_bytes = _load_on_cpu


def is_gat_payload(payload: dict) -> bool:
    return "model_state_dict" in payload and "classifier" not in payload


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StructureStats:
    structure: str
    windows: int
    tp: int
    fp: int
    tn: int
    fn: int
    known_motifs: int
    motifs_in_windows: int
    motifs_detected: int


@dataclass
class LoadedModel:
    name: str
    path: Path
    kind: str  # "GAT" or "classical"
    bundle: dict[str, Any]
    window_size: int
    csv_path: Path
    quality_path: Path
    rows_written: int = 0
    has_probability: Optional[bool] = None
    stats: list[StructureStats] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


def load_models(runner: ModuleType, model_folder: Path, result_folder: Path) -> list[LoadedModel]:
    pkl_files = sorted(model_folder.glob("*.pkl"))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found in {model_folder.resolve()}")

    default_n = getattr(runner, "DEFAULT_NT_NUMBER", 8)
    models: list[LoadedModel] = []
    for path in pkl_files:
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("not a bundled model dict")
            window_size = int(payload.get("window_size", default_n))
            kind = "GAT" if is_gat_payload(payload) else "classical"
            # The runner's loader validates the bundle and, for GAT bundles,
            # wraps the state_dict in its sklearn-like GATPredictor.
            bundle = runner.load_model_bundle(path, window_size)
        except Exception as exc:  # noqa: BLE001 — report and keep going
            print(f"[skip model] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        models.append(
            LoadedModel(
                name=path.stem,
                path=path,
                kind=kind,
                bundle=bundle,
                window_size=window_size,
                csv_path=result_folder / f"{path.stem}.csv",
                quality_path=result_folder / f"{path.stem}_quality.txt",
            )
        )
        print(f"Loaded {path.name} -> {kind}, window_size={window_size}")

    if not models:
        raise SystemExit("None of the .pkl files could be loaded.")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Structures
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


def read_atoms(path: Path) -> pd.DataFrame:
    """Like the runner's parse_structure_file, but also reads .gz files."""
    lower = path.name.lower()
    gzipped = lower.endswith(".gz")
    inner = lower[:-3] if gzipped else lower
    opener = gzip.open if gzipped else open
    with opener(path, "rt") as handle:
        if inner.endswith((".cif", ".mmcif")):
            atoms_df = parse_cif_atoms(handle)
        elif inner.endswith(".pdb"):
            atoms_df = parse_pdb_atoms(handle)
        else:
            raise ValueError(f"Unsupported structure format: {path.name}")
    if atoms_df.empty:
        raise ValueError("no atoms parsed")
    return atoms_df


def load_structure(runner: ModuleType, path: Path, requested_model: Optional[int]) -> Structure:
    atoms_df = read_atoms(path)
    atoms_df, _ = runner.select_structure_model(atoms_df, requested_model)
    return Structure(atoms_df)


# ─────────────────────────────────────────────────────────────────────────────
# Residue keys and ground truth
# ─────────────────────────────────────────────────────────────────────────────

def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def residue_key(chain_id: Any, residue_number: Any, insertion_code: Any = "") -> ResidueKey:
    icode = _clean(insertion_code)
    if icode in {"?", "."}:
        icode = ""
    return (_clean(chain_id), int(float(residue_number)), icode)


_RESIDUE_RE = re.compile(r"^(-?\d+)(.*)$")


def parse_residue_id(text: str) -> ResidueKey:
    """'A:12' / 'A:12B' / 'A:-3' -> ResidueKey (format written by the runner)."""
    chain, rest = text.strip().rsplit(":", 1)
    match = _RESIDUE_RE.match(rest.strip())
    if match is None:
        raise ValueError(f"cannot parse residue id {text!r}")
    return residue_key(chain, match.group(1), match.group(2))


def unit_to_key(unit: Any) -> ResidueKey:
    """Accepts the unit formats likely to appear in an answers file:
    {"chain_id": "A", "residue_number": 12, "insertion_code": ""},
    "A:12", BGSU unit ids "1ABC|1|A|G|12", or ["A", 12]."""
    if isinstance(unit, dict):
        chain = next((unit[k] for k in ("chain_id", "chain", "auth_asym_id") if k in unit), None)
        number = next((unit[k] for k in ("residue_number", "number", "auth_seq_id") if k in unit), None)
        if chain is None or number is None:
            raise ValueError(f"unit without chain / residue number: {unit}")
        icode = next(
            (unit[k] for k in ("insertion_code", "icode", "pdbx_PDB_ins_code") if k in unit), ""
        )
        return residue_key(chain, number, icode)
    if isinstance(unit, str):
        if "|" in unit:
            parts = unit.split("|")
            return residue_key(parts[2], parts[4], parts[7] if len(parts) > 7 else "")
        return parse_residue_id(unit)
    if isinstance(unit, (list, tuple)) and len(unit) in (2, 3):
        return residue_key(*unit)
    raise ValueError(f"unrecognised unit format: {unit!r}")


def load_answers(path: Path) -> dict[str, set[MotifKey]]:
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{path.name}: expected a JSON object mapping structure name -> motifs")

    answers: dict[str, set[MotifKey]] = {}
    for struct, motifs in data.items():
        keys: set[MotifKey] = set()
        for motif in motifs or []:
            units = motif.get("unit_ids", motif.get("residues")) if isinstance(motif, dict) else motif
            if not units:
                print(f"[answers] {struct}: motif without residues skipped", file=sys.stderr)
                continue
            keys.add(frozenset(unit_to_key(u) for u in units))
        answers[struct.lower()] = keys

    n_motifs = sum(len(v) for v in answers.values())
    print(f"Loaded {n_motifs} known motifs for {len(answers)} structures from {path.name}")
    return answers


def lookup_answers(name: str, answers: dict[str, set[MotifKey]]) -> Optional[set[MotifKey]]:
    """Exact name first ('1abc-assembly1'), then the bare PDB id ('1abc')."""
    lower = name.lower()
    if lower in answers:
        return answers[lower]
    pdb_id = lower.split("-")[0]
    if pdb_id in answers:
        return answers[pdb_id]
    return None


def match_windows(
    keys: list[MotifKey], known: set[MotifKey], mode: str, min_overlap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns truth[n_windows] and hits[n_motifs, n_windows]."""
    motifs = list(known)
    hits = np.zeros((len(motifs), len(keys)), dtype=bool)
    for m_idx, motif in enumerate(motifs):
        size = len(motif)
        for w_idx, window in enumerate(keys):
            if mode == "exact":
                hit = window == motif
            elif mode == "contains":
                hit = motif <= window
            elif mode == "centered":
                ordered = sorted(window, key=lambda k: (k[0], k[1], k[2]))
                extra = len(window) - size
                left = extra // 2
                hit = (
                    motif <= window
                    and extra >= 0
                    and frozenset(ordered[left:left + size]) == motif
                )
            else:
                hit = size > 0 and len(window & motif) / size >= min_overlap
            hits[m_idx, w_idx] = hit
    truth = hits.any(axis=0) if motifs else np.zeros(len(keys), dtype=bool)
    return truth, hits


# ─────────────────────────────────────────────────────────────────────────────
# Windows and scoring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowSet:
    meta: pd.DataFrame
    keys: list[MotifKey]
    features: pd.DataFrame
    skipped: int


def build_windows(
    runner: ModuleType, path: Path, structure: Structure, window_size: int
) -> Optional[WindowSet]:
    """Exactly the runner's own window generation (build_window_rows)."""
    residues_by_chain = runner.get_nucleotide_residues_by_chain(structure)
    if not residues_by_chain:
        return None
    rows, features, skipped = runner.build_window_rows(path, residues_by_chain, window_size)
    if features.empty:
        return None
    meta = pd.DataFrame(rows)
    keys = [
        frozenset(parse_residue_id(r) for r in ids.split(","))
        for ids in meta["residue_ids"]
    ]
    return WindowSet(meta, keys, features, skipped)


def score(
    runner: ModuleType, model: LoadedModel, windows: WindowSet, threshold: float
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    bundle = model.bundle
    feature_columns = bundle["feature_columns"]

    missing = [c for c in feature_columns if c not in windows.features.columns]
    if missing:
        raise ValueError(
            f"generated features lack {len(missing)} columns the model expects "
            f"(e.g. {', '.join(missing[:5])})"
        )
    feature_matrix = windows.features.reindex(columns=feature_columns)
    if feature_matrix.isna().any().any():
        raise ValueError("feature matrix contains NaN values")

    transformed = bundle["scaler"].transform(feature_matrix)
    classifier = bundle["classifier"]
    positive_label = bundle.get("positive_label", True)

    probabilities = runner.positive_class_probabilities(classifier, transformed, positive_label)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        predictions = probabilities >= threshold
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


def compute_metrics(model: LoadedModel) -> dict[str, Any]:
    s = model.stats
    tp = sum(x.tp for x in s)
    fp = sum(x.fp for x in s)
    tn = sum(x.tn for x in s)
    fn = sum(x.fn for x in s)
    known = sum(x.known_motifs for x in s)
    covered = sum(x.motifs_in_windows for x in s)
    detected = sum(x.motifs_detected for x in s)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "model": model.name,
        "type": model.kind,
        "classifier": model.bundle.get("classifier_name", "?"),
        "structures": len(s),
        "windows": tp + fp + tn + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else float("nan"),
        "known_motifs": known,
        "motifs_in_windows": covered,
        "motifs_detected": detected,
        "motif_recall": _ratio(detected, known),
        "failed_structures": len(model.failures),
    }


def write_quality_report(
    model: LoadedModel, m: dict[str, Any], args: argparse.Namespace, not_in_answers: list[str]
) -> None:
    if model.has_probability is False:
        rule = "classifier.predict() (no predict_proba available)"
    else:
        rule = f"positive if P(motif) >= {args.threshold}"
    match_rule = {
        "centered": "motif centred in the window",
        "exact": "window residue set == motif residue set",
        "contains": "window contains every motif residue",
        "overlap": f"window covers >= {args.min_overlap:.0%} of the motif residues",
    }[args.match]

    lines = [
        f"Model:              {model.name}",
        f"Model file:         {model.path}",
        f"Type:               {model.kind}",
        f"Classifier:         {m['classifier']}",
        f"Window size:        {model.window_size} consecutive nucleotides per chain",
        f"Answers file:       {args.answers}",
        f"Decision rule:      {rule}",
        f"Ground truth rule:  {match_rule}",
        "",
        f"Structures scored:  {m['structures']}",
        f"Windows scored:     {m['windows']}",
        "",
        "Confusion matrix (per window)",
        f"  True positives  (TP): {m['tp']}",
        f"  False positives (FP): {m['fp']}",
        f"  True negatives  (TN): {m['tn']}",
        f"  False negatives (FN): {m['fn']}",
        "",
        f"  Precision:    {m['precision']:.4f}",
        f"  Recall:       {m['recall']:.4f}",
        f"  Specificity:  {m['specificity']:.4f}",
        f"  F1:           {m['f1']:.4f}",
        f"  Accuracy:     {m['accuracy']:.4f}",
        f"  MCC:          {m['mcc']:.4f}",
        "",
        "Per motif",
        f"  Known motifs in scored structures:         {m['known_motifs']}",
        f"  Matched by at least one window:            {m['motifs_in_windows']}",
        f"  Hit by at least one positive window:       {m['motifs_detected']}",
        f"  Motif-level recall (detected / known):     {m['motif_recall']:.4f}",
        "  (A motif no window matches cannot be found by any model: it spans several",
        "   chains, has a different length than the window, or has residues without C1'.)",
    ]

    if model.stats:
        lines += [
            "",
            "Per-structure breakdown",
            pd.DataFrame([asdict(x) for x in model.stats]).to_string(index=False),
        ]

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    allow_cuda_pickles_on_cpu()
    runner = import_runner(args.runner)

    model_folder = args.model_folder if args.model_folder is not None else Path.cwd()
    result_folder: Path = args.result_folder
    result_folder.mkdir(parents=True, exist_ok=True)

    models = load_models(runner, model_folder, result_folder)
    structures = find_structures(args.structures)
    answers = load_answers(args.answers) if args.answers is not None else None
    not_in_answers: list[str] = []

    if answers is not None and args.match == "exact":
        sizes = {len(k) for keys in answers.values() for k in keys}
        windows = {m.window_size for m in models}
        if sizes and not sizes & windows:
            print(
                f"WARNING: motif sizes in the answers {sorted(sizes)} never equal the window "
                f"size {sorted(windows)}, so --match exact cannot find any. "
                "Consider --match contains or --match overlap.",
                file=sys.stderr,
            )

    print(f"\nTesting {len(models)} model(s) on {len(structures)} structure(s)\n")

    for s_idx, path in enumerate(structures, start=1):
        name = structure_name(path)
        print(f"[{s_idx}/{len(structures)}] {path.name}", flush=True)

        try:
            structure = load_structure(runner, path, args.structure_model)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not parse: {exc}", file=sys.stderr)
            for model in models:
                model.failures.append((name, f"parse error: {exc}"))
            continue

        known: set[MotifKey] = set()
        if answers is not None:
            found = lookup_answers(name, answers)
            if found is None:
                not_in_answers.append(name)
            else:
                known = found

        # Windows depend only on the window size, so build them once per
        # structure and reuse them for every model (classical and GAT alike).
        cache: dict[int, Any] = {}

        for model in models:
            if model.window_size not in cache:
                try:
                    cache[model.window_size] = build_windows(
                        runner, path, structure, model.window_size
                    )
                except Exception as exc:  # noqa: BLE001
                    cache[model.window_size] = exc
            windows = cache[model.window_size]

            if isinstance(windows, Exception):
                print(f"  {model.name}: window generation failed: {windows}", file=sys.stderr)
                model.failures.append((name, f"window generation: {windows}"))
                continue

            if windows is None:
                print(f"  {model.name}: no windows")
                if answers is not None:
                    model.stats.append(StructureStats(name, 0, 0, 0, 0, 0, len(known), 0, 0))
                continue

            try:
                predictions, probabilities = score(runner, model, windows, args.threshold)
            except Exception as exc:  # noqa: BLE001
                print(f"  {model.name}: scoring failed: {exc}", file=sys.stderr)
                model.failures.append((name, f"scoring: {exc}"))
                continue

            model.has_probability = probabilities is not None

            df = pd.DataFrame(
                {
                    "structure": name,
                    "window_index": windows.meta["window_index"].to_numpy(),
                    "chain_id": windows.meta["chain_id"].to_numpy(),
                    "sequence": windows.meta["sequence"].to_numpy(),
                    "residue_ids": windows.meta["residue_ids"].to_numpy(),
                    "prediction": predictions,
                }
            )
            if probabilities is not None:
                df["probability"] = probabilities

            n_pos = int(predictions.sum())
            msg = f"  {model.name}: {len(df)} windows, {n_pos} predicted positive"

            if answers is not None:
                truth, hits = match_windows(windows.keys, known, args.match, args.min_overlap)
                df["ground_truth"] = truth
                covered = int(hits.any(axis=1).sum()) if len(known) else 0
                detected = int((hits & predictions).any(axis=1).sum()) if len(known) else 0
                model.stats.append(
                    StructureStats(
                        structure=name,
                        windows=len(df),
                        tp=int((predictions & truth).sum()),
                        fp=int((predictions & ~truth).sum()),
                        tn=int((~predictions & ~truth).sum()),
                        fn=int((~predictions & truth).sum()),
                        known_motifs=len(known),
                        motifs_in_windows=covered,
                        motifs_detected=detected,
                    )
                )
                msg += f", {covered}/{len(known)} known motifs matched by a window, {detected} detected"
                if known and covered == 0:
                    msg += "  <-- check chain IDs / residue numbering / --match"

            print(msg)
            append_csv(model, df)

    # Finalise outputs
    print()
    summary_rows = []
    for model in models:
        if model.rows_written == 0:
            columns = ["structure", "window_index", "chain_id", "sequence",
                       "residue_ids", "prediction", "probability"]
            if answers is not None:
                columns.append("ground_truth")
            pd.DataFrame(columns=columns).to_csv(model.csv_path, index=False)

        line = f"{model.name} [{model.kind}]: {model.rows_written} rows -> {model.csv_path}"
        if answers is not None:
            m = compute_metrics(model)
            write_quality_report(model, m, args, not_in_answers)
            summary_rows.append(m)
            line += (
                f"  | TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']} "
                f"MCC={m['mcc']:.3f} -> {model.quality_path.name}"
            )
        if model.failures:
            line += f"  ({len(model.failures)} structure(s) failed)"
        print(line)

    if summary_rows:
        summary_path = result_folder / "summary.csv"
        pd.DataFrame(summary_rows).sort_values("mcc", ascending=False, na_position="last").to_csv(
            summary_path, index=False, float_format="%.4f"
        )
        print(f"\nModel comparison -> {summary_path}")


if __name__ == "__main__":
    main()
