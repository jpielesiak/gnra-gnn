#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the classical models from 10-remove-redundancy-and-train-models.ipynb
on a folder of structures, keeping track of what each model was trained on.

The notebook trains 30 models: 6 cluster-based splits ({approximate,exact} x
{hierarchical, affinity-propagation, facility-location}) x 5 classifiers.
Each split holds out a different set of GNRA loops (HL_xxxx_nnn ids), so a
structure that is "test data" for one model can be training data for another.
Pooling everything into one score, as _14_test_all_window_models.py does,
hides that. This script reconstructs every split exactly as the notebook does
and reports the metrics per group:

  unseen          PDB id never appears in geometric_features.csv
  heldout         every GNRA loop of this PDB was held out of this model's training
  mixed           some of its GNRA loops were held out, some were trained on
  train           all of its GNRA loops were in this model's training set
  negatives-only  PDB appears only as stem (negative) fragments in the CSV
  unclassified    the bundle has no split_mode/split_method, or no cluster JSON

"unseen" is the cleanest estimate of real-world performance; "heldout" is the
one the cluster split was designed for. If the answers JSON carries a motif id
(motif_id / loop_id / id / name), motif-level recall is also split into
held-out loops and trained loops.

Windows, features and scoring reuse the sliding-window runner (-runner), so the
numbers match what the runner would produce on a single structure.

Example:
    uv run test_models.py -structures gnn/cif_files/ -answers gnn/gnra_motifs_by_pdb.json -runner gnn/05-run-inference.py -model_folder ./ -clusters ./ -features geometric_features.csv -result_folder split_test/ --match centered

"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure

DEFAULT_RUNNER = "run_model_window.py"
DEFAULT_RESULT_FOLDER = Path("models_split_test")
STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".mmcif", ".mmcif.gz", ".pdb", ".pdb.gz")

CLUSTER_MODES = ("approximate", "exact")
CLUSTER_METHODS = ("hierarchical", "affinity-propagation", "facility-location")
# Number of positive test rows each split produced in the notebook (the "True"
# support in its classification reports). Used to check the reconstruction.
NOTEBOOK_HELDOUT_POSITIVES = {
    ("approximate", "hierarchical"): 81,
    ("approximate", "affinity-propagation"): 79,
    ("approximate", "facility-location"): 80,
    ("exact", "hierarchical"): 81,
    ("exact", "affinity-propagation"): 88,
    ("exact", "facility-location"): 83,
}
GROUP_ORDER = ("all", "unseen", "heldout", "mixed", "train", "negatives-only", "unclassified")
MOTIF_ID_KEYS = ("motif_id", "loop_id", "id", "name", "source_file")

ResidueKey = tuple[str, int, str]
MotifKey = frozenset


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-structures", "--structures", dest="structures", type=Path, required=True,
                   help="Folder with the .cif / .cif.gz / .pdb files to test on")
    p.add_argument("-answers", "--answers", dest="answers", type=Path, default=None,
                   help="JSON with the known motif locations (enables ground truth + metrics)")
    p.add_argument("-model_folder", "--model-folder", dest="model_folder", type=Path, default=None,
                   help="Folder with the .pkl bundles (default: current directory)")
    p.add_argument("-clusters", "--clusters", dest="clusters", type=Path, default=None,
                   help="Folder with {approximate,exact}-{method}.json (default: model folder)")
    p.add_argument("-features", "--features", dest="features", type=Path, default=None,
                   help="geometric_features.csv used for training (needed for split groups)")
    p.add_argument("-result_folder", "-resultfolder", "--result-folder", dest="result_folder",
                   type=Path, default=DEFAULT_RESULT_FOLDER,
                   help="Output folder (default: %(default)s)")
    p.add_argument("-runner", "--runner", dest="runner", type=Path, default=None,
                   help=f"Sliding-window inference script (default: {DEFAULT_RUNNER} next to this file)")
    p.add_argument("--match", choices=("exact", "contains", "centered", "overlap"), default="exact",
                   help="When a window counts as a true motif (default: %(default)s)")
    p.add_argument("--min-overlap", type=float, default=0.5,
                   help="Fraction of motif residues needed with --match overlap (default: %(default)s)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for a positive prediction (default: %(default)s)")
    p.add_argument("--structure-model", type=int, default=None,
                   help="1-based model number for multi-model files (default: first model)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def import_runner(path: Optional[Path]) -> ModuleType:
    if path is None:
        path = Path(__file__).resolve().parent / DEFAULT_RUNNER
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"Runner script not found: {path}\nPass it with  -runner path/to/script.py")

    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise SystemExit(
            f"Importing {path.name} failed: {exc}\n"
            "The runner imports torch / torch_geometric at the top. Install them, or wrap "
            "those imports in try/except ImportError in the runner (classical models do "
            "not need them)."
        )

    needed = ("get_nucleotide_residues_by_chain", "build_window_rows",
              "positive_class_probabilities", "calculate_window_features",
              "select_structure_model")
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise SystemExit(f"{path.name} is not the sliding-window runner (missing: {', '.join(missing)})")
    print(f"Using runner {path.name}")
    return module


_feature_name_cache: dict[int, set[str]] = {}


def runner_feature_names(runner: ModuleType, window_size: int) -> set[str]:
    """Column names the runner generates for a window of this size."""
    if window_size not in _feature_name_cache:
        rng = np.random.default_rng(0)
        coords = [tuple(float(v) for v in row) for row in rng.normal(scale=10, size=(window_size, 3))]
        _feature_name_cache[window_size] = set(runner.calculate_window_features(coords, window_size))
    return _feature_name_cache[window_size]


# ─────────────────────────────────────────────────────────────────────────────
# Training data and splits (mirrors the notebook)
# ─────────────────────────────────────────────────────────────────────────────

_PDB_RE = re.compile(r"^[0-9][a-z0-9]{3}$")


def pdb_from_source(source: str) -> Optional[str]:
    """'HL_1G1X_001' -> '1g1x', 'stems_9did_0009' -> '9did'."""
    for token in source.lower().split("_"):
        if _PDB_RE.match(token):
            return token
    return None


def pdb_from_structure(name: str) -> str:
    """'1abc-assembly1' -> '1abc'."""
    match = re.match(r"^([0-9][a-z0-9]{3})", name.lower())
    return match.group(1) if match else name.lower()


def clean_name(name: str) -> str:
    """Same as clean_name() in the notebook."""
    return name.split("/")[-1].split(".")[0]


@dataclass
class TrainingSet:
    positive_ids: set[str]                 # lowercase HL ids with gnra == True
    positives_by_pdb: dict[str, set[str]]  # pdb -> its positive ids
    all_pdbs: set[str]                     # every pdb present in the CSV
    n_positive: int


def load_training_set(path: Path) -> TrainingSet:
    df = pd.read_csv(path, usecols=["source_file", "gnra"])
    is_pos = df["gnra"].map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"})
    positive_ids: set[str] = set()
    positives_by_pdb: dict[str, set[str]] = defaultdict(set)
    all_pdbs: set[str] = set()
    for source, pos in zip(df["source_file"].astype(str), is_pos):
        pdb = pdb_from_source(source)
        if pdb is not None:
            all_pdbs.add(pdb)
        if pos:
            positive_ids.add(source.lower())
            if pdb is not None:
                positives_by_pdb[pdb].add(source.lower())
    print(f"Training CSV {path.name}: {len(df)} rows, {int(is_pos.sum())} positive, "
          f"{len(all_pdbs)} PDB ids")
    return TrainingSet(positive_ids, dict(positives_by_pdb), all_pdbs, int(is_pos.sum()))


def load_splits(folder: Path, training: TrainingSet) -> dict[tuple[str, str], set[str]]:
    """Held-out positive ids per (mode, method), built exactly like the notebook."""
    splits: dict[tuple[str, str], set[str]] = {}
    for mode in CLUSTER_MODES:
        for method in CLUSTER_METHODS:
            path = folder / f"{mode}-{method}.json"
            if not path.is_file():
                print(f"[clusters] {path} not found; models of this split will be 'unclassified'",
                      file=sys.stderr)
                continue
            with open(path) as handle:
                data = json.load(handle)
            clusters = [
                (clean_name(c["representative"]), [clean_name(m) for m in c["members"]])
                for c in data["clustering"]["clusters"]
            ]
            clusters.sort(key=lambda c: len(c[1]))  # stable sort, as in the notebook
            test_names: list[str] = []
            for representative, members in clusters:
                test_names.extend([representative] + members)
                if len(test_names) >= 0.25 * training.n_positive:
                    break
            heldout = {n.lower() for n in test_names}
            splits[(mode, method)] = heldout

            n_rows = len(heldout & training.positive_ids)
            expected = NOTEBOOK_HELDOUT_POSITIVES.get((mode, method))
            note = ""
            if expected is not None and n_rows != expected:
                note = f"  WARNING: notebook had {expected}; split reconstruction may differ"
            print(f"Split {mode}-{method}: {n_rows} held-out GNRA loops{note}")
    return splits


def structure_group(name: str, heldout: Optional[set[str]], training: Optional[TrainingSet]) -> str:
    if training is None:
        return "unclassified"
    pdb = pdb_from_structure(name)
    if pdb not in training.all_pdbs:
        return "unseen"
    positives = training.positives_by_pdb.get(pdb, set())
    if not positives:
        return "negatives-only"
    if heldout is None:
        return "unclassified"
    held = positives & heldout
    if held == positives:
        return "heldout"
    return "mixed" if held else "train"


def motif_status(motif_id: Optional[str], heldout: Optional[set[str]],
                 training: Optional[TrainingSet]) -> str:
    if motif_id is None or training is None or heldout is None:
        return "unknown"
    if motif_id in heldout:
        return "heldout"
    if motif_id in training.positive_ids:
        return "train"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"classifier", "classifier_name", "scaler", "feature_columns", "window_size"}


@dataclass
class StructureStats:
    structure: str
    group: str
    windows: int
    tp: int
    fp: int
    tn: int
    fn: int
    known_motifs: int
    motifs_in_windows: int
    motifs_detected: int
    heldout_motifs: int = 0
    heldout_detected: int = 0
    train_motifs: int = 0
    train_detected: int = 0


@dataclass
class LoadedModel:
    name: str
    path: Path
    bundle: dict[str, Any]
    window_size: int
    split_mode: Optional[str]
    split_method: Optional[str]
    heldout: Optional[set[str]]
    csv_path: Path
    quality_path: Path
    rows_written: int = 0
    has_probability: Optional[bool] = None
    stats: list[StructureStats] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


def validate_bundle(runner: ModuleType, payload: Any) -> int:
    if not isinstance(payload, dict):
        raise ValueError("not a bundled model dict")
    if "model_state_dict" in payload and "classifier" not in payload:
        raise ValueError("GAT bundle; use _14_test_all_window_models.py for GAT models")
    missing = sorted(REQUIRED_KEYS.difference(payload))
    if missing:
        raise ValueError(f"missing bundle keys: {', '.join(missing)}")

    window_size = int(payload["window_size"])
    columns = payload["feature_columns"]
    if not isinstance(columns, list) or not columns:
        raise ValueError("invalid feature_columns")
    for part in ("classifier", "scaler"):
        n = getattr(payload[part], "n_features_in_", None)
        if n is not None and int(n) != len(columns):
            raise ValueError(f"{part} expects {n} features, bundle lists {len(columns)}")

    absent = [c for c in columns if c not in runner_feature_names(runner, window_size)]
    if absent:
        raise ValueError(
            f"{len(absent)} feature columns are not generated by the runner "
            f"(e.g. {', '.join(absent[:5])})"
        )
    return window_size


def load_models(runner: ModuleType, model_folder: Path, result_folder: Path,
                splits: dict[tuple[str, str], set[str]]) -> list[LoadedModel]:
    pkl_files = sorted(model_folder.glob("*.pkl"))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found in {model_folder.resolve()}")

    models: list[LoadedModel] = []
    for path in pkl_files:
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            window_size = validate_bundle(runner, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip model] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        payload.setdefault("positive_label", True)
        mode, method = payload.get("split_mode"), payload.get("split_method")
        heldout = splits.get((mode, method)) if mode and method else None
        models.append(LoadedModel(
            name=path.stem, path=path, bundle=payload, window_size=window_size,
            split_mode=mode, split_method=method, heldout=heldout,
            csv_path=result_folder / f"{path.stem}.csv",
            quality_path=result_folder / f"{path.stem}_quality.txt",
        ))
        split_txt = f"{mode}-{method}" if mode else "no split info"
        print(f"Loaded {path.name}: {payload['classifier_name']}, {split_txt}, window_size={window_size}")

    if not models:
        raise SystemExit("None of the .pkl files could be loaded.")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Structures and ground truth
# ─────────────────────────────────────────────────────────────────────────────

def find_structures(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    files = [p for p in sorted(folder.iterdir())
             if p.is_file() and p.name.lower().endswith(STRUCTURE_SUFFIXES)]
    if not files:
        raise SystemExit(f"No structure files found in {folder.resolve()}")
    return files


def structure_name(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    for suffix in (".cif", ".mmcif", ".pdb"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_structure(runner: ModuleType, path: Path, requested_model: Optional[int]) -> Structure:
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
    atoms_df, _ = runner.select_structure_model(atoms_df, requested_model)
    return Structure(atoms_df)


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
    chain, rest = text.strip().rsplit(":", 1)
    match = _RESIDUE_RE.match(rest.strip())
    if match is None:
        raise ValueError(f"cannot parse residue id {text!r}")
    return residue_key(chain, match.group(1), match.group(2))


def unit_to_key(unit: Any) -> ResidueKey:
    if isinstance(unit, dict):
        chain = next((unit[k] for k in ("chain_id", "chain", "auth_asym_id") if k in unit), None)
        number = next((unit[k] for k in ("residue_number", "number", "auth_seq_id") if k in unit), None)
        if chain is None or number is None:
            raise ValueError(f"unit without chain / residue number: {unit}")
        icode = next((unit[k] for k in ("insertion_code", "icode", "pdbx_PDB_ins_code") if k in unit), "")
        return residue_key(chain, number, icode)
    if isinstance(unit, str):
        if "|" in unit:
            parts = unit.split("|")
            return residue_key(parts[2], parts[4], parts[7] if len(parts) > 7 else "")
        return parse_residue_id(unit)
    if isinstance(unit, (list, tuple)) and len(unit) in (2, 3):
        return residue_key(*unit)
    raise ValueError(f"unrecognised unit format: {unit!r}")


def load_answers(path: Path) -> dict[str, dict[MotifKey, Optional[str]]]:
    """structure name -> {motif residue set: motif id or None}."""
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{path.name}: expected a JSON object mapping structure name -> motifs")

    answers: dict[str, dict[MotifKey, Optional[str]]] = {}
    with_id = 0
    for struct, motifs in data.items():
        entry: dict[MotifKey, Optional[str]] = {}
        for motif in motifs or []:
            motif_id = None
            if isinstance(motif, dict):
                units = motif.get("unit_ids", motif.get("residues"))
                raw_id = next((motif[k] for k in MOTIF_ID_KEYS if motif.get(k)), None)
                motif_id = clean_name(str(raw_id)).lower() if raw_id is not None else None
            else:
                units = motif
            if not units:
                print(f"[answers] {struct}: motif without residues skipped", file=sys.stderr)
                continue
            entry[frozenset(unit_to_key(u) for u in units)] = motif_id
            with_id += motif_id is not None
        answers[struct.lower()] = entry

    n = sum(len(v) for v in answers.values())
    print(f"Loaded {n} known motifs for {len(answers)} structures from {path.name} "
          f"({with_id} with a motif id)")
    return answers


def lookup_answers(name: str, answers: dict) -> Optional[dict[MotifKey, Optional[str]]]:
    lower = name.lower()
    if lower in answers:
        return answers[lower]
    pdb_id = lower.split("-")[0]
    return answers.get(pdb_id)


def match_windows(keys: list[MotifKey], motifs: list[MotifKey], mode: str,
                  min_overlap: float) -> tuple[np.ndarray, np.ndarray]:
    """truth[n_windows], hits[n_motifs, n_windows]."""
    hits = np.zeros((len(motifs), len(keys)), dtype=bool)
    for m_idx, motif in enumerate(motifs):
        size = len(motif)
        for w_idx, window in enumerate(keys):
            if mode == "exact":
                hit = window == motif
            elif mode == "contains":
                hit = motif <= window
            elif mode == "centered":
                extra = len(window) - size
                if extra < 0 or not motif <= window:
                    hit = False
                else:
                    ordered = sorted(window)
                    left = extra // 2
                    hit = frozenset(ordered[left:left + size]) == motif
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


def build_windows(runner: ModuleType, path: Path, structure: Structure,
                  window_size: int) -> Optional[WindowSet]:
    residues_by_chain = runner.get_nucleotide_residues_by_chain(structure)
    if not residues_by_chain:
        return None
    rows, features, _ = runner.build_window_rows(path, residues_by_chain, window_size)
    if features.empty:
        return None
    meta = pd.DataFrame(rows)
    keys = [frozenset(parse_residue_id(r) for r in ids.split(",")) for ids in meta["residue_ids"]]
    return WindowSet(meta, keys, features)


def score(runner: ModuleType, model: LoadedModel, windows: WindowSet,
          threshold: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
    bundle = model.bundle
    feature_matrix = windows.features.reindex(columns=bundle["feature_columns"])
    if feature_matrix.isna().any().any():
        raise ValueError("feature matrix contains NaN values")
    transformed = bundle["scaler"].transform(feature_matrix)
    classifier = bundle["classifier"]
    positive_label = bundle["positive_label"]

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
# Metrics and reports
# ─────────────────────────────────────────────────────────────────────────────

def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def compute_metrics(model: LoadedModel, group: str) -> Optional[dict[str, Any]]:
    stats = model.stats if group == "all" else [s for s in model.stats if s.group == group]
    if not stats:
        return None
    tot = {k: sum(getattr(s, k) for s in stats) for k in (
        "tp", "fp", "tn", "fn", "known_motifs", "motifs_in_windows", "motifs_detected",
        "heldout_motifs", "heldout_detected", "train_motifs", "train_detected")}
    tp, fp, tn, fn = tot["tp"], tot["fp"], tot["tn"], tot["fn"]
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "model": model.name,
        "classifier": model.bundle.get("classifier_name", "?"),
        "split_mode": model.split_mode,
        "split_method": model.split_method,
        "group": group,
        "structures": len(stats),
        "windows": tp + fp + tn + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else float("nan"),
        "known_motifs": tot["known_motifs"],
        "motifs_in_windows": tot["motifs_in_windows"],
        "motifs_detected": tot["motifs_detected"],
        "motif_recall": _ratio(tot["motifs_detected"], tot["known_motifs"]),
        "heldout_motifs": tot["heldout_motifs"],
        "heldout_motif_recall": _ratio(tot["heldout_detected"], tot["heldout_motifs"]),
        "train_motifs": tot["train_motifs"],
        "train_motif_recall": _ratio(tot["train_detected"], tot["train_motifs"]),
    }


def metric_block(m: dict[str, Any]) -> list[str]:
    return [
        f"[{m['group']}]  {m['structures']} structures, {m['windows']} windows",
        f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}",
        f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
        f"specificity={m['specificity']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}",
        f"  motifs: {m['known_motifs']} known, {m['motifs_in_windows']} matched by a window, "
        f"{m['motifs_detected']} detected (recall {m['motif_recall']:.4f})",
        f"  held-out loops: {m['heldout_motifs']} (recall {m['heldout_motif_recall']:.4f})   "
        f"trained loops: {m['train_motifs']} (recall {m['train_motif_recall']:.4f})",
    ]


def write_quality_report(model: LoadedModel, blocks: list[dict[str, Any]],
                         args: argparse.Namespace, not_in_answers: list[str]) -> None:
    rule = ("classifier.predict() (no predict_proba)" if model.has_probability is False
            else f"positive if P(motif) >= {args.threshold}")
    lines = [
        f"Model:          {model.name}",
        f"Model file:     {model.path}",
        f"Classifier:     {model.bundle.get('classifier_name', '?')}",
        f"Split:          {model.split_mode}-{model.split_method}"
        + ("" if model.heldout is not None else "  (cluster JSON not loaded)"),
        f"Window size:    {model.window_size}",
        f"Answers file:   {args.answers}",
        f"Decision rule:  {rule}",
        f"Match rule:     --match {args.match}",
        "",
        "Groups: unseen = PDB not in training CSV; heldout = all its GNRA loops held out of",
        "this model; mixed = some held out; train = all trained on; negatives-only = PDB only",
        "in the CSV as stem fragments (about 75% of those were training negatives).",
        "",
    ]
    for m in blocks:
        lines += metric_block(m) + [""]

    if model.stats:
        lines += ["Per-structure breakdown",
                  pd.DataFrame([asdict(s) for s in model.stats]).to_string(index=False)]
    if not_in_answers:
        lines += ["", f"Structures absent from the answers file ({len(not_in_answers)}), "
                      "treated as containing no motifs:"] + [f"  {n}" for n in not_in_answers]
    if model.failures:
        lines += ["", f"Structures that failed ({len(model.failures)}):"]
        lines += [f"  {n}: {e}" for n, e in model.failures]
    model.quality_path.write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    runner = import_runner(args.runner)

    model_folder = args.model_folder if args.model_folder is not None else Path.cwd()
    cluster_folder = args.clusters if args.clusters is not None else model_folder
    result_folder: Path = args.result_folder
    result_folder.mkdir(parents=True, exist_ok=True)

    training: Optional[TrainingSet] = None
    splits: dict[tuple[str, str], set[str]] = {}
    if args.features is not None:
        training = load_training_set(args.features)
        splits = load_splits(cluster_folder, training)
    else:
        print("WARNING: no -features given; every structure is 'unclassified' and the "
              "metrics mix training and test data.", file=sys.stderr)

    models = load_models(runner, model_folder, result_folder, splits)
    structures = find_structures(args.structures)
    answers = load_answers(args.answers) if args.answers is not None else None
    not_in_answers: list[str] = []

    if answers is not None and args.match == "exact":
        sizes = {len(k) for entry in answers.values() for k in entry}
        if sizes and not sizes & {m.window_size for m in models}:
            print(f"WARNING: motif sizes {sorted(sizes)} never equal the window size, so "
                  "--match exact finds nothing. Try --match centered or contains.", file=sys.stderr)

    # How the test structures fall into groups for each split
    if training is not None:
        print("\nTest structures per group:")
        for key in sorted({(m.split_mode, m.split_method) for m in models}, key=str):
            heldout = splits.get(key) if all(key) else None
            counts = defaultdict(int)
            for path in structures:
                counts[structure_group(structure_name(path), heldout, training)] += 1
            print(f"  {key[0]}-{key[1]}: " + ", ".join(f"{g}={counts[g]}" for g in GROUP_ORDER if counts[g]))

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

        known: dict[MotifKey, Optional[str]] = {}
        if answers is not None:
            found = lookup_answers(name, answers)
            if found is None:
                not_in_answers.append(name)
            else:
                known = found
        motif_keys = list(known)
        motif_ids = [known[k] for k in motif_keys]

        cache: dict[int, Any] = {}
        for model in models:
            group = structure_group(name, model.heldout, training)
            status = [motif_status(mid, model.heldout, training) for mid in motif_ids]

            if model.window_size not in cache:
                try:
                    cache[model.window_size] = build_windows(runner, path, structure, model.window_size)
                except Exception as exc:  # noqa: BLE001
                    cache[model.window_size] = exc
            windows = cache[model.window_size]

            if isinstance(windows, Exception):
                model.failures.append((name, f"window generation: {windows}"))
                continue
            if windows is None:
                if answers is not None:
                    model.stats.append(StructureStats(
                        name, group, 0, 0, 0, 0, 0, len(known), 0, 0,
                        heldout_motifs=status.count("heldout"), train_motifs=status.count("train")))
                continue

            try:
                predictions, probabilities = score(runner, model, windows, args.threshold)
            except Exception as exc:  # noqa: BLE001
                print(f"  {model.name}: scoring failed: {exc}", file=sys.stderr)
                model.failures.append((name, f"scoring: {exc}"))
                continue
            model.has_probability = probabilities is not None

            df = pd.DataFrame({
                "structure": name,
                "group": group,
                "window_index": windows.meta["window_index"].to_numpy(),
                "chain_id": windows.meta["chain_id"].to_numpy(),
                "sequence": windows.meta["sequence"].to_numpy(),
                "residue_ids": windows.meta["residue_ids"].to_numpy(),
                "prediction": predictions,
            })
            if probabilities is not None:
                df["probability"] = probabilities

            if answers is not None:
                truth, hits = match_windows(windows.keys, motif_keys, args.match, args.min_overlap)
                df["ground_truth"] = truth
                df["motif_ids"] = [
                    ",".join(motif_ids[m] or "?" for m in np.flatnonzero(hits[:, w]))
                    for w in range(len(windows.keys))
                ]
                detected_mask = (hits & predictions).any(axis=1) if motif_keys else np.zeros(0, bool)
                model.stats.append(StructureStats(
                    structure=name,
                    group=group,
                    windows=len(df),
                    tp=int((predictions & truth).sum()),
                    fp=int((predictions & ~truth).sum()),
                    tn=int((~predictions & ~truth).sum()),
                    fn=int((~predictions & truth).sum()),
                    known_motifs=len(known),
                    motifs_in_windows=int(hits.any(axis=1).sum()) if motif_keys else 0,
                    motifs_detected=int(detected_mask.sum()),
                    heldout_motifs=status.count("heldout"),
                    heldout_detected=sum(d for d, s in zip(detected_mask, status) if s == "heldout"),
                    train_motifs=status.count("train"),
                    train_detected=sum(d for d, s in zip(detected_mask, status) if s == "train"),
                ))
            append_csv(model, df)

        n_windows = next((len(w.keys) for w in cache.values() if isinstance(w, WindowSet)), 0)
        print(f"  {n_windows} windows, {len(known)} known motifs")

    # Outputs
    print()
    summary_rows = []
    for model in models:
        if model.rows_written == 0:
            pd.DataFrame(columns=["structure", "group", "window_index", "chain_id", "sequence",
                                  "residue_ids", "prediction", "probability"]).to_csv(
                model.csv_path, index=False)
        if answers is None:
            continue
        blocks = [m for g in GROUP_ORDER if (m := compute_metrics(model, g)) is not None]
        write_quality_report(model, blocks, args, not_in_answers)
        summary_rows.extend(blocks)
        by_group = {m["group"]: m for m in blocks}
        parts = [f"MCC[{g}]={by_group[g]['mcc']:.3f}" for g in ("all", "unseen", "heldout") if g in by_group]
        if "all" in by_group:
            parts.append(f"heldout-loop recall={by_group['all']['heldout_motif_recall']:.3f}")
        print(f"{model.name}: " + "  ".join(parts))

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary["group"] = pd.Categorical(summary["group"], categories=GROUP_ORDER, ordered=True)
        summary = summary.sort_values(["group", "mcc"], ascending=[True, False], na_position="last")
        summary_path = result_folder / "summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.4f")
        print(f"\nModel comparison (one row per model and group) -> {summary_path}")


if __name__ == "__main__":
    main()