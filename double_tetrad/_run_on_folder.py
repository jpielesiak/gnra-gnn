#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python batch_evaluate.py \
  --model models/gat_model.pkl \
  --structures-dir cif_files/ \
  --dates-csv dates.csv --cutoff-date 2020-01-01 --seed 42


motif_cif_files
  
    python _12_batch_evaluate.py \
  --model approximate-affinity-propagation-gat_v10.pkl \
  --structures-dir negative_cif_files/ \
  --dates-csv rna_pdb_release_dates.csv --cutoff-date 2024-10-20 --seed 42

  python _12_batch_evaluate.py \
  --model approximate-facility-location-gat_v10.pkl \
  --structures-dir motif_cif_files/ \
  --dates-csv rna_pdb_release_dates.csv --cutoff-date 2024-10-20 --seed 42

  python _12_batch_evaluate.py\
  --model approximate-hierarchical-gat_v10.pkl \
  --structures-dir motif_cif_files/ \
  --dates-csv rna_pdb_release_dates.csv --cutoff-date 2024-10-20 --seed 42
Batch evaluation of a trained double-tetrad model on whole structures.
 
For every clustering split, each clustered structure is scanned with the same
KD-tree candidate enumeration as run_model.py, and the results are divided
into structures the model saw during training ("seen") and those it did not
("unseen").
 
Output layout
-------------
    test_results_<model>/
        summary.csv                    pooled metrics per split and group
        errors.csv                     only written if something was skipped
        <mode>-<method>/
            results-seen.csv
            results-unseen.csv
            results-not-assessed.csv   structures excluded from scoring
            detailed/<structure>.txt   one line per candidate 8-tuple
 
Seen / unseen
-------------
Replays dedup_clusters_for_split(n_keep=1) from the training notebook:
  * a cluster with members in the pre-cutoff pool and none in the post-cutoff
    pool kept one member for training (fit or val)  -> every member is SEEN
  * cross-pool clusters (all train-side members dropped), post-only clusters,
    and clusters with no dated members                 -> every member is UNSEEN
The column in_training_set marks the one structure per seen cluster that was
actually trained on; the other seen rows are its removed near-duplicates.
 
Ground truth
------------
  --ground-truth CSV  columns source_file,residue_ids with one row per motif,
                      residue_ids formatted like run_model.py ("A:2,A:3,...").
                      A row with empty residue_ids marks a motif-free file.
  otherwise           positive rows of geometric_features.csv, matched to the
                      candidates by their sorted pairwise C1' distances, which
                      does not depend on residue order within the tuple.
 
Counting (per structure)
------------------------
  num_true_pos   real motifs whose candidate was predicted positive
  num_false_neg  real motifs not detected (includes motifs never enumerated)
  num_false_pos  non-motif candidates predicted positive
  num_true_neg   non-motif candidates predicted negative
so num_true_pos + num_false_neg == num_motifs_real.
 
Not assessed
------------
If any real motif of a structure is not among the generated candidates, the
model never got to score it, so the structure is left out of results-seen,
results-unseen and all metrics, and is listed in results-not-assessed.csv
instead, with the reason:
  too_far_apart       the motif's largest C1'-C1' distance exceeds --radius
                      (or --max-distance), so enumeration pruned it
  not_matched         the motif is within the limits but no candidate matched
                      it; this points to a ground-truth or coordinate problem,
                      not to distance, and is worth investigating
  not_enumerated      motif spread unknown (--ground-truth mode)
"""

import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Optional
import re
import numpy as np
import pandas as pd
from rnapolis.tertiary_v2 import Structure

# The inference script, saved next to this file as run_model.py.
# Change the module name here if you saved it under a different name.
from _11_run_inference import (
    build_candidate_rows,
    get_guanine_residues,
    load_model_bundle,
    parse_structure_file,
    positive_class_probabilities,
    select_structure_model,
)

STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".pdb", ".pdb.gz")


RESULT_COLUMNS = [
    "cif_file_name",
    "num_motifs_real",
    "num_true_pos",
    "num_false_pos",
    "num_true_neg",
    "num_false_neg",
    "in_training_set",
]
 
NOT_ASSESSED_COLUMNS = [
    "cif_file_name",
    "group",
    "num_motifs_real",
    "num_not_enumerated",
    "motif_max_distance",
    "num_candidates",
    "reason",
    "in_training_set",
]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Clusters (identical to the notebook)
# ─────────────────────────────────────────────────────────────────────────────
 
class Cluster:
    def __init__(self, representative, members):
        self.representative = representative
        self.members = members
 
 
def clean_name(name: str) -> str:
    return name.split("/")[-1].split(".")[0]
 
 
def load_clusters(array) -> list:
    return [
        Cluster(clean_name(obj["representative"]), list(map(clean_name, obj["members"])))
        for obj in array
    ]
 
 
def load_split_clusters(clusters_dir: Path, mode: str, method: str) -> list:
    with open(clusters_dir / f"{mode}-{method}.json") as f:
        data = json.load(f)
    return load_clusters(data["clustering"]["clusters"])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Temporal split (same rule as filter_by_date in the notebook)
# ─────────────────────────────────────────────────────────────────────────────
 
def _pdbid(src: str) -> str:
    parts = src.split("_")
    pid = parts[2] if len(parts) == 4 else (parts[1] if len(parts) >= 2 else src)
    return pid.split("-")[0]
 
 
def split_ids_by_date(ids, dates_csv: Path, cutoff_date: str) -> tuple[set, set]:
    cutoff = pd.to_datetime(cutoff_date, utc=True)
    release = {}
    with open(dates_csv) as f:
        for row in csv.DictReader(f):
            release[row["pdbid"]] = pd.to_datetime(row["release_date"], utc=True)
 
    pre, post = set(), set()
    for src in ids:
        pid = _pdbid(src)
        rd = release.get(pid) or release.get(pid.upper()) or release.get(pid.lower())
        if rd is None:
            continue
        (pre if rd < cutoff else post).add(src)
    return pre, post
 
 
def classify_members(clusters, pre_ids: set, post_ids: set, seed: int) -> dict:
    """Replay dedup_clusters_for_split(n_keep=1), consuming the RNG in exactly
    the same order so the kept training member is reproduced.
    Returns {member: (group, in_training_set)}."""
    rng = np.random.default_rng(seed)
    status = {}
    for cl in clusters:
        members = list(cl.members)
        in_pre = [m for m in members if m in pre_ids]
        in_post = [m for m in members if m in post_ids]
        trained = set()
        if in_pre and in_post:
            rng.shuffle(in_post)
            group = "unseen"
        elif in_pre:
            rng.shuffle(in_pre)
            trained = {in_pre[0]}
            group = "seen"
        elif in_post:
            rng.shuffle(in_post)
            group = "unseen"
        else:
            group = "unseen"
        for m in members:
            status[m] = (group, m in trained)
    return status
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────
 
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source_file" not in df.columns:
        df = pd.read_csv(path, index_col=0)
        df.index.name = "source_file"
        df = df.reset_index()
    df["source_file"] = df["source_file"].astype(str)
    labels = df["double_tetrad"]
    if labels.dtype == object:
        labels = labels.astype(str).str.strip().str.lower() == "true"
    df["double_tetrad"] = labels.astype(bool)
    return df
 
 
def distance_columns(n_nodes: int) -> list[str]:
    return [f"d{i}{j}" for i, j in combinations(range(n_nodes), 2)]
 
 
class GroundTruth:
    def __init__(self, features: pd.DataFrame, gt_csv: Optional[Path],
                 n_nodes: int, tol: float):
        self.tol = tol
        self.dist_cols = distance_columns(n_nodes)
        self.by_residues: Optional[dict] = None
        self.by_fingerprint: dict = {}
 
        if gt_csv is not None:
            gt = pd.read_csv(gt_csv, dtype=str, keep_default_na=False)
            self.by_residues = {}
            for src, res in zip(gt["source_file"], gt["residue_ids"]):
                motifs = self.by_residues.setdefault(clean_name(src), [])
                if res.strip():
                    motifs.append(frozenset(r.strip() for r in res.split(",")))
        else:
            for src in features["source_file"].unique():
                self.by_fingerprint[assembly_name(src)] = []
            pos = features[features["double_tetrad"]]
            for src, row in zip(pos["source_file"], pos[self.dist_cols].to_numpy(float)):
                self.by_fingerprint[assembly_name(src)].append(np.sort(row))
 
    def motif_spreads(self, member: str) -> Optional[list[float]]:
        """Largest pairwise C1' distance of each real motif (fingerprint mode
        only; None when ground truth comes from residue IDs)."""
        if self.by_residues is not None:
            return None
        return [float(m[-1]) for m in self.by_fingerprint.get(member, [])]
 
    def match(self, member: str, scored: pd.DataFrame) -> Optional[list]:
        """One boolean array per real motif, marking the candidates that are
        that motif. None if this structure has no ground truth at all."""
        if self.by_residues is not None:
            motifs = self.by_residues.get(member)
            if motifs is None:
                return None
            keys = [frozenset(r.split(",")) for r in scored["residue_ids"]]
            return [np.array([k == m for k in keys], dtype=bool) for m in motifs]
 
        motifs = self.by_fingerprint.get(member)
        if motifs is None:
            return None
        cand = np.sort(scored[self.dist_cols].to_numpy(float), axis=1)
        return [np.max(np.abs(cand - m), axis=1) <= self.tol for m in motifs]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Inference on one structure
# ─────────────────────────────────────────────────────────────────────────────
 
def score_structure(path: Path, bundle: dict, radius: float,
                    max_distance: Optional[float],
                    threshold: Optional[float]) -> pd.DataFrame:
    dist_cols = distance_columns(bundle["window_size"])
    empty = pd.DataFrame(columns=["residue_ids", "prediction", "probability", *dist_cols])
 
    atoms_df = parse_structure_file(path)
    atoms_df, _ = select_structure_model(atoms_df, None)
    guanines = get_guanine_residues(Structure(atoms_df))
    if not guanines:
        return empty
 
    rows, feats, _ = build_candidate_rows(
        path, guanines, bundle["window_size"], radius, max_distance
    )
    if feats.empty:
        return empty
 
    X = feats.reindex(columns=bundle["feature_columns"])
    if X.isna().any().any():
        raise ValueError("NaN features after reindexing (bundle/feature mismatch)")
 
    Xt = bundle["scaler"].transform(X)
    clf = bundle["classifier"]
    pos_label = bundle.get("positive_label", True)
    probs = positive_class_probabilities(clf, Xt, pos_label)
 
    if threshold is not None and probs is not None:
        preds = probs >= threshold
    else:
        preds = np.asarray(clf.predict(Xt)) == pos_label
 
    out = pd.DataFrame({
        "residue_ids": [r["residue_ids"] for r in rows],
        "prediction": np.asarray(preds, dtype=bool),
        "probability": probs if probs is not None else np.nan,
    })
    return pd.concat([out, feats[dist_cols].reset_index(drop=True)], axis=1)
 
def index_structures(root):
    index = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name.lower().endswith(STRUCTURE_SUFFIXES):
            index.setdefault(clean_name(str(p)), p)
    return index
  
# def index_structures(root: Path) -> dict[str, Path]:
#     index = {}
#     for p in sorted(root.rglob("*")):
#         if p.is_file() and p.suffix.lower() in STRUCTURE_SUFFIXES:
#             index.setdefault(clean_name(str(p)), p)
#     return index
def assembly_name(member: str) -> str:
    # 'DT_139d-assembly1_000' -> '139d-assembly1'
    return re.sub(r"_\d+$", "", re.sub(r"^[A-Za-z]+_", "", member))

def collapse_to_assemblies(status: dict) -> dict:
    """{member: (group, trained)} -> {assembly: (group, trained)}.
    An assembly counts as 'seen' if any of its members was seen, which is
    the conservative choice against leakage."""
    out = {}
    for member, (group, trained) in status.items():
        a = assembly_name(member)
        g, t = out.get(a, (group, trained))
        out[a] = ("seen" if "seen" in (g, group) else "unseen", t or trained)
    return out 
 
def write_detailed(path: Path, scored: pd.DataFrame, is_motif: np.ndarray) -> None:
    pd.DataFrame({
        "nt_group": scored["residue_ids"],
        "is_motif": is_motif,
        "model_answer": scored["prediction"].to_numpy(bool),
        "probability": scored["probability"],
    }).to_csv(path, sep="\t", index=False, float_format="%.6f")
 
 
def pooled_metrics(rows: list[dict]) -> dict:
    tp = sum(r["num_true_pos"] for r in rows)
    fp = sum(r["num_false_pos"] for r in rows)
    tn = sum(r["num_true_neg"] for r in rows)
    fn = sum(r["num_false_neg"] for r in rows)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "structures": len(rows),
        "motifs_real": tp + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "mcc": (tp * tn - fp * fn) / denom if denom else np.nan,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True,
                   help="Model bundle used for every split, or a template with "
                        "{mode} and {method} to use each split's own model, "
                        "e.g. 'models/gat_{mode}-{method}.pkl'")
    p.add_argument("--structures-dir", type=Path, required=True,
                   help="Directory (searched recursively) with the .cif/.pdb files")
    p.add_argument("--dates-csv", type=Path, required=True)
    p.add_argument("--cutoff-date", required=True)
    p.add_argument("--seed", type=int, required=True,
                   help="SEED used in the training notebook (for in_training_set)")
    p.add_argument("--features-csv", type=Path, default=Path("geometric_features.csv"))
    p.add_argument("--clusters-dir", type=Path, default=Path("."))
    p.add_argument("--modes", nargs="+", default=["approximate"])
    p.add_argument("--methods", nargs="+",
                   default=["hierarchical", "affinity-propagation", "facility-location"])
    p.add_argument("--nt-number", type=int, default=8)
    p.add_argument("--radius", type=float, default=25.0)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Probability threshold; default uses classifier.predict()")
    p.add_argument("--ground-truth", type=Path, default=None)
    p.add_argument("--match-tol", type=float, default=0.01,
                   help="Max distance difference (Å) for fingerprint matching")
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--run-name", default=None,
                   help="Name used in test_results_<name> (default: model file stem)")
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
 
    run_name = args.run_name or Path(
        args.model.replace("{mode}", "").replace("{method}", "")
    ).stem.strip("-_")
    root = args.output_root / f"test_results_{run_name}"
    root.mkdir(parents=True, exist_ok=True)
 
    features = load_features(args.features_csv)
    pre_ids, post_ids = split_ids_by_date(
        features["source_file"].unique(), args.dates_csv, args.cutoff_date
    )
    structure_index = index_structures(args.structures_dir)
    print(f"Indexed {len(structure_index)} structure files; "
          f"{len(pre_ids)} pre-cutoff / {len(post_ids)} post-cutoff ids")
 
    bundles: dict = {}
    cache: dict = {}          # (model_path, member) -> (scored, matches) or error str
    ground_truth: Optional[GroundTruth] = None
    errors: list[dict] = []
    summary: list[dict] = []
 
    for mode in args.modes:
        for method in args.methods:
            split = f"{mode}-{method}"
            model_path = Path(args.model.format(mode=mode, method=method))
            if model_path not in bundles:
                bundles[model_path] = load_model_bundle(model_path, args.nt_number)
            bundle = bundles[model_path]
            if ground_truth is None:
                ground_truth = GroundTruth(features, args.ground_truth,
                                           bundle["window_size"], args.match_tol)
 
            status = classify_members(
                load_split_clusters(args.clusters_dir, mode, method),
                pre_ids, post_ids, args.seed,
            )
            status = collapse_to_assemblies(status)
            split_dir = root / split
            detailed_dir = split_dir / "detailed"
            detailed_dir.mkdir(parents=True, exist_ok=True)
            results = {"seen": [], "unseen": []}
            not_assessed = []
            distance_limit = min(
                args.radius,
                args.max_distance if args.max_distance is not None else np.inf,
            )
 
            print(f"\n=== {split}  ({len(status)} structures, model {model_path.name}) ===")
            for i, (member, (group, trained)) in enumerate(sorted(status.items()), 1):
                path = structure_index.get(member)
                if path is None:
                    errors.append({"split": split, "structure": member,
                                   "error": "structure file not found"})
                    continue
 
                key = (model_path, member)
                if key not in cache:
                    t0 = time.perf_counter()
                    try:
                        scored = score_structure(path, bundle, args.radius,
                                                 args.max_distance, args.threshold)
                        matches = ground_truth.match(member, scored)
                        if matches is None:
                            cache[key] = "no ground truth for this structure"
                        else:
                            cache[key] = (scored[["residue_ids", "prediction",
                                                  "probability"]], matches)
                    except Exception as exc:  # keep going on bad files
                        cache[key] = f"{type(exc).__name__}: {exc}"
                    dt = time.perf_counter() - t0
                    n_cand = len(cache[key][0]) if isinstance(cache[key], tuple) else 0
                    print(f"  [{i}/{len(status)}] {member} ({group}) "
                          f"candidates={n_cand} {dt:.1f}s")
 
                entry = cache[key]
                if isinstance(entry, str):
                    errors.append({"split": split, "structure": member, "error": entry})
                    continue
 
                scored, matches = entry
                pred = scored["prediction"].to_numpy(bool)
                is_motif = (np.logical_or.reduce(matches) if matches
                            else np.zeros(len(scored), dtype=bool))
                tp = sum(bool((m & pred).any()) for m in matches)
                missing = [k for k, m in enumerate(matches) if not m.any()]
                write_detailed(detailed_dir / f"{member}.txt", scored, is_motif)
 
                if missing:
                    spreads = ground_truth.motif_spreads(member)
                    if spreads is None:
                        reason, max_spread = "not_enumerated", np.nan
                    else:
                        max_spread = max(spreads[k] for k in missing)
                        reason = ("too_far_apart" if max_spread > distance_limit
                                  else "not_matched")
                    not_assessed.append({
                        "cif_file_name": path.name,
                        "group": group,
                        "num_motifs_real": len(matches),
                        "num_not_enumerated": len(missing),
                        "motif_max_distance": max_spread,
                        "num_candidates": len(scored),
                        "reason": reason,
                        "in_training_set": trained,
                    })
                    print(f"    not assessed: {member} ({reason}, "
                          f"motif spread {max_spread:.1f} Å)")
                    continue
 
                results[group].append({
                    "cif_file_name": path.name,
                    "num_motifs_real": len(matches),
                    "num_true_pos": tp,
                    "num_false_pos": int((pred & ~is_motif).sum()),
                    "num_true_neg": int((~pred & ~is_motif).sum()),
                    "num_false_neg": len(matches) - tp,
                    "in_training_set": trained,
                })
 
            pd.DataFrame(not_assessed, columns=NOT_ASSESSED_COLUMNS).to_csv(
                split_dir / "results-not-assessed.csv", index=False,
                float_format="%.3f",
            )
            for group, rows in results.items():
                pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                    split_dir / f"results-{group}.csv", index=False
                )
                skipped = [r for r in not_assessed if r["group"] == group]
                summary.append({"split": split, "group": group,
                                "not_assessed": len(skipped),
                                **pooled_metrics(rows)})
                if group == "seen":
                    mates = [r for r in rows if not r["in_training_set"]]
                    summary.append({
                        "split": split, "group": "seen-cluster-mates-only",
                        "not_assessed": sum(not r["in_training_set"] for r in skipped),
                        **pooled_metrics(mates),
                    })
 
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(root / "summary.csv", index=False, float_format="%.4f")
    print("\n" + summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
 
    if errors:
        pd.DataFrame(errors).to_csv(root / "errors.csv", index=False)
        print(f"\n{len(errors)} structure(s) skipped; see {root / 'errors.csv'}")
    print(f"\nResults written to {root}")
 
 
if __name__ == "__main__":
    main()
 