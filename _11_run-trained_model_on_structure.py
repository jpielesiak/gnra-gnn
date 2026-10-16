#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import pickle
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Residue, Structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bundled classical model on all 8-nt windows from a structure"
        )
    )
    parser.add_argument("structure", type=Path, help="Path to an input .cif or .pdb")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a bundled pickle from classical.ipynb",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV path (default: derived from input/model names)",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Optional output PNG path for per-window positive-class probabilities",
    )
    parser.add_argument(
        "--structure-model",
        type=int,
        help="Optional 1-based structure model number for multi-model files",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def format_residue_id(residue: Residue) -> str:
    chain_id = clean_text(residue.chain_id)
    insertion_code = clean_text(residue.insertion_code)
    return f"{chain_id}:{residue.residue_number}{insertion_code}"


def default_output_csv_path(structure_path: Path, model_path: Path) -> Path:
    return structure_path.with_name(
        f"{structure_path.stem}-{model_path.stem}-predictions.csv"
    )


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    with open(model_path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"{model_path} is not a bundled model artifact. Regenerate it from classical.ipynb."
        )

    required_keys = {
        "classifier",
        "classifier_name",
        "scaler",
        "feature_columns",
        "window_size",
    }
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        raise ValueError(
            f"{model_path} is missing required bundle keys: {', '.join(missing_keys)}"
        )

    if payload["window_size"] != 8:
        raise ValueError(
            f"This script only supports 8-nt windows, got window_size={payload['window_size']}"
        )

    feature_columns = payload["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"{model_path} has invalid feature_columns metadata")

    classifier = payload["classifier"]
    scaler = payload["scaler"]
    expected_feature_count = len(feature_columns)

    classifier_feature_count = getattr(classifier, "n_features_in_", None)
    if (
        classifier_feature_count is not None
        and int(classifier_feature_count) != expected_feature_count
    ):
        raise ValueError(
            f"{model_path} classifier expects {classifier_feature_count} features but bundle metadata lists {expected_feature_count}"
        )

    scaler_feature_count = getattr(scaler, "n_features_in_", None)
    if (
        scaler_feature_count is not None
        and int(scaler_feature_count) != expected_feature_count
    ):
        raise ValueError(
            f"{model_path} scaler expects {scaler_feature_count} features but bundle metadata lists {expected_feature_count}"
        )

    return payload


def parse_structure_file(structure_path: Path) -> pd.DataFrame:
    suffix = structure_path.suffix.lower()
    with open(structure_path, "r") as handle:
        if suffix == ".cif":
            atoms_df = parse_cif_atoms(handle)
        elif suffix == ".pdb":
            atoms_df = parse_pdb_atoms(handle)
        else:
            raise ValueError(
                f"Unsupported structure format '{structure_path.suffix}'. Expected .cif or .pdb."
            )

    if atoms_df.empty:
        raise ValueError(f"No atoms were parsed from {structure_path}")

    return atoms_df


def select_structure_model(
    atoms_df: pd.DataFrame, requested_model: Optional[int]
) -> tuple[pd.DataFrame, Optional[int]]:
    structure_format = atoms_df.attrs.get("format")
    if structure_format == "PDB":
        model_column = "model"
    elif structure_format == "mmCIF":
        model_column = "pdbx_PDB_model_num"
    else:
        model_column = None

    if model_column is None or model_column not in atoms_df.columns:
        return atoms_df, None

    model_values = [
        int(value)
        for value in atoms_df[model_column].dropna().tolist()
        if not pd.isna(value)
    ]
    available_models = list(dict.fromkeys(model_values))
    if not available_models:
        return atoms_df, None

    selected_model = (
        requested_model if requested_model is not None else available_models[0]
    )
    if selected_model not in available_models:
        raise ValueError(
            f"Model {selected_model} not found. Available models: {available_models}"
        )

    filtered = atoms_df[atoms_df[model_column] == selected_model].copy()
    filtered.attrs = atoms_df.attrs.copy()
    return filtered, selected_model


def get_nucleotide_residues_by_chain(structure: Structure) -> dict[str, list[Residue]]:
    residues_by_chain: dict[str, list[Residue]] = {}
    for residue in structure.residues:
        if not residue.is_nucleotide:
            continue
        chain_id = clean_text(residue.chain_id)
        residues_by_chain.setdefault(chain_id, []).append(residue)
    return residues_by_chain


def calculate_distance(
    p1: tuple[float, float, float], p2: tuple[float, float, float]
) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def calculate_planar_angle(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> tuple[float, float, float]:
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0, 0.0, 1.0

    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.acos(cos_angle)
    sin_angle = math.sqrt(1.0 - cos_angle * cos_angle)
    return angle, sin_angle, cos_angle


def calculate_torsion_angle(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    p4: tuple[float, float, float],
) -> tuple[float, float, float]:
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])

    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm == 0 or n2_norm == 0:
        return 0.0, 0.0, 1.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_angle = np.dot(n1, n2)
    sin_angle = np.dot(np.cross(n1, n2), v2 / np.linalg.norm(v2))
    torsion_angle = math.atan2(sin_angle, cos_angle)
    return torsion_angle, sin_angle, cos_angle


def first_c1_prime_coordinates(
    residue: Residue,
) -> Optional[tuple[float, float, float]]:
    atom_name_col = residue._col("atom_name")
    x_col = residue._col("x")
    y_col = residue._col("y")
    z_col = residue._col("z")

    matching = residue.atoms[residue.atoms[atom_name_col] == "C1'"]
    if matching.empty:
        return None

    row = matching.iloc[0]
    coords = (row[x_col], row[y_col], row[z_col])
    if any(pd.isna(value) for value in coords):
        return None

    return float(coords[0]), float(coords[1]), float(coords[2])


def calculate_window_features(
    coords: list[tuple[float, float, float]],
) -> dict[str, float]:
    if len(coords) != 8:
        raise ValueError(f"Expected exactly 8 coordinates, got {len(coords)}")

    result: dict[str, float] = {}

    for i, j in combinations(range(8), 2):
        result[f"d{i}{j}"] = calculate_distance(coords[i], coords[j])

    for i, j, k in combinations(range(8), 3):
        _, sin_angle, cos_angle = calculate_planar_angle(
            coords[i], coords[j], coords[k]
        )
        result[f"as{i}{j}{k}"] = sin_angle
        result[f"aa{i}{j}{k}"] = cos_angle

    for i, j, k, l in combinations(range(8), 4):
        _, sin_torsion, cos_torsion = calculate_torsion_angle(
            coords[i], coords[j], coords[k], coords[l]
        )
        result[f"ts{i}{j}{k}{l}"] = sin_torsion
        result[f"ta{i}{j}{k}{l}"] = cos_torsion

    return result


def build_window_rows(
    structure_path: Path, residues_by_chain: dict[str, list[Residue]], window_size: int
) -> tuple[list[dict[str, Any]], pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, float]] = []
    skipped_windows = 0
    global_window_index = 1

    for chain_id, chain_residues in residues_by_chain.items():
        if len(chain_residues) < window_size:
            continue

        for chain_window_index, start in enumerate(
            range(len(chain_residues) - window_size + 1), start=1
        ):
            window_residues = chain_residues[start : start + window_size]
            coords: list[tuple[float, float, float]] = []
            for residue in window_residues:
                c1_prime_coords = first_c1_prime_coordinates(residue)
                if c1_prime_coords is None:
                    skipped_windows += 1
                    break
                coords.append(c1_prime_coords)
            else:
                start_residue = window_residues[0]
                end_residue = window_residues[-1]
                rows.append(
                    {
                        "source_structure": structure_path.name,
                        "window_index": global_window_index,
                        "chain_id": chain_id,
                        "chain_window_index": chain_window_index,
                        "sequence": "".join(
                            residue.one_letter_name for residue in window_residues
                        ),
                        "start_residue_number": start_residue.residue_number,
                        "start_insertion_code": clean_text(
                            start_residue.insertion_code
                        ),
                        "start_residue_name": start_residue.residue_name,
                        "end_residue_number": end_residue.residue_number,
                        "end_insertion_code": clean_text(end_residue.insertion_code),
                        "end_residue_name": end_residue.residue_name,
                        "residue_ids": ",".join(
                            format_residue_id(residue) for residue in window_residues
                        ),
                    }
                )
                feature_rows.append(calculate_window_features(coords))
                global_window_index += 1

    return rows, pd.DataFrame(feature_rows), skipped_windows


def positive_class_probabilities(
    classifier: Any, features: np.ndarray, positive_label: Any
):
    if not hasattr(classifier, "predict_proba"):
        return None

    probabilities = classifier.predict_proba(features)
    classes = list(getattr(classifier, "classes_", []))
    if positive_label in classes:
        positive_index = classes.index(positive_label)
    elif len(classes) == 2:
        positive_index = 1
    else:
        return None

    return probabilities[:, positive_index]



def save_probability_plot(results_df: pd.DataFrame, output_plot: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(
        results_df["window_index"],
        results_df["probability"],
        marker="o",
        linewidth=1.5,
        markersize=3,
    )
    plt.xlabel("Window index")
    plt.ylabel("Positive-class probability")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    model_bundle = load_model_bundle(args.model)
    window_size = model_bundle["window_size"]
    feature_columns = model_bundle["feature_columns"]
    classifier = model_bundle["classifier"]
    scaler = model_bundle["scaler"]
    positive_label = model_bundle.get("positive_label", True)

    atoms_df = parse_structure_file(args.structure)
    atoms_df, selected_model = select_structure_model(atoms_df, args.structure_model)
    structure = Structure(atoms_df)
    residues_by_chain = get_nucleotide_residues_by_chain(structure)

    if not residues_by_chain:
        raise ValueError(f"No nucleotide residues found in {args.structure}")

    row_metadata, features_df, skipped_windows = build_window_rows(
        args.structure, residues_by_chain, window_size
    )
    if features_df.empty:
        raise ValueError(f"No valid {window_size}-nt windows found in {args.structure}")

    missing_features = sorted(set(feature_columns).difference(features_df.columns))
    if missing_features:
        raise ValueError(
            "Generated features do not match the model bundle. Missing columns: "
            + ", ".join(missing_features[:10])
        )

    feature_matrix = features_df.reindex(columns=feature_columns)
    if feature_matrix.isna().any().any():
        raise ValueError(
            "Feature matrix contains NaN values after reindexing; refusing to run inference"
        )

    transformed = scaler.transform(feature_matrix)
    predictions = classifier.predict(transformed)
    probabilities = positive_class_probabilities(
        classifier, transformed, positive_label
    )

    results_df = pd.DataFrame(row_metadata)
    results_df["classifier_name"] = model_bundle["classifier_name"]
    results_df["model_path"] = str(args.model)
    results_df["structure_model"] = selected_model
    results_df["prediction"] = predictions

    if probabilities is not None:
        results_df["probability"] = probabilities

    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else default_output_csv_path(args.structure, args.model)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    print(f"Parsed {len(residues_by_chain)} nucleotide chains")
    print(f"Generated {len(results_df)} valid windows")
    if skipped_windows:
        print(f"Skipped {skipped_windows} windows due to missing C1' coordinates")
    print(f"Saved predictions to {output_csv}")

    if args.output_plot is not None:
        if probabilities is None:
            print(
                "The selected classifier does not expose predict_proba; skipping probability plot."
            )
        else:
            args.output_plot.parent.mkdir(parents=True, exist_ok=True)
            save_probability_plot(results_df, args.output_plot)
            print(f"Saved probability plot to {args.output_plot}")


if __name__ == "__main__":
    main()
