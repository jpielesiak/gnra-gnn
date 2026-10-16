#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import math
import os
from pathlib import Path
from typing import List, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms


def calculate_distance(
    p1: Tuple[float, float, float], p2: Tuple[float, float, float]
) -> float:
    """
    Calculate Euclidean distance between two 3D points.

    Args:
        p1: First point as (x, y, z) tuple
        p2: Second point as (x, y, z) tuple

    Returns:
        Distance between the two points
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def calculate_planar_angle(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    p3: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Calculate planar angle between three points (angle at p2).

    Args:
        p1: First point as (x, y, z) tuple
        p2: Vertex point as (x, y, z) tuple
        p3: Third point as (x, y, z) tuple

    Returns:
        Tuple of (angle_radians, sin_angle, cos_angle)
    """
    # Convert to numpy arrays for easier vector operations
    v1 = np.array(p1) - np.array(p2)  # Vector from p2 to p1
    v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3

    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0, 0.0, 1.0

    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Clamp to valid range for arccos to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate angle and sine (using sqrt for better numerical stability)
    angle = math.acos(cos_angle)
    sin_angle = math.sqrt(1.0 - cos_angle * cos_angle)

    return angle, sin_angle, cos_angle


def calculate_torsion_angle(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    p3: Tuple[float, float, float],
    p4: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Calculate torsion (dihedral) angle between four points.

    Args:
        p1: First point as (x, y, z) tuple
        p2: Second point as (x, y, z) tuple
        p3: Third point as (x, y, z) tuple
        p4: Fourth point as (x, y, z) tuple

    Returns:
        Tuple of (angle_radians, sin_angle, cos_angle)
    """
    # Convert to numpy arrays
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])

    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3

    # Calculate normal vectors to the planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Normalize the normal vectors
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    # Avoid division by zero
    if n1_norm == 0 or n2_norm == 0:
        return 0.0, 0.0, 1.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Calculate the torsion angle
    cos_angle = np.dot(n1, n2)
    sin_angle = np.dot(np.cross(n1, n2), v2 / np.linalg.norm(v2))

    # Use atan2 to get the correct sign and full range
    torsion_angle = math.atan2(sin_angle, cos_angle)

    return torsion_angle, sin_angle, cos_angle


def calculate_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all geometric features from a dataframe with exactly 8 C1' atoms.

    Args:
        df: DataFrame with 8 rows containing C1' atoms with coordinates

    Returns:
        Single-row DataFrame with all geometric features:
        - source_file: original filename
        - d{i}{j}: distances between atoms i and j
        - a{i}{j}{k}: planar angles for triplets i,j,k (angle at j)
        - t{i}{j}{k}{l}: torsion angles for quadruplets i,j,k,l
    """
    if len(df) != 8:
        raise ValueError(f"Expected exactly 8 atoms, got {len(df)}")

    # Extract coordinates and source file
    coords = []
    for _, row in df.iterrows():
        coords.append((row["Cartn_x"], row["Cartn_y"], row["Cartn_z"]))

    source_file = df["source_file"].iloc[0]

    # Initialize result dictionary
    result = {"source_file": source_file}

    # Calculate all pairwise distances (28 pairs for 8 atoms)
    for i, j in combinations(range(8), 2):
        distance = calculate_distance(coords[i], coords[j])
        result[f"d{i}{j}"] = distance

    # Calculate all planar angles (56 triplets for 8 atoms)
    for i, j, k in combinations(range(8), 3):
        angle, sin_angle, cos_angle = calculate_planar_angle(
            coords[i], coords[j], coords[k]
        )
        result[f"a{i}{j}{k}"] = angle
        result[f"as{i}{j}{k}"] = sin_angle
        result[f"aa{i}{j}{k}"] = cos_angle

    # Calculate all torsion angles (70 quadruplets for 8 atoms)
    for i, j, k, l in combinations(range(8), 4):
        torsion, sin_torsion, cos_torsion = calculate_torsion_angle(
            coords[i], coords[j], coords[k], coords[l]
        )
        result[f"t{i}{j}{k}{l}"] = torsion
        result[f"ts{i}{j}{k}{l}"] = sin_torsion
        result[f"ta{i}{j}{k}{l}"] = cos_torsion

    # Return as single-row DataFrame
    return pd.DataFrame([result])


def process_cif_files_for_c1_prime(directory: str) -> List[pd.DataFrame]:
    """
    Process all *.cif files in a directory, extract C1' atoms, and return a dataframe.

    Only includes files that have exactly 8 C1' atoms.

    Args:
        directory: Path to directory containing .cif files

    Returns:
        DataFrame with C1' atoms and a 'source_file' column indicating the origin file
    """
    all_dataframes = []

    # Find all .cif files in the directory
    cif_pattern = os.path.join(directory, "*.cif")
    cif_files = sorted(glob.glob(cif_pattern))

    print(f"Found {len(cif_files)} .cif files in {directory}")

    for cif_file in cif_files:
        try:
            # Parse the CIF file
            with open(cif_file, "r") as fd:
                atoms_df = parse_cif_atoms(fd)

            # Filter for C1' atoms only
            c1_prime_atoms = atoms_df[atoms_df["auth_atom_id"] == "C1'"]

            # Remove duplicate C1' atoms within the same residue - keep only the first occurrence
            # Group by residue identifiers and take the first occurrence of each group
            c1_prime_atoms = c1_prime_atoms.drop_duplicates(
                subset=["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"],
                keep="first",
            )

            # Check if we have exactly 8 C1' atoms
            if len(c1_prime_atoms) == 8:
                # Add source file column
                filename = Path(cif_file).stem  # Get filename without extension
                c1_prime_atoms = c1_prime_atoms.copy()
                c1_prime_atoms["source_file"] = filename

                all_dataframes.append(c1_prime_atoms)
                print(f"  ✓ {filename}: Found exactly 8 C1' atoms")
            else:
                filename = Path(cif_file).stem
                print(
                    f"  ✗ {filename}: Found {len(c1_prime_atoms)} C1' atoms (expected 8)"
                )

        except Exception as e:
            filename = Path(cif_file).stem
            print(f"  ✗ {filename}: Error parsing file - {e}")

    # Combine all dataframes
    if all_dataframes:
        print(
            f"\nSuccessfully processed {len(all_dataframes)} files with exactly 8 C1' atoms"
        )
        return all_dataframes
    else:
        print("\nNo files with exactly 8 C1' atoms found")
        return [pd.DataFrame()]


if __name__ == "__main__":
    # Process positive examples (GNRA motifs)
    print("Processing positive examples from motif_cif_files...")
    positive_dfs = process_cif_files_for_c1_prime("motif_cif_files")

    positive_features = []
    for df in positive_dfs:
        if not df.empty:
            features = calculate_geometric_features(df)
            features["gnra"] = True
            positive_features.append(features)

    # Process negative examples
    print("\nProcessing negative examples from negative_cif_files...")
    negative_dfs = process_cif_files_for_c1_prime("negative_cif_files")

    negative_features = []
    for df in negative_dfs:
        if not df.empty:
            features = calculate_geometric_features(df)
            features["gnra"] = False
            negative_features.append(features)

    # Combine all features
    all_features = []
    if positive_features:
        all_features.extend(positive_features)
    if negative_features:
        all_features.extend(negative_features)

    if all_features:
        # Concatenate all feature dataframes
        final_df = pd.concat(all_features, ignore_index=True)

        # Save to CSV
        output_file = "geometric_features.csv"
        final_df.to_csv(output_file, index=False)

        print(f"\nSaved {len(final_df)} samples to {output_file}")
        print(f"Positive samples: {len(positive_features)}")
        print(f"Negative samples: {len(negative_features)}")
        print(f"Total features per sample: {len(final_df.columns)}")
    else:
        print("\nNo valid samples found to process")
