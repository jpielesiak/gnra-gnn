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

def processedIntoCoordinates(dfs: List[pd.DataFrame],arePositive:bool) -> pd.DataFrame:
    """
    Process a list of DataFrames containing C1' atoms, to end up with structure like:
    id, NT1,NT2,NT3,NT4,NT5,NT6,NT7,NT8, is_positive
    where NT1-NT8 are the coordinates of C1' atoms. saved as eg. "72.379,32.575,10.796"
    so for example:
    id, NT1,NT2,NT3,NT4,NT5,NT6,NT7,NT8, is_positive
    0,"72.379,32.575,10.796","77.895,33.755,9.769","84.753,31.765,9.188","84.547,36.507,11.124","82.675,41.303,11.068","78.004,41.475,11.663",1
    """
    all_coordinates = []

    for df in dfs:
        if len(df) != 8:
            print(f"Skipping DataFrame with {len(df)} rows (expected 8)")
            continue
        if "auth_comp_id" not in df.columns:
            print("Warning: 'auth_comp_id' column not found in DataFrame")
            continue
        # legal_letters = {"C", "U", "G", "A"}
        # df = df[df["auth_comp_id"].isin(legal_letters)]
        # if len(df) != 8:
        #     print(f"Skipping DataFrame with {len(df)} rows after filtering (expected 8)")
        #     continue
        # Extract coordinates and format them
        coordinates = [
            f"{row['Cartn_x']},{row['Cartn_y']},{row['Cartn_z']}"
            for _, row in df.iterrows()
        ]
        
        # Create a new row with coordinates and is_positive flag
        is_positive = 1 if arePositive else 0
        new_row = [len(all_coordinates)] + coordinates + [is_positive]
        all_coordinates.append(new_row)

    # Create final DataFrame
    columns = ["id"] + [f"NT{i+1}" for i in range(8)] + ["is_positive"]
    return pd.DataFrame(all_coordinates, columns=columns)
def processIntoSequences(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Processes a list of DataFrames containing C1' atoms to extract sequences, so that each row
    of the DataFrame contains the nucleotide sequence of a molecule in a given dataframe, like
    CGAAAGAA
    CGAAGGAG
    ..."""
    sequences = []

    for df in dfs:
        if len(df) != 8:
            print(f"Skipping DataFrame with {len(df)} rows (expected 8)")
            continue
        # Extract nucleotide sequence from the 'auth_comp_id' column
        #some files have values from auth_asym_id in place of auth_com_id so we must ensure that we use auth_comp_id and only
        #legal letters so C U G and A
        if "auth_comp_id" not in df.columns:
            print("Warning: 'auth_comp_id' column not found in DataFrame")
            continue
        # legal_letters = {"C", "U", "G", "A"}
        # df2 = df[df["auth_comp_id"].isin(legal_letters)]
        # if len(df2) != 8:
        #     print(f"Skipping DataFrame with {len(df2)} rows after filtering (expected 8)")
        #     print(df)
        #     continue
        sequence = "".join(df["auth_comp_id"].tolist())
        sequences.append(sequence)

    # Create final DataFrame
    return pd.DataFrame(sequences, columns=["sequence"])
def filterOutIndexes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out columns with specific indexes from the DataFrame.
    dataframe geometric_features.csv has multiple columns with calculated geomteric features, 
    but we only want fetaures calculated between middle 6 nucleotides. Therefore all features calculated between the first nucleotide
    and the last nucleotide must be filtered out.
    source_file,d01,d02,d03,d04,d05,d06,d07,d12,d13,d14,d15,d16,d17,d23,d24,d25,d26,d27,d34,d35,d36,d37,d45,d46,d47,d56,d57,d67,a012,as012,aa012,a013,as013,aa013,a014,as014,aa014,a015,as015,aa015,a016,as016,aa016,a017,as017,aa017,a023,as023,aa023,a024,as024,aa024,a025,as025,aa025,a026,as026,aa026,a027,as027,aa027,a034,as034,aa034,a035,as035,aa035,a036,as036,aa036,a037,as037,aa037,a045,as045,aa045,a046,as046,aa046,a047,as047,aa047,a056,as056,aa056,a057,as057,aa057,a067,as067,aa067,a123,as123,aa123,a124,as124,aa124,a125,as125,aa125,a126,as126,aa126,a127,as127,aa127,a134,as134,aa134,a135,as135,aa135,a136,as136,aa136,a137,as137,aa137,a145,as145,aa145,a146,as146,aa146,a147,as147,aa147,a156,as156,aa156,a157,as157,aa157,a167,as167,aa167,a234,as234,aa234,a235,as235,aa235,a236,as236,aa236,a237,as237,aa237,a245,as245,aa245,a246,as246,aa246,a247,as247,aa247,a256,as256,aa256,a257,as257,aa257,a267,as267,aa267,a345,as345,aa345,a346,as346,aa346,a347,as347,aa347,a356,as356,aa356,a357,as357,aa357,a367,as367,aa367,a456,as456,aa456,a457,as457,aa457,a467,as467,aa467,a567,as567,aa567,t0123,ts0123,ta0123,t0124,ts0124,ta0124,t0125,ts0125,ta0125,t0126,ts0126,ta0126,t0127,ts0127,ta0127,t0134,ts0134,ta0134,t0135,ts0135,ta0135,t0136,ts0136,ta0136,t0137,ts0137,ta0137,t0145,ts0145,ta0145,t0146,ts0146,ta0146,t0147,ts0147,ta0147,t0156,ts0156,ta0156,t0157,ts0157,ta0157,t0167,ts0167,ta0167,t0234,ts0234,ta0234,t0235,ts0235,ta0235,t0236,ts0236,ta0236,t0237,ts0237,ta0237,t0245,ts0245,ta0245,t0246,ts0246,ta0246,t0247,ts0247,ta0247,t0256,ts0256,ta0256,t0257,ts0257,ta0257,t0267,ts0267,ta0267,t0345,ts0345,ta0345,t0346,ts0346,ta0346,t0347,ts0347,ta0347,t0356,ts0356,ta0356,t0357,ts0357,ta0357,t0367,ts0367,ta0367,t0456,ts0456,ta0456,t0457,ts0457,ta0457,t0467,ts0467,ta0467,t0567,ts0567,ta0567,t1234,ts1234,ta1234,t1235,ts1235,ta1235,t1236,ts1236,ta1236,t1237,ts1237,ta1237,t1245,ts1245,ta1245,t1246,ts1246,ta1246,t1247,ts1247,ta1247,t1256,ts1256,ta1256,t1257,ts1257,ta1257,t1267,ts1267,ta1267,t1345,ts1345,ta1345,t1346,ts1346,ta1346,t1347,ts1347,ta1347,t1356,ts1356,ta1356,t1357,ts1357,ta1357,t1367,ts1367,ta1367,t1456,ts1456,ta1456,t1457,ts1457,ta1457,t1467,ts1467,ta1467,t1567,ts1567,ta1567,t2345,ts2345,ta2345,t2346,ts2346,ta2346,t2347,ts2347,ta2347,t2356,ts2356,ta2356,t2357,ts2357,ta2357,t2367,ts2367,ta2367,t2456,ts2456,ta2456,t2457,ts2457,ta2457,t2467,ts2467,ta2467,t2567,ts2567,ta2567,t3456,ts3456,ta3456,t3457,ts3457,ta3457,t3467,ts3467,ta3467,t3567,ts3567,ta3567,t4567,ts4567,ta4567,gnra
    some features contain more than 2 nucleotides, those should be filtered out as well.
    Args:
        df: Input DataFrame
        indexes: List of row indexes to filter out

    Returns:
        Filtered DataFrame
    """
    # Create a set of indexes to filter out
    filter_set = ["0","8"]
    #drop all columns that contain these indexes in their names
    filtered_df = df.drop(
        columns=[col for col in df.columns if any(i in col for i in filter_set)]
    )
    return filtered_df

def processForSeqAndNtCords():
     # Process positive examples (GNRA motifs)
    print("Processing positive examples from motif_cif_files...")
    positive_dfs = process_cif_files_for_c1_prime("motif_cif_files")

    # Process negative examples
    print("\nProcessing negative examples from negative_cif_files...")
    negative_dfs = process_cif_files_for_c1_prime("negative_cif_files")
    #join all positive dfs into positive_dfs_all dataframe and save it to csv
    if positive_dfs:
        positive_dfs_cord = processedIntoCoordinates(positive_dfs,True)
        positive_dfs_seqs = processIntoSequences(positive_dfs)
    else:
        positive_dfs = pd.DataFrame()
    if negative_dfs:
        negative_dfs_cord = processedIntoCoordinates(negative_dfs,True)
        negative_dfs_seqs = processIntoSequences(negative_dfs)
        
    positive_dfs_cord.to_csv('positve.csv', index=False)
    negative_dfs_cord.to_csv('negative.csv', index=False)
    positive_dfs_seqs.to_csv('positve_seq.csv', index=False)
    negative_dfs_seqs.to_csv('negative_seq.csv', index=False)

if __name__ == "__main__":
    #read geometric_features.csv file as dataframe
    geometric_features_file = "geometric_features.csv"
    
    if not os.path.exists(geometric_features_file):
        print(f"File {geometric_features_file} does not exist. Please run the previous script to generate it.")
        exit(1)
    #open geometric_features.csv as dataframe
    gf = pd.read_csv(geometric_features_file)
    df= filterOutIndexes(gf)
    df.to_csv('filtered_geometric_features.csv', index=False)
    #processForSeqAndNtCords()

# C:\Users\jmp\Downloads\rnative-competition-tests\gnra-gnn\08-data-extraction-GNN.py
# /mnt/c/Users/jmp/Downloads/rnative-competition-tests/gnra-gnn/08-data-extraction-GNN.py
# source ~/envs/rnapolis-env/bin/activate