import glob
import importlib
import math
import os
from pathlib import Path
from typing import List, Tuple,Any, Dict, Optional
from itertools import combinations
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, write_cif
from rnapolis.tertiary_v2 import Residue, Structure

import importlib.util

spec = importlib.util.spec_from_file_location(
    "generate_csv_module",  # arbitrary name
    "06-generate-csv.py"
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def process_cif_files_for_c1_prime(directory: str, ids: List[List[int]] = None, filepath: str = None) -> Tuple[List[pd.DataFrame], int]:
    """
    Process all *.cif files in a directory, extract C1' atoms, and return a dataframe.

    Turns one mmcif file into a dataframe of each group of 8 C1' atoms.

    Args:
        directory: Path to directory containing .cif files
        ids: List of lists of integers, where each sublist contains starting positions 
             of GNRA motifs for the corresponding file. If provided, marks groups as gnra=True/False.
        filepath: Optional path to a single .cif file to process instead of the whole directory

    Returns:
        Tuple of:
        - DataFrame with C1' atoms and a 'source_file' column indicating the origin file
        - Number of CIF files processed
    """
    all_dataframes = []
    # Find all .cif files in the directory
    if filepath:
        cif_files = [filepath]
    else:
        cif_pattern = os.path.join(directory, "*.cif")
        cif_files = sorted(glob.glob(cif_pattern))

    print(f"Found {len(cif_files)} .cif files in {directory}")

    # Default to empty list for each file if ids not provided
    if ids is None:
        ids = [[] for _ in range(len(cif_files))]

    for file_idx, cif_file in enumerate(cif_files):
        # Get GNRA positions for this file, default to empty list if index out of range
        gnra_positions = ids[file_idx] if file_idx < len(ids) else []
        
        try:
            # Parse the CIF file
            with open(cif_file, "r") as fd:
                atoms_df = parse_cif_atoms(fd)

            # Filter for C1' atoms only
            c1_prime_atoms = atoms_df[atoms_df["auth_atom_id"] == "C1'"]
            # ── NEW: keep only standard RNA residues ──────────────────────────────────────
            RNA_RESIDUES = {
                # Standard
                "A", "U", "G", "C",
                # Legacy 3-letter codes
                "ADE", "URA", "URI", "GUA", "CYT",
                # Common modified RNA bases (extend as needed)
                "PSU", "5MC", "5MU", "H2U", "M2G",
            }
            c1_prime_atoms = c1_prime_atoms[
                c1_prime_atoms["label_comp_id"].isin(RNA_RESIDUES)
            ]
            # ─────────────────────────────────────────────────────────────────────────────
            # Remove duplicate C1' atoms within the same residue - keep only the first occurrence
            # Group by residue identifiers and take the first occurrence of each group
            c1_prime_atoms = c1_prime_atoms.drop_duplicates(
                subset=["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"],
                keep="first",
            )
            # Check if we have exactly 8 C1' atoms
            for i in range(0, len(c1_prime_atoms) - 7):
                group_of_8 = c1_prime_atoms.iloc[i:i+8]
                # Add source file column
                filename = Path(cif_file).stem  # Get filename without extension
                group_of_8 = group_of_8.copy()
                group_of_8["source_file"] = f"{filename}_NT:{i}_{i+7}"
                
                # Mark as GNRA if this starting position is in the ids list for this file
                is_gnra = i in gnra_positions
                group_of_8["gnra"] = is_gnra
                
                # Extract nucleotide sequence (residue names) for this group of 8
                # Get the residue names from the group using label_comp_id
                residue_names = group_of_8["label_comp_id"].tolist()
                # Map 3-letter codes to 1-letter codes (RNA nucleotides)
                three_to_one = {
                    # Standard single-letter
                    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
                    # Alternate 3-letter names for standard bases
                    'ADE': 'A', 'URA': 'U', 'URI': 'U', 'GUA': 'G', 'CYT': 'C',
                    # Modified bases → closest standard base
                    'PSU': 'U',  # pseudouridine
                    '5MC': 'C',  # 5-methylcytosine
                    '5MU': 'U',  # 5-methyluridine (ribothymidine)
                    'H2U': 'U',  # dihydrouridine
                    'M2G': 'G',  # N2-methylguanosine
                    '2MG': 'G',  # 2-methylguanosine
                    '7MG': 'G',  # 7-methylguanosine
                    '1MA': 'A',  # 1-methyladenosine
                    '6MA': 'A',  # N6-methyladenosine
                    'OMG': 'G',  # 2'-O-methylguanosine
                    'OMC': 'C',  # 2'-O-methylcytosine
                    'OMA': 'A',  # 2'-O-methyladenosine
                    'OMU': 'U',  # 2'-O-methyluridine
                }
                # Then when building seq, skip the whole window if any residue is unmapped:
                mapped = [three_to_one.get(str(rn)) for rn in residue_names]
                if any(m is None for m in mapped):
                    print(f" {filename}: Skipping {i}-{i+7}, unknown residue code(s): "
                        f"{[rn for rn in residue_names if str(rn) not in three_to_one]}")
                    continue
                seq = ''.join(mapped)
                #seq = ''.join([three_to_one.get(str(rn), rn) for rn in residue_names])
                group_of_8["seq"] = seq

                all_dataframes.append(group_of_8)
                print(f" {filename}: Saving {i} {i+7} C1' atoms (gnra={is_gnra}, seq={seq})")


        except Exception as e:
            filename = Path(cif_file).stem
            print(f"  ✗ {filename}: Error parsing file - {e}")

    # Combine all dataframes
    if all_dataframes:
        print(
            f"\nSuccessfully processed {len(all_dataframes)} files with exactly 8 C1' atoms"
        )
        return all_dataframes, len(cif_files)
    else:
        print("\nNo files with exactly 8 C1' atoms found")
        return [pd.DataFrame()], len(cif_files)
    



def load_gnra_motifs(
    filename: str = "gnra_motifs_by_pdb.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load GNRA motifs from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)

def find_motif_residue_indices(
    residues: List[Residue], motifs: List[Dict[str, Any]], pdb_id: str = None
) -> List[Dict[str, Any]]:
    """Find residue indices for each motif, extending to 8 residues."""
    motif_data = []

    # Normalize PDB ID for comparison (strip extension, uppercase)
    target_pdb = None
    if pdb_id:
        target_pdb = Path(pdb_id).stem.upper()

    for motif_idx, motif in enumerate(motifs):
        # FIX 1: correct key names
        motif_key = motif.get("motif_id", f"motif_{motif_idx}")
        alignment = motif.get("alignment", {})

        # FIX 2: find ALL alignment entries that belong to our PDB (not just the first one)
        matching_unit_ids_list = []
        for align_key, unit_id_list in alignment.items():
            if not unit_id_list:
                continue
            # FIX 3: parse the pipe-separated format: PDB|model|chain|nuc|resnum[|ins_code]
            entry_pdb = unit_id_list[0].split("|")[0].upper()
            if target_pdb and entry_pdb != target_pdb:
                continue
            # Collect ALL matching alignment entries, not just the first one
            matching_unit_ids_list.append(unit_id_list)

        if not matching_unit_ids_list:
            continue

        # Process each matching alignment entry (each represents a different motif instance)
        for matching_unit_ids in matching_unit_ids_list:
            indices = []
            motif_residues = []
            motif_chain_ids = set()

            for unit_id_str in matching_unit_ids:
                parts = unit_id_str.split("|")
                if len(parts) < 5:
                    continue

                chain_id = parts[2]
                try:
                    residue_number = int(parts[4])
                except ValueError:
                    continue
                insertion_code = parts[5] if len(parts) > 5 else ""
                motif_chain_ids.add(chain_id)

                for i, residue in enumerate(residues):
                    res_ins = residue.insertion_code or ""
                    if (
                        residue.chain_id == chain_id
                        and residue.residue_number == residue_number
                        and res_ins == insertion_code
                    ):
                        indices.append(i)
                        motif_residues.append(residue)
                        break

            if len(indices) != 6:
                print(f"    Warning: {motif_key} - Expected 6 residues, found {len(indices)}: {indices}")
                continue

            sorted_indices = sorted(indices)
            is_consecutive = all(
                sorted_indices[i] + 1 == sorted_indices[i + 1] for i in range(5)
            )
            if not is_consecutive:
                print(f"    Warning: {motif_key} - Residues are not consecutive: {sorted_indices}")
                continue

            min_idx = min(sorted_indices)
            max_idx = max(sorted_indices)

            if min_idx == 0 or max_idx == len(residues) - 1:
                print(f"    Warning: {motif_key} - Cannot extend to 8 residues (boundary constraints)")
                continue

            before_residue = residues[min_idx - 1]
            after_residue = residues[max_idx + 1]
            motif_chain = residues[sorted_indices[0]].chain_id

            if before_residue.chain_id != motif_chain or after_residue.chain_id != motif_chain:
                print(
                    f"    Warning: {motif_key} - Cannot extend to 8 residues "
                    f"(chain mismatch: motif={motif_chain}, "
                    f"before={before_residue.chain_id}, after={after_residue.chain_id})"
                )
                continue

            extended_indices = [min_idx - 1] + sorted_indices + [max_idx + 1]
            extended_residues = [residues[i] for i in extended_indices]

            motif_data.append(
                {
                    "motif_key": motif_key,
                    "indices": extended_indices,
                    "residues": extended_residues,
                    "chains": motif_chain_ids,
                }
            )

    return motif_data

# def find_motif_residue_indices(
#     residues: List[Residue], motifs: List[Dict[str, Any]]
# ) -> List[Dict[str, Any]]:
#     """Find residue indices and residue objects for each motif's unit_ids, extending to 8 residues."""
#     motif_data = []

#     for motif_idx, motif in enumerate(motifs):
#         indices = []
#         motif_residues = []
#         unit_ids = motif.get("unit_ids", [])
#         motif_key = motif.get("motif_key", f"motif_{motif_idx}")

#         # Track chains for this motif
#         motif_chain_ids = set()

#         for unit_id_dict in unit_ids:
#             chain_id = unit_id_dict.get("chain_id")
#             motif_chain_ids.add(chain_id)

#             # Find matching residue by comparing unit_id components
#             for i, residue in enumerate(residues):
#                 unit_insertion_code = unit_id_dict.get("insertion_code", "")
#                 residue_insertion_code = residue.insertion_code or ""

#                 if (
#                     residue.chain_id == chain_id
#                     and residue.residue_number == unit_id_dict.get("residue_number")
#                     and residue_insertion_code == unit_insertion_code
#                 ):
#                     indices.append(i)
#                     motif_residues.append(residue)
#                     break

#         # Log when we don't find exactly 6 indices
#         if len(indices) != 6:
#             print(
#                 f"    Warning: {motif_key} - Expected 6 residues, found {len(indices)}: {indices}"
#             )
#             continue  # Skip adding this motif to motif_data

#         # Log when indices are not consecutive
#         sorted_indices = sorted(indices)
#         is_consecutive = all(
#             sorted_indices[i] + 1 == sorted_indices[i + 1] for i in range(5)
#         )
#         if not is_consecutive:
#             print(
#                 f"    Warning: {motif_key} - Residues are not consecutive: {sorted_indices}"
#             )
#             continue  # Skip adding this motif to motif_data

#         # Extend to 8 residues (add 1 before and 1 after)
#         min_idx = min(sorted_indices)
#         max_idx = max(sorted_indices)

#         # Check if we can add residues before and after
#         if min_idx == 0 or max_idx == len(residues) - 1:
#             print(
#                 f"    Warning: {motif_key} - Cannot extend to 8 residues (boundary constraints)"
#             )
#             continue  # Skip adding this motif to motif_data

#         # Check if extended residues are from the same chain as the motif
#         before_residue = residues[min_idx - 1]
#         after_residue = residues[max_idx + 1]
#         motif_chain = residues[sorted_indices[0]].chain_id

#         if (
#             before_residue.chain_id != motif_chain
#             or after_residue.chain_id != motif_chain
#         ):
#             print(
#                 f"    Warning: {motif_key} - Cannot extend to 8 residues (chain mismatch: motif={motif_chain}, before={before_residue.chain_id}, after={after_residue.chain_id})"
#             )
#             continue  # Skip adding this motif to motif_data

#         # Create extended indices and residues
#         extended_indices = [min_idx - 1] + sorted_indices + [max_idx + 1]
#         extended_residues = [residues[i] for i in extended_indices]

#         motif_data.append(
#             {
#                 "motif_key": motif_key,
#                 "indices": extended_indices,
#                 "residues": extended_residues,
#                 "chains": motif_chain_ids,
#             }
#         )

#     return motif_data

def parse_and_process_mmcif_file(folder: str, pdb_id: str, motifs: List[Dict[str, Any]]) -> bool:
    mmcif_file = os.path.join(folder, f"{pdb_id}")
    if not os.path.exists(mmcif_file):
        print(f"  Warning: {mmcif_file} not found")
        return []

    try:
        print(f"Parsing {mmcif_file}...")
        with open(mmcif_file, "r") as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)

        #residues = [residue for residue in structure.residues if residue.is_nucleotide]
        
        residues = [
            residue for residue in structure.residues
            if residue.is_nucleotide
            and residue.residue_name in {
                "A", "U", "G", "C",
                "ADE", "URA", "URI", "GUA", "CYT",
                "PSU", "5MC", "5MU", "H2U", "M2G",
            }
        ]
        # FIX: pass pdb_id so we can filter by PDB in the alignment dict
        motif_data = find_motif_residue_indices(residues, motifs, pdb_id=pdb_id)

        positions = []
        for motif_dict in motif_data:
            first_pos = motif_dict["indices"][1]  # [0] is extended-before, [1] is true start
            positions.append(first_pos)

            residue_summary = " -> ".join(
                f"{r.chain_id}.{r.residue_name}{r.residue_number}"
                for r in motif_dict["residues"]
            )
            print(f"    {motif_dict['motif_key']} @ position {first_pos}: {residue_summary}")

        print(positions)
        return positions

    except Exception as e:
        print(f"  Error parsing {pdb_id}: {e}")
        return []

# def parse_and_process_mmcif_file(folder:str,pdb_id: str, motifs: List[Dict[str, Any]]) -> bool:
#     """Parse mmCIF file for a PDB ID and immediately process its motifs."""
#     #mmcif_file = f"mmcif_files/{pdb_id}.cif"
#     mmcif_file = os.path.join(folder, f"{pdb_id}")
#     if not os.path.exists(mmcif_file):
#         print(f"  Warning: {mmcif_file} not found")
#         return False

#     try:
#         print(f"Parsing {mmcif_file}...")
#         with open(mmcif_file, "r") as f:
#             atoms_df = parse_cif_atoms(f)
#         structure = Structure(atoms_df)
#         print(f"  Successfully parsed {pdb_id}")

#         # Process motifs immediately
#         print(f"Processing motifs for {pdb_id}...")
#         residues = [residue for residue in structure.residues if residue.is_nucleotide]
#         motif_data = find_motif_residue_indices(residues, motifs)

#         print(f"  Found {len(residues)} residues")
#         print(f"  Processed {len(motifs)} motifs")
#         print(residues[0])
#         print("-----")
#         print(motifs[0])

#         # Process valid motifs and extract CIF files
#         for motif_dict in motif_data:
#             motif_key = motif_dict["motif_key"]
#             indices = motif_dict["indices"]
#             print(
#                 f"    Motif {motif_key}: {len(indices)} residues at indices {indices}"
#             )

#         return True

#     except Exception as e:
#         print(f"  Error parsing {pdb_id}: {e}")
#         return False



def parse_and_process_mmcif_files(directory: str,motifs,filepath: str = None,):
    """
    Process all *.cif files in a directory, and find motifs in them.
    Args:
        directory: Path to directory containing .cif files
        filepath: Optional path to a single .cif file to process instead of the whole directory

    Returns:
        list of lists of positions of motifs in each file
    """
    all_dataframes = []
    # Find all .cif files in the directory
    if filepath:
        cif_files = [filepath]
    else:
        cif_pattern = os.path.join(directory, "*.cif")
        cif_files = sorted(glob.glob(cif_pattern))

    print(f"Found {len(cif_files)} .cif files in {directory}")
    found_motifs = []
    for cif_file in cif_files:
        #ensure file has its extension not lost
        file = Path(cif_file).name 
        print(f"Processing {file}...")
        found =parse_and_process_mmcif_file(directory,file,motifs)
        found_motifs.append(found)
    print(f"Found motifs: {found_motifs}")
    return found_motifs



#this script will take a mmcif file and parse it into a dataset containing 8 C1' atoms per row
#to make it compatible with the gnn, 08a-generate-coordinates.py should be called on the result 


if __name__ == "__main__":
    # Process positive examples (GNRA motifs)

    
    ids = parse_and_process_mmcif_files("test_cif_files",load_gnra_motifs("hl_3.97.json"))#,"1JID.cif"

    
    print("Processing positive examples from motif_cif_files...")
    dfs, numfiles = process_cif_files_for_c1_prime("test_cif_files",ids)

    features = []
    for df in dfs:
        if not df.empty:
            # Extract seq and gnra from the input dataframe before calculating features
            seq_value = df["seq"].iloc[0] if "seq" in df.columns else ""
            gnra_value = df["gnra"].iloc[0] if "gnra" in df.columns else False
            
            feats = module.calculate_geometric_features(df)
            # Add seq and gnra columns to the features dataframe
            feats["seq"] = seq_value
            feats["gnra"] = gnra_value
            features.append(feats)

    if features:
        # Concatenate all feature dataframes
        final_df = pd.concat(features, ignore_index=True)

        # Reorder columns: move 'seq' to second-to-last, 'gnra' to last
        cols = [c for c in final_df.columns if c not in ('seq', 'gnra')]
        cols.extend(['seq', 'gnra'])
        final_df = final_df[cols]
        final_df = final_df.rename(columns={"gnra": "is_positive"})

        # Save to CSV
        output_file = "geometric_features_to_test8.csv"
        final_df.to_csv(output_file, index=False)

        print(f"\nSaved {len(final_df)} samples to {output_file}")
        print(f"Total features per sample: {len(final_df.columns)}")
    else:
        print("\nNo valid samples found to process")
