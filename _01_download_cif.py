#!/usr/bin/env python3
"""
Script to parse hl_3.97.json and find the GNRA motif with motif_id HL_37824.7
"""

import gzip
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Dict, Set


@dataclass
class UnitID:
    """Represents a Unit ID with all its components"""

    pdb_id: str  # 4 characters, case-insensitive
    model_number: int  # integer, range 1-99
    chain_id: str  # string, case-sensitive
    residue_id: str  # 1-3 characters, case-insensitive
    residue_number: int  # integer, range: -999..9999
    atom_name: Optional[str] = (
        ""  # 0-4 characters, case-insensitive, blank means all atoms
    )
    alternate_id: Optional[str] = ""  # One of ['A', 'B', 'C', '0'], case-insensitive
    insertion_code: Optional[str] = ""  # 1 character, case-insensitive
    symmetry_operation: Optional[str] = "1_555"  # 5-6 characters, case-insensitive


def parse_unit_id(unit_id_str: str) -> UnitID:
    """Parse a Unit ID string into a UnitID object"""
    parts = unit_id_str.split("|")

    if len(parts) < 5:
        raise ValueError(f"Invalid Unit ID format: {unit_id_str}")

    return UnitID(
        pdb_id=parts[0],
        model_number=int(parts[1]),
        chain_id=parts[2],
        residue_id=parts[3],
        residue_number=int(parts[4]),
        atom_name=parts[5] if len(parts) > 5 else "",
        alternate_id=parts[6] if len(parts) > 6 else "",
        insertion_code=parts[7] if len(parts) > 7 else "",
        symmetry_operation=parts[8] if len(parts) > 8 else "1_555",
    )


def process_alignment(alignment: Dict) -> Dict[str, List[UnitID]]:
    """Process the alignment field and convert Unit ID strings to UnitID objects"""
    processed_alignment = {}

    for key, unit_id_list in alignment.items():
        if not isinstance(unit_id_list, list) or len(unit_id_list) != 6:
            print(
                f"Warning: Expected list of 6 elements for key '{key}', got {len(unit_id_list) if isinstance(unit_id_list, list) else 'non-list'}"
            )
            continue

        processed_units = []
        for unit_id_str in unit_id_list:
            try:
                unit_id = parse_unit_id(unit_id_str)
                processed_units.append(unit_id)
            except (ValueError, IndexError) as e:
                print(f"Error parsing Unit ID '{unit_id_str}': {e}")
                continue

        processed_alignment[key] = processed_units

    return processed_alignment


def unit_id_to_dict(unit_id: UnitID) -> Dict:
    """Convert a UnitID object to a dictionary for JSON serialization"""
    return {
        "pdb_id": unit_id.pdb_id,
        "model_number": unit_id.model_number,
        "chain_id": unit_id.chain_id,
        "residue_id": unit_id.residue_id,
        "residue_number": unit_id.residue_number,
        "atom_name": unit_id.atom_name,
        "alternate_id": unit_id.alternate_id,
        "insertion_code": unit_id.insertion_code,
        "symmetry_operation": unit_id.symmetry_operation,
    }


def create_gnra_motifs_by_pdb(
    processed_alignment: Dict[str, List[UnitID]],
) -> Dict[str, List[Dict]]:
    """Create a dictionary of GNRA motifs organized by PDB ID"""
    gnra_by_pdb = {}

    for motif_key, unit_ids in processed_alignment.items():
        # Group unit IDs by PDB ID for this motif
        pdb_groups = {}
        for unit_id in unit_ids:
            pdb_id = unit_id.pdb_id.lower()
            if pdb_id not in pdb_groups:
                pdb_groups[pdb_id] = []
            pdb_groups[pdb_id].append(unit_id_to_dict(unit_id))

        # Add motif entries to each PDB ID
        for pdb_id, unit_id_dicts in pdb_groups.items():
            if pdb_id not in gnra_by_pdb:
                gnra_by_pdb[pdb_id] = []

            motif_entry = {"motif_key": motif_key, "unit_ids": unit_id_dicts}

            gnra_by_pdb[pdb_id].append(motif_entry)

    return gnra_by_pdb


def save_gnra_motifs_json(
    gnra_by_pdb: Dict[str, List[Dict]], filename: str = "gnra_motifs_by_pdb.json"
) -> None:
    """Save the GNRA motifs organized by PDB ID to a JSON file"""
    try:
        with open(filename, "w") as f:
            json.dump(gnra_by_pdb, f, indent=2)
        print(f"\nSaved GNRA motifs data to {filename}")
    except Exception as e:
        print(f"Error saving JSON file {filename}: {e}")


def extract_unique_pdb_ids(processed_alignment: Dict[str, List[UnitID]]) -> Set[str]:
    """Extract unique PDB IDs from the processed alignment data"""
    pdb_ids = set()

    for unit_ids in processed_alignment.values():
        for unit_id in unit_ids:
            pdb_ids.add(unit_id.pdb_id.lower())  # Store as lowercase for consistency

    return pdb_ids


def download_mmcif_file(pdb_id: str) -> bool:
    """Download mmCIF file for a given PDB ID if it doesn't already exist"""
    # Create subdirectory for mmCIF files
    mmcif_dir = "mmcif_files"
    os.makedirs(mmcif_dir, exist_ok=True)

    filename = f"{pdb_id.lower()}.cif"
    filepath = os.path.join(mmcif_dir, filename)

    # Check if file already exists
    if os.path.exists(filepath):
        print(f"File {filepath} already exists, skipping download")
        return True

    # Construct URL with uppercase PDB ID
    url = f"http://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
    temp_gz_filepath = filepath + ".gz"

    try:
        print(f"Downloading {temp_gz_filepath} from {url}")
        urllib.request.urlretrieve(url, temp_gz_filepath)

        # Ungzip the file
        print(f"Uncompressing {temp_gz_filepath} to {filepath}")
        with gzip.open(temp_gz_filepath, "rb") as f_in:
            with open(filepath, "wb") as f_out:
                f_out.write(f_in.read())

        # Remove the temporary .gz file
        os.remove(temp_gz_filepath)

        print(f"Successfully downloaded and uncompressed {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading {filepath}: {e}")
        # Clean up temporary file if it exists
        if os.path.exists(temp_gz_filepath):
            os.remove(temp_gz_filepath)
        return False


def download_all_mmcif_files(pdb_ids: Set[str]) -> None:
    """Download mmCIF files for all unique PDB IDs"""
    print(f"\nDownloading mmCIF files for {len(pdb_ids)} unique PDB IDs...")

    successful_downloads = 0
    for pdb_id in sorted(pdb_ids):
        if download_mmcif_file(pdb_id):
            successful_downloads += 1

    print(
        f"\nDownload summary: {successful_downloads}/{len(pdb_ids)} files downloaded successfully"
    )


def find_gnra_motif():
    """Parse hl_3.97.json and find the object with motif_id HL_37824.7"""

    try:
        # Load the JSON file
        with open("hl_3.97.json", "r") as f:
            data = json.load(f)

        # Search for the object with the specified motif_id
        for obj in data:
            if obj.get("motif_id") == "HL_37824.7":
                print(f"Found GNRA motif with motif_id: {obj['motif_id']}")
                print(json.dumps(obj, indent=2))

                # Process the alignment field
                if "alignment" in obj:
                    print("\nProcessing alignment field...")
                    processed_alignment = process_alignment(obj["alignment"])

                    print(f"\nProcessed {len(processed_alignment)} alignment entries:")
                    for key, unit_ids in processed_alignment.items():
                        print(f"\n{key}:")
                        for i, unit_id in enumerate(unit_ids):
                            print(f"  {i + 1}. {unit_id}")

                    # Extract unique PDB IDs and download mmCIF files
                    unique_pdb_ids = extract_unique_pdb_ids(processed_alignment)
                    print(
                        f"\nFound {len(unique_pdb_ids)} unique PDB IDs: {sorted(unique_pdb_ids)}"
                    )

                    download_all_mmcif_files(unique_pdb_ids)

                    # Create and save GNRA motifs organized by PDB ID
                    gnra_by_pdb = create_gnra_motifs_by_pdb(processed_alignment)
                    print(f"\nOrganized GNRA motifs by {len(gnra_by_pdb)} PDB IDs")
                    save_gnra_motifs_json(gnra_by_pdb)

                return obj

        print("No object found with motif_id HL_37824.7")
        return None

    except FileNotFoundError:
        print("Error: hl_3.97.json file not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in hl_3.97.json")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    find_gnra_motif()
