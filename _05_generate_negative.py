#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, write_cif
from rnapolis.tertiary_v2 import Residue, Structure


def load_gnra_motifs(
    filename: str = "gnra_motifs_by_pdb.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load GNRA motifs from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def load_structure_json(pdb_id: str) -> Dict[str, Any]:
    """Load structure JSON file for a PDB ID."""
    json_file = f"json_files/{pdb_id}.json"

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, "r") as f:
        return json.load(f)


def parse_and_process_mmcif_file(
    pdb_id: str, motifs: List[Dict[str, Any]]
) -> tuple[bool, Dict[str, Any]]:
    """Parse mmCIF file for a PDB ID and process its motifs."""
    mmcif_file = f"mmcif_files/{pdb_id}.cif"

    if not os.path.exists(mmcif_file):
        print(f"  Warning: {mmcif_file} not found")
        return False, {}

    try:
        print(f"Parsing {mmcif_file}...")
        with open(mmcif_file, "r") as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)
        print(f"  Successfully parsed {pdb_id}")

        # Process motifs to find their indices
        residues = [residue for residue in structure.residues if residue.is_nucleotide]
        motif_data, motif_chains = find_motif_residue_indices(residues, motifs)

        print(f"  Found {len(residues)} residues")
        print(f"  Processed {len(motifs)} motifs")
        print(f"  GNRA motifs found in chains: {sorted(motif_chains)}")

        # Get GNRA motif indices (extended to 8 residues)
        gnra_indices = set()
        for motif_dict in motif_data:
            gnra_indices.update(motif_dict["indices"])

        # Load and analyze structure JSON
        try:
            structure_data = load_structure_json(pdb_id)
            negative_regions = find_negative_regions(
                structure_data, gnra_indices, residues, motif_chains
            )

            print("  Found negative strands:")
            print(f"    Stem strands: {len(negative_regions['stems'])}")
            print(f"    Single strands: {len(negative_regions['single_strands'])}")
            print(f"    Hairpin strands: {len(negative_regions['hairpins'])}")
            print(f"    Loop strands: {len(negative_regions['loops'])}")

            # Add PDB ID to each region for tracking
            pdb_negative_regions = {"pdb_id": pdb_id, "regions": negative_regions}

            return True, pdb_negative_regions

        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            return False, {}

    except Exception as e:
        print(f"  Error parsing {pdb_id}: {e}")
        return False, {}


def find_motif_residue_indices(
    residues: List[Residue], motifs: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Set[str]]:
    """Find residue indices and residue objects for each motif's unit_ids, extending to 8 residues.

    Returns:
        tuple: (motif_data, motif_chains) where motif_chains is the set of chain IDs containing motifs
    """
    motif_data = []
    motif_chains = set()

    for motif_idx, motif in enumerate(motifs):
        indices: List[int] = []
        motif_residues = []
        unit_ids = motif.get("unit_ids", [])
        motif_key = motif.get("motif_key", f"motif_{motif_idx}")

        # Track chains for this motif
        motif_chain_ids = set()

        for unit_id_dict in unit_ids:
            chain_id = unit_id_dict.get("chain_id")
            motif_chain_ids.add(chain_id)

            # Find matching residue by comparing unit_id components
            for i, residue in enumerate(residues):
                unit_insertion_code = unit_id_dict.get("insertion_code", "")
                residue_insertion_code = residue.insertion_code or ""

                if (
                    residue.chain_id == chain_id
                    and residue.residue_number == unit_id_dict.get("residue_number")
                    and residue_insertion_code == unit_insertion_code
                ):
                    indices.append(i)
                    motif_residues.append(residue)
                    break

        # Log when we don't find exactly 6 indices
        if len(indices) != 6:
            print(
                f"    Warning: {motif_key} - Expected 6 residues, found {len(indices)}: {indices}"
            )
            continue  # Skip adding this motif to motif_data

        # Log when indices are not consecutive
        sorted_indices = sorted(indices)
        is_consecutive = all(
            sorted_indices[i] + 1 == sorted_indices[i + 1] for i in range(5)
        )
        if not is_consecutive:
            print(
                f"    Warning: {motif_key} - Residues are not consecutive: {sorted_indices}"
            )
            continue  # Skip adding this motif to motif_data

        # Extend to 8 residues (add 1 before and 1 after)
        min_idx = min(sorted_indices)
        max_idx = max(sorted_indices)

        # Check if we can add residues before and after
        if min_idx == 0 or max_idx == len(residues) - 1:
            print(
                f"    Warning: {motif_key} - Cannot extend to 8 residues (boundary constraints)"
            )
            continue  # Skip adding this motif to motif_data

        # Create extended indices and residues
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

        # Add chains to global set
        motif_chains.update(motif_chain_ids)

    return motif_data, motif_chains


def get_strand_residue_indices(
    strand: Dict[str, Any],
    residues: List[Residue],
    bpseq_index: Dict[str, Any],
    motif_chains: Set[str],
) -> List[int]:
    """Extract 0-based residue indices from a strand using bpseq_index mapping.

    Accepts strands that are either:
    1. 8+ nucleotides long, OR
    2. 6 nucleotides long that can be extended to 8 by adding one before and one after

    Only processes strands that are in chains containing GNRA motifs.
    Ensures all 8 nucleotides are consecutive and from the same chain.
    """
    strand_length = strand["last"] - strand["first"] + 1

    # Early check: if strand length is less than 6, skip processing
    if strand_length < 6:
        print(
            f"    DEBUG: Strand {strand['first']}-{strand['last']} too short ({strand_length} < 6), skipping"
        )
        return []

    print(
        f"    DEBUG: Strand range {strand['first']}-{strand['last']} (length {strand_length}), bpseq_index has {len(bpseq_index)} entries"
    )

    # Get base strand indices first
    base_indices: List[int] = []
    strand_chains = set()

    for pos in range(strand["first"], strand["last"] + 1):
        residue_obj = bpseq_index.get(str(pos))
        if residue_obj is not None:
            # Use auth values to find matching residue in structure
            auth_chain = residue_obj.get("auth", {}).get("chain", None)
            auth_number = residue_obj.get("auth", {}).get("number", None)
            auth_icode = residue_obj.get("auth", {}).get("icode", "") or ""

            strand_chains.add(auth_chain)

            # Find matching residue by comparing auth values
            found_match = False
            for i, residue in enumerate(residues):
                residue_insertion_code = residue.insertion_code or ""

                if (
                    residue.chain_id == auth_chain
                    and residue.residue_number == auth_number
                    and residue_insertion_code == auth_icode
                ):
                    base_indices.append(i)
                    found_match = True
                    break

            if not found_match:
                print(
                    f"    DEBUG: No match found for pos {pos}: auth_chain={auth_chain}, auth_number={auth_number}, auth_icode='{auth_icode}'"
                )
        else:
            print(f"    DEBUG: No bpseq_index entry for position {pos}")

    # Check if this strand is in a chain that contains GNRA motifs
    if not strand_chains.intersection(motif_chains):
        print(
            f"    DEBUG: Strand chains {strand_chains} don't overlap with motif chains {motif_chains}, skipping"
        )
        return []

    # Check if strand spans multiple chains
    if len(strand_chains) > 1:
        print(f"    DEBUG: Strand spans multiple chains {strand_chains}, skipping")
        return []

    print(f"    DEBUG: Found {len(base_indices)} base residue indices for strand")

    # Check if base indices are consecutive
    if len(base_indices) > 1:
        sorted_indices = sorted(base_indices)
        is_consecutive = all(
            sorted_indices[i] + 1 == sorted_indices[i + 1]
            for i in range(len(sorted_indices) - 1)
        )
        if not is_consecutive:
            print(
                f"    DEBUG: Base indices not consecutive: {sorted_indices}, skipping"
            )
            return []

        # Check if all residues are from the same chain
        chain_ids = [residues[i].chain_id for i in base_indices]
        if len(set(chain_ids)) > 1:
            print(
                f"    DEBUG: Base indices span multiple chains: {set(chain_ids)}, skipping"
            )
            return []

    # If we have 8+ residues, take exactly 8 consecutive ones
    if len(base_indices) >= 8:
        print(
            f"    DEBUG: Strand has {len(base_indices)} residues (>=8), taking first 8"
        )
        return base_indices[:8]

    # If we have exactly 6 residues, try to extend to 8
    if len(base_indices) == 6:
        print("    DEBUG: Strand has 6 residues, attempting to extend to 8")

        sorted_indices = sorted(base_indices)
        min_idx = min(sorted_indices)
        max_idx = max(sorted_indices)

        # Check boundary constraints
        if min_idx == 0 or max_idx == len(residues) - 1:
            print(
                f"    DEBUG: Cannot extend - boundary constraints (min_idx={min_idx}, max_idx={max_idx}, total_residues={len(residues)})"
            )
            return []

        # Check if extended residues are from the same chain
        before_residue = residues[min_idx - 1]
        after_residue = residues[max_idx + 1]
        base_chain = residues[base_indices[0]].chain_id

        if (
            before_residue.chain_id != base_chain
            or after_residue.chain_id != base_chain
        ):
            print(
                f"    DEBUG: Cannot extend - extended residues not from same chain (base: {base_chain}, before: {before_residue.chain_id}, after: {after_residue.chain_id})"
            )
            return []

        # Create extended indices
        extended_indices = [min_idx - 1] + sorted_indices + [max_idx + 1]
        print(f"    DEBUG: Extended 6-residue strand to 8 residues: {extended_indices}")
        return extended_indices

    # For strands with 7 residues or other lengths between 6-7, skip
    print(
        f"    DEBUG: Strand has {len(base_indices)} residues (not 6 or >=8), skipping"
    )
    return []


def indices_overlap(indices1: List[int], indices2: Set[int]) -> bool:
    """Check if any indices from indices1 overlap with indices2."""
    overlap_found = any(idx in indices2 for idx in indices1)
    overlapping_indices = [idx for idx in indices1 if idx in indices2]

    print(f"      DEBUG: Checking overlap between {indices1} and {sorted(indices2)}")
    print(f"      DEBUG: Overlapping indices: {overlapping_indices}")
    print(f"      DEBUG: Overlap result: {overlap_found}")

    return overlap_found


def find_negative_regions(
    structure_data: Dict[str, Any],
    gnra_indices: Set[int],
    residues: List[Residue],
    motif_chains: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Find individual strands with at least 8 nucleotides that don't overlap with GNRA motifs.

    Only considers structural elements in chains that contain GNRA motifs.
    """
    negative_regions: Dict[str, List[Dict[str, Any]]] = {
        "stems": [],
        "single_strands": [],
        "hairpins": [],
        "loops": [],
    }

    # Get the bpseq_index mapping from the structure data
    bpseq_index = structure_data.get("bpseq_index", {})
    print(f"  DEBUG: Structure bpseq_index has {len(bpseq_index)} entries")

    print(f"  DEBUG: GNRA indices: {sorted(gnra_indices)}")
    print(
        f"  DEBUG: Structure has {len(structure_data.get('stems', []))} stems, {len(structure_data.get('single_strands', []))} single_strands, {len(structure_data.get('hairpins', []))} hairpins, {len(structure_data.get('loops', []))} loops"
    )

    # Process stems - check each strand separately
    for i, stem in enumerate(structure_data.get("stems", [])):
        print(f"  DEBUG: Processing stem {i}")
        # Check strand5p
        if "strand5p" in stem:
            strand5p = stem["strand5p"]
            print(
                f"    DEBUG: Checking strand5p {strand5p['first']}-{strand5p['last']}"
            )
            strand_residue_indices = get_strand_residue_indices(
                strand5p, residues, bpseq_index, motif_chains
            )
            print(f"    DEBUG: strand5p residue indices: {strand_residue_indices}")
            has_overlap = indices_overlap(strand_residue_indices, gnra_indices)
            print(
                f"    DEBUG: strand5p has {len(strand_residue_indices)} residues, overlap with GNRA: {has_overlap}"
            )
            if len(strand_residue_indices) == 8 and not has_overlap:
                negative_regions["stems"].append(
                    {
                        "region": strand5p,
                        "indices": strand_residue_indices,
                        "type": "stem_5p",
                    }
                )
                print("    DEBUG: Added stem_5p to negative regions")

        # Check strand3p
        if "strand3p" in stem:
            strand3p = stem["strand3p"]
            print(
                f"    DEBUG: Checking strand3p {strand3p['first']}-{strand3p['last']}"
            )
            strand_residue_indices = get_strand_residue_indices(
                strand3p, residues, bpseq_index, motif_chains
            )
            print(f"    DEBUG: strand3p residue indices: {strand_residue_indices}")
            has_overlap = indices_overlap(strand_residue_indices, gnra_indices)
            print(
                f"    DEBUG: strand3p has {len(strand_residue_indices)} residues, overlap with GNRA: {has_overlap}"
            )
            if len(strand_residue_indices) == 8 and not has_overlap:
                negative_regions["stems"].append(
                    {
                        "region": strand3p,
                        "indices": strand_residue_indices,
                        "type": "stem_3p",
                    }
                )
                print("    DEBUG: Added stem_3p to negative regions")

    # Process single strands
    for i, single_strand in enumerate(structure_data.get("single_strands", [])):
        print(f"  DEBUG: Processing single_strand {i}")
        if "strand" in single_strand:
            strand = single_strand["strand"]
            print(
                f"    DEBUG: Checking single strand {strand['first']}-{strand['last']}"
            )
            strand_residue_indices = get_strand_residue_indices(
                strand, residues, bpseq_index, motif_chains
            )
            print(f"    DEBUG: single strand residue indices: {strand_residue_indices}")
            has_overlap = indices_overlap(strand_residue_indices, gnra_indices)
            print(
                f"    DEBUG: single strand has {len(strand_residue_indices)} residues, overlap with GNRA: {has_overlap}"
            )
            if len(strand_residue_indices) == 8 and not has_overlap:
                negative_regions["single_strands"].append(
                    {
                        "region": strand,
                        "indices": strand_residue_indices,
                        "type": "single_strand",
                    }
                )
                print("    DEBUG: Added single_strand to negative regions")

    # Process hairpins
    for i, hairpin in enumerate(structure_data.get("hairpins", [])):
        print(f"  DEBUG: Processing hairpin {i}")
        if "strand" in hairpin:
            strand = hairpin["strand"]
            print(
                f"    DEBUG: Checking hairpin strand {strand['first']}-{strand['last']}"
            )
            strand_residue_indices = get_strand_residue_indices(
                strand, residues, bpseq_index, motif_chains
            )
            print(
                f"    DEBUG: hairpin strand residue indices: {strand_residue_indices}"
            )
            has_overlap = indices_overlap(strand_residue_indices, gnra_indices)
            print(
                f"    DEBUG: hairpin strand has {len(strand_residue_indices)} residues, overlap with GNRA: {has_overlap}"
            )
            if len(strand_residue_indices) == 8 and not has_overlap:
                negative_regions["hairpins"].append(
                    {
                        "region": strand,
                        "indices": strand_residue_indices,
                        "type": "hairpin",
                    }
                )
                print("    DEBUG: Added hairpin to negative regions")

    # Process loops - check each strand separately
    for i, loop in enumerate(structure_data.get("loops", [])):
        print(f"  DEBUG: Processing loop {i}")
        if "strands" in loop:
            for j, strand in enumerate(loop["strands"]):
                print(
                    f"    DEBUG: Checking loop strand {j}: {strand['first']}-{strand['last']}"
                )
                strand_residue_indices = get_strand_residue_indices(
                    strand, residues, bpseq_index, motif_chains
                )
                print(
                    f"    DEBUG: loop strand residue indices: {strand_residue_indices}"
                )
                has_overlap = indices_overlap(strand_residue_indices, gnra_indices)
                print(
                    f"    DEBUG: loop strand has {len(strand_residue_indices)} residues, overlap with GNRA: {has_overlap}"
                )
                if len(strand_residue_indices) == 8 and not has_overlap:
                    negative_regions["loops"].append(
                        {
                            "region": strand,
                            "indices": strand_residue_indices,
                            "type": "loop",
                        }
                    )
                    print("    DEBUG: Added loop strand to negative regions")

    return negative_regions


def process_pdb_wrapper(args):
    """Wrapper function for parallel processing."""
    pdb_id, motifs = args
    success, negative_regions = parse_and_process_mmcif_file(pdb_id, motifs)
    return pdb_id, success, negative_regions


def process_all_pdb_files(
    gnra_motifs: Dict[str, List[Dict[str, Any]]], max_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Process all PDB files and their motifs in parallel."""
    successful_count = 0
    failed_count = 0
    all_negative_regions = []

    # Determine number of workers (default to number of CPU cores)
    if max_workers is None:
        max_workers = os.cpu_count()

    print(f"Processing {len(gnra_motifs)} PDB files using {max_workers} workers...")
    print("DEBUG: Limiting to first 5 files for debugging...")

    # Prepare arguments for parallel processing
    pdb_args = list(gnra_motifs.items())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdb = {
            executor.submit(process_pdb_wrapper, args): args[0] for args in pdb_args
        }

        # Process completed tasks
        for future in as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            try:
                _, success, negative_regions = future.result()
                if success:
                    successful_count += 1
                    if negative_regions:
                        all_negative_regions.append(negative_regions)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"  Error processing {pdb_id}: {e}")
                failed_count += 1

    print("\nProcessing complete:")
    print(f"  Successfully processed: {successful_count} PDB files")
    print(f"  Failed to process: {failed_count} PDB files")

    return all_negative_regions


def extract_and_save_negative_region(
    pdb_id: str,
    region_data: Dict[str, Any],
    region_type: str,
    counter: int,
    residues: List[Residue],
) -> bool:
    """Extract residues for a negative region and save as CIF file."""
    indices = region_data["indices"]

    # Create output directory
    output_dir = "negative_cif_files"
    os.makedirs(output_dir, exist_ok=True)

    # Create filename: type_pdbid_counter.cif
    output_file = os.path.join(output_dir, f"{region_type}_{pdb_id}_{counter:04d}.cif")

    # Check if file already exists
    if os.path.exists(output_file):
        print(f"    File {output_file} already exists, skipping")
        return False

    try:
        # Get residues for the indices
        region_residues = [residues[i] for i in indices]

        # Get atoms for the residues
        atoms_df = pd.concat(residue.atoms for residue in region_residues)

        # Write to CIF file
        with open(output_file, "w") as f:
            write_cif(atoms_df, f)

        print(f"    Saved {region_type} region to {output_file}")
        return True

    except Exception as e:
        print(f"    Error saving {region_type} region: {e}")
        return False


def extract_all_negative_regions(negative_regions: List[Dict[str, Any]]) -> None:
    """Extract all negative regions as CIF files."""
    print("\nExtracting negative regions as CIF files...")

    total_extracted = 0
    total_skipped = 0

    for pdb_data in negative_regions:
        pdb_id = pdb_data["pdb_id"]
        regions = pdb_data["regions"]

        print(f"Processing negative regions for {pdb_id}...")

        # Load the structure to get residues
        try:
            mmcif_file = f"mmcif_files/{pdb_id}.cif"
            with open(mmcif_file, "r") as f:
                atoms_df = parse_cif_atoms(f)
            structure = Structure(atoms_df)
            residues = [
                residue for residue in structure.residues if residue.is_nucleotide
            ]

            # Process each type of region
            for region_type, region_list in regions.items():
                for i, region_data in enumerate(region_list):
                    success = extract_and_save_negative_region(
                        pdb_id, region_data, region_type, i, residues
                    )
                    if success:
                        total_extracted += 1
                    else:
                        total_skipped += 1

        except Exception as e:
            print(f"  Error processing {pdb_id}: {e}")

    print("\nExtraction complete:")
    print(f"  Total extracted: {total_extracted} negative regions")
    print(f"  Total skipped: {total_skipped} negative regions")


def save_negative_regions(
    negative_regions: List[Dict[str, Any]], filename: str = "negative_regions.json"
) -> None:
    """Save all negative regions to a JSON file."""
    with open(filename, "w") as f:
        json.dump(negative_regions, f, indent=2)
    print(
        f"Saved {len(negative_regions)} PDB structures with negative regions to {filename}"
    )


def load_negative_regions(
    filename: str = "negative_regions.json",
) -> List[Dict[str, Any]]:
    """Load negative regions from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def check_negative_regions_exist(filename: str = "negative_regions.json") -> bool:
    """Check if negative regions file already exists."""
    return os.path.exists(filename)


def print_negative_regions_summary(negative_regions: List[Dict[str, Any]]) -> None:
    """Print summary statistics of negative strands."""
    total_stems = 0
    total_single_strands = 0
    total_hairpins = 0
    total_loops = 0

    for pdb_data in negative_regions:
        regions = pdb_data.get("regions", {})
        total_stems += len(regions.get("stems", []))
        total_single_strands += len(regions.get("single_strands", []))
        total_hairpins += len(regions.get("hairpins", []))
        total_loops += len(regions.get("loops", []))

    print("\nNegative strands summary:")
    print(f"  Total PDB structures: {len(negative_regions)}")
    print(f"  Total stem strands: {total_stems}")
    print(f"  Total single strands: {total_single_strands}")
    print(f"  Total hairpin strands: {total_hairpins}")
    print(f"  Total loop strands: {total_loops}")
    print(
        f"  Total negative strands: {total_stems + total_single_strands + total_hairpins + total_loops}"
    )


def main():
    """Main function to find negative regions for GNRA motifs."""
    negative_regions_file = "negative_regions.json"

    # Check if negative regions file already exists
    if check_negative_regions_exist(negative_regions_file):
        print(f"Negative regions file '{negative_regions_file}' already exists.")
        print("Loading existing negative regions...")
        negative_regions = load_negative_regions(negative_regions_file)

        # Extract negative regions as CIF files
        extract_all_negative_regions(negative_regions)

        print_negative_regions_summary(negative_regions)
        print("\nTo regenerate, delete the file and run the script again.")
        return

    gnra_motifs = load_gnra_motifs()

    print(f"Loaded GNRA motifs for {len(gnra_motifs)} PDB structures")

    # Print summary information
    total_motifs = sum(len(motifs) for motifs in gnra_motifs.values())
    print(f"Total number of GNRA motifs: {total_motifs}")

    print("\nAnalyzing structures for negative strands...")
    negative_regions = process_all_pdb_files(gnra_motifs)

    # Save negative regions to file
    save_negative_regions(negative_regions, negative_regions_file)

    # Extract negative regions as CIF files
    extract_all_negative_regions(negative_regions)

    # Print summary
    print_negative_regions_summary(negative_regions)


if __name__ == "__main__":
    main()
