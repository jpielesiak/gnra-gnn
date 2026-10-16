#!/bin/bash

# Array to collect CIF files that need processing
files_to_process=()

# Check each CIF file in mmcif_files directory
for cif_file in mmcif_files/*.cif; do
    # Extract PDB ID from filename (remove path and .cif extension)
    pdb_id=$(basename "$cif_file" .cif)
    
    # Check if corresponding FR3D output file exists
    fr3d_output="mmcif_files/fr3d-${pdb_id}-basepair_detail.txt"
    
    if [[ ! -f "$fr3d_output" ]]; then
        # Add to array if FR3D output doesn't exist
        files_to_process+=("$cif_file")
        echo "Will process: $cif_file (missing $fr3d_output)"
    else
        echo "Skipping: $cif_file (already has $fr3d_output)"
    fi
done

# Run cli2rest-bio only if there are files to process
if [[ ${#files_to_process[@]} -gt 0 ]]; then
    echo "Processing ${#files_to_process[@]} files with FR3D..."
    cli2rest-bio fr3d "${files_to_process[@]}"
else
    echo "All CIF files already have FR3D output. Nothing to process."
fi
