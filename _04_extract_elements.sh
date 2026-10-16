#!/bin/bash

mkdir -p json_files

# Array to collect CIF files that need processing
files_to_process=()

# Check each CIF file in mmcif_files directory
for cif_file in mmcif_files/*.cif; do
    # Extract PDB ID from filename (remove path and .cif extension)
    pdb_id=$(basename "$cif_file" .cif)

    # Check if corresponding JSON output file exists
    json_output="json_files/${pdb_id}.json"
    
    if [[ ! -f "$json_output" ]]; then
        # Add to array if FR3D output doesn't exist
        files_to_process+=("$cif_file")
        echo "Will process: $cif_file (missing $json_output)"
    else
        echo "Skipping: $cif_file (already has $json_output)"
    fi
done

# Run cli2rest-bio only if there are files to process
if [[ ${#files_to_process[@]} -gt 0 ]]; then
    echo "Processing ${#files_to_process[@]} files with rnapolis adapter..."
    for cif_file in "${files_to_process[@]}"; do
        # Extract PDB ID from filename (remove path and .cif extension)
        pdb_id=$(basename "$cif_file" .cif)
        fr3d_output="mmcif_files/fr3d-${pdb_id}-basepair_detail.txt"    
        echo adapter "$cif_file" --external "$fr3d_output" --tool fr3d --json "json_files/${pdb_id}.json"
    done | parallel --bar
else
    echo "All CIF files already have JSON output. Nothing to process."
fi
