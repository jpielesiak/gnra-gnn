# Array to collect CIF files that need processing
$filesToProcess = @()

# Check each CIF file in mmcif_files directory
Get-ChildItem -Path "mmcif_files\*.cif" | ForEach-Object {
    $cifFile = $_.FullName
    # Extract PDB ID from filename (remove path and .cif extension)
    $pdbId = $_.BaseName

    # Check if corresponding FR3D output file exists
    $fr3dOutput = "mmcif_files\fr3d-${pdbId}-basepair_detail.txt"

    if (-not (Test-Path $fr3dOutput)) {
        # Add to array if FR3D output doesn't exist
        $filesToProcess += $cifFile
        Write-Host "Will process: $cifFile (missing $fr3dOutput)"
    } else {
        Write-Host "Skipping: $cifFile (already has $fr3dOutput)"
    }
}

# Run cli2rest-bio only if there are files to process
if ($filesToProcess.Count -gt 0) {
    Write-Host "Processing $($filesToProcess.Count) files with FR3D..."
    cli2rest-bio fr3d @filesToProcess
} else {
    Write-Host "All CIF files already have FR3D output. Nothing to process."
}