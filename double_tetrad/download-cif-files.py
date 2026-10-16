import json
import time
import requests
from pathlib import Path


with open("double_tetrad_motifs_by_pdb.json") as f:
    data = json.load(f)

pdb_ids = sorted({key.split("-assembly")[0].lower() for key in data})
print(len(pdb_ids), "distinct PDB entries")

with open("pdb_ids.txt", "w") as f:
    f.write("\n".join(pdb_ids))


out_dir = Path("mmcif")
out_dir.mkdir(exist_ok=True)

with open("pdb_ids.txt") as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

for pdb_id in pdb_ids:
    dest = out_dir / f"{pdb_id}.cif"
    if dest.exists():
        continue
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        dest.write_bytes(r.content)
        print("OK", pdb_id)
    else:
        print("FAILED", pdb_id, r.status_code)
    time.sleep(0.2)  # be gentle on the server