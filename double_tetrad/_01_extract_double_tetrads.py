#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 01 — extract consecutive G4 tetrad pairs from ElTetrado JSON.

Reads every ``<pdb>-assembly<N>.json`` produced by ElTetrado from
``ONQUADRO_JSON_DIR``, finds **pairs of consecutive G4 tetrads** within each
annotated quadruplex, and writes two artefacts:

* ``double_tetrad_motifs_by_pdb.json`` — ``"<pdb>-assembly<N>" -> [motif]``
  where each motif has 8 ``unit_ids`` (4 from the lower tetrad + 4 from the
  upper tetrad).  A pair is kept only when *both* tetrads are G4 (all four
  nucleotides are guanines — RNA ``G`` or DNA ``DG``, detected via
  ``shortName == "G"``).
* ``double_tetrad_exclusion_sets.json`` — per file, the set of 8-tuples
  (``chain, number, icode``) belonging to annotated double tetrads, for use
  by step 03's tuple-level exclusion.

Blacklisted PDB entries (``blacklist.txt``) are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_JSON_DIR = "/mnt/data-ssd/tzok/onquadro-main/json"
DEFAULT_BLACKLIST = "/mnt/data-ssd/tzok/onquadro-main/blacklist.txt"
TETRAD_SIZE = 4
DOUBLE_TETRAD_SIZE = 8


def load_blacklist(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip().lower() for line in f if line.strip()}


def is_g4_tetrad(tetrad: Dict[str, Any], short_name_map: Dict[str, str]) -> bool:
    for key in ("nt1", "nt2", "nt3", "nt4"):
        full_name = tetrad[key]
        short = short_name_map.get(full_name)
        if short is None:
            short = next(
                (v for k, v in short_name_map.items() if k.lower() == full_name.lower()),
                None,
            )
        if short is None or short.upper() != "G":
            return False
    return True


def build_full_name_maps(nucleotides: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    short_map: Dict[str, str] = {}
    detail_map: Dict[str, Dict[str, Any]] = {}
    for nt in nucleotides:
        fn = nt.get("fullName")
        if fn is None:
            continue
        short_map[fn] = nt.get("shortName", "")
        detail_map[fn] = nt
    return short_map, detail_map


def residue_ref(full_name: str, detail_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    nt = detail_map.get(full_name)
    if nt is None:
        nt = next(
            (v for k, v in detail_map.items() if k.lower() == full_name.lower()),
            None,
        )
    if nt is None:
        return None
    return {
        "chain_id": nt.get("chain"),
        "residue_number": nt.get("number"),
        "insertion_code": nt.get("icode") or "",
        "full_name": full_name,
    }


def extract_double_tetrads_from_file(
    json_path: str, short_map: Dict[str, str], detail_map: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int, str]]]:
    with open(json_path) as f:
        data = json.load(f)

    motifs: List[Dict[str, Any]] = []
    exclusion: List[Tuple[str, int, str]] = []
    motif_index = 0

    for helix in data.get("helices", []):
        for quad in helix.get("quadruplexes", []):
            tetrads = quad.get("tetrads", [])
            # G4 flags per tetrad
            g4_flags = [is_g4_tetrad(t, short_map) for t in tetrads]
            # Consecutive G4 pairs
            for i in range(len(g4_flags) - 1):
                if not (g4_flags[i] and g4_flags[i + 1]):
                    continue
                t_lo = tetrads[i]
                t_hi = tetrads[i + 1]
                unit_ids: List[Dict[str, Any]] = []
                for key in ("nt1", "nt2", "nt3", "nt4"):
                    ref = residue_ref(t_lo[key], detail_map)
                    if ref is not None:
                        unit_ids.append(ref)
                for key in ("nt1", "nt2", "nt3", "nt4"):
                    ref = residue_ref(t_hi[key], detail_map)
                    if ref is not None:
                        unit_ids.append(ref)
                if len(unit_ids) != DOUBLE_TETRAD_SIZE:
                    continue
                file_stem = Path(json_path).stem
                motif_key = f"DT_{file_stem}_{motif_index:03d}"
                motifs.append(
                    {
                        "motif_key": motif_key,
                        "unit_ids": unit_ids,
                        "lower_tetrad": {
                            "onz": t_lo.get("onz"),
                            "gba": t_lo.get("gbaClassification"),
                            "id": t_lo.get("id"),
                        },
                        "upper_tetrad": {
                            "onz": t_hi.get("onz"),
                            "gba": t_hi.get("gbaClassification"),
                            "id": t_hi.get("id"),
                        },
                        "quadruplex_handedness": quad.get("handedness"),
                    }
                )
                for u in unit_ids:
                    exclusion.append((u["chain_id"], u["residue_number"], u["insertion_code"]))
                motif_index += 1

    return motifs, exclusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    parser.add_argument("--blacklist", default=DEFAULT_BLACKLIST)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent),
    )
    args = parser.parse_args()

    blacklist = load_blacklist(args.blacklist)
    json_files = sorted(f for f in os.listdir(args.json_dir) if f.endswith(".json"))
    print(f"Found {len(json_files)} JSON files in {args.json_dir}")
    print(f"Blacklist: {len(blacklist)} PDB entries")

    motifs_by_pdb: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    exclusion_by_pdb: Dict[str, List[List[Any]]] = {}

    total_pairs = 0
    files_with_pairs = 0

    for fn in json_files:
        pdb_id = fn.split("-")[0].lower()
        if pdb_id in blacklist:
            continue
        json_path = os.path.join(args.json_dir, fn)
        with open(json_path) as f:
            data = json.load(f)
        short_map, detail_map = build_full_name_maps(data.get("nucleotides", []))
        if not short_map:
            continue
        stem = fn.replace(".json", "")
        motifs, exclusion = extract_double_tetrads_from_file(json_path, short_map, detail_map)
        if motifs:
            motifs_by_pdb[stem] = motifs
            exclusion_by_pdb[stem] = [list(e) for e in exclusion]
            total_pairs += len(motifs)
            files_with_pairs += 1

    output_dir = Path(args.output_dir)
    motifs_path = output_dir / "double_tetrad_motifs_by_pdb.json"
    exclusion_path = output_dir / "double_tetrad_exclusion_sets.json"

    with open(motifs_path, "w") as f:
        json.dump(dict(motifs_by_pdb), f, indent=2)
    with open(exclusion_path, "w") as f:
        json.dump(exclusion_by_pdb, f, indent=2)

    print(f"\nExtracted {total_pairs} consecutive G4 tetrad pairs from {files_with_pairs} files")
    print(f"Distinct PDB entries: {len({k.split('-')[0] for k in motifs_by_pdb})}")
    print(f"Saved motifs    -> {motifs_path}")
    print(f"Saved exclusion -> {exclusion_path}")

    counts = [len(v) for v in motifs_by_pdb.values()]
    from collections import Counter

    dist = Counter(counts)
    print(f"Pairs-per-file distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
