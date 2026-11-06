#! /usr/bin/env python3
"""Explain, for ONE structure, why a known GNRA motif is (not) matched by any window.

Prints:
  1. the chains / residue ids the runner sees,
  2. every known motif from the answers JSON (size, residues, which ones are absent),
  3. every window that overlaps a motif residue, with its probability,
  4. the top-scoring windows overall.

Example:
    uv run debug_gnra_windows.py -structure cif_files/8hzl.cif.gz \
        -answers gnra_motifs_by_pdb.json \
        -model approximate-hierarchical-gat_v10.pkl \
        -evaluator _14_test_all_window_models.py -runner 05-run-inference.py
"""

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace


def load_module(path: Path):
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def fmt(key) -> str:
    chain, number, icode = key
    return f"{chain}:{number}{icode}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-structure", type=Path, required=True)
    p.add_argument("-answers", type=Path, required=True)
    p.add_argument("-model", type=Path, required=True)
    p.add_argument("-evaluator", type=Path, default=Path("_14_test_all_window_models.py"))
    p.add_argument("-runner", type=Path, default=Path("05-run-inference.py"))
    p.add_argument("-top", type=int, default=5)
    args = p.parse_args()

    ev = load_module(args.evaluator)
    ev.allow_cuda_pickles_on_cpu()
    runner = ev.import_runner(args.runner)

    # ── 1. What the runner sees in the structure ─────────────────────────────
    structure = ev.load_structure(runner, args.structure, None)
    by_chain = runner.get_nucleotide_residues_by_chain(structure)
    present = {}
    print("\n=== Nucleotide chains (as the runner sees them) ===")
    for chain, residues in by_chain.items():
        keys = [ev.residue_key(runner.clean_text(r.chain_id), r.residue_number, r.insertion_code)
                for r in residues]
        for k, r in zip(keys, residues):
            present[k] = r
        seq = "".join(r.one_letter_name for r in residues)
        print(f"  chain {chain!r}: {len(residues)} nt, {fmt(keys[0])} .. {fmt(keys[-1])}")
        print(f"    {seq}")

    # ── 2. Known motifs from the answers file ────────────────────────────────
    answers = ev.load_answers(args.answers)
    name = ev.structure_name(args.structure)
    known = ev.lookup_answers(name, answers) or set()
    print(f"\n=== Known motifs for {name}: {len(known)} ===")
    for motif in known:
        ordered = sorted(motif, key=lambda k: (k[0], k[1], k[2]))
        missing = [k for k in ordered if k not in present]
        seq = "".join(present[k].one_letter_name if k in present else "?" for k in ordered)
        print(f"  size {len(motif)}  seq {seq}  residues {', '.join(map(fmt, ordered))}")
        if missing:
            print(f"    NOT FOUND in structure: {', '.join(map(fmt, missing))}"
                  "  <-- chain id / numbering convention mismatch?")

    # ── 3. Score windows ─────────────────────────────────────────────────────
    with open(args.model, "rb") as handle:
        window_size = int(pickle.load(handle)["window_size"])
    bundle = runner.load_model_bundle(args.model, window_size)
    windows = ev.build_windows(runner, args.structure, structure, window_size)
    if windows is None:
        raise SystemExit("No windows generated.")
    preds, probs = ev.score(runner, SimpleNamespace(bundle=bundle), windows, 0.5)

    motif_residues = set().union(*known) if known else set()
    print(f"\n=== Windows overlapping a motif residue (window size {window_size}) ===")
    for i, key in enumerate(windows.keys):
        overlap = len(key & motif_residues)
        if overlap:
            best = max((len(key & m) / len(m), len(m)) for m in known)
            exact = any(key == m for m in known)
            contains = any(m <= key for m in known)
            print(f"  win {windows.meta['window_index'].iloc[i]:>4}  "
                  f"{windows.meta['sequence'].iloc[i]}  p={probs[i]:.3f}  "
                  f"overlap={overlap}  best_frac={best[0]:.2f}  "
                  f"exact={exact}  contains={contains}  [{windows.meta['residue_ids'].iloc[i]}]")

    print(f"\n=== Top {args.top} windows by probability ===")
    for i in probs.argsort()[::-1][: args.top]:
        print(f"  win {windows.meta['window_index'].iloc[i]:>4}  "
              f"{windows.meta['sequence'].iloc[i]}  p={probs[i]:.3f}  "
              f"[{windows.meta['residue_ids'].iloc[i]}]")


if __name__ == "__main__":
    main()