#! /usr/bin/env python3
"""Break down the false positives in the CSVs written by _13_test_all_models.py.

Every predicted-positive candidate that is not an exact known motif is put in a bucket
by how many of its 8 residues belong to the best-overlapping known motif:

  near-miss (6-7/8)  : almost the true motif, differs by 1-2 guanines
  partial  (1-5/8)   : touches a known motif, e.g. tetrad 1 + tetrad 3 of a 3-tetrad stack
  unrelated (0/8)    : no shared residue with any annotated motif
                       (an unannotated quadruplex / copy, or a real model error)

It then applies non-maximum suppression (keep the best candidate, drop the
ones overlapping it, repeat) and reports TP / FP / FN on what is left -- the
numbers a user of the detector would actually see.

It also groups the positives of each structure into "sites" (positives that share
residues) and reports, per known motif, whether the top-scoring candidate of its
site is the motif itself.

Example:
    uv run analyze_false_positives.py -results models_test/ -answers double_tetrad_motifs_by_pdb.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

_RES = re.compile(r"^(.*):(-?\d+)(.*)$")


def parse_ids(text: str) -> frozenset:
    keys = []
    for item in text.split(","):
        m = _RES.match(item.strip())
        chain, num, icode = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
        keys.append((chain, num, "" if icode in {"?", "."} else icode))
    return frozenset(keys)


def load_answers(path: Path) -> dict:
    data = json.loads(path.read_text())
    out = {}
    for struct, motifs in data.items():
        keys = set()
        for motif in motifs or []:
            keys.add(frozenset(
                (str(u["chain_id"]).strip(), int(u["residue_number"]),
                 "" if str(u.get("insertion_code") or "").strip() in {"", "?", "."}
                 else str(u.get("insertion_code")).strip())
                for u in motif["unit_ids"]
            ))
        out[struct.lower()] = keys
    return out


def sites(keys: list) -> list[list[int]]:
    """Connected components of candidates that share at least one residue."""
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, k in enumerate(keys):
        for r in k:
            if r in owner:
                parent[find(i)] = find(owner[r])
            else:
                owner[r] = i
    groups = {}
    for i in range(len(keys)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def nms(keys: list, probs: list, threshold: float, max_shared: int) -> list[int]:
    """Greedy non-maximum suppression: take candidates from the highest
    probability down; drop any candidate sharing more than ``max_shared``
    residues with one already taken. max_shared=4 keeps two double tetrads
    that share one tetrad (1+2 and 2+3 of a three-tetrad stack)."""
    order = sorted(range(len(keys)), key=lambda i: -probs[i])
    kept: list[int] = []
    for i in order:
        if probs[i] < threshold:
            break
        if all(len(keys[i] & keys[j]) <= max_shared for j in kept):
            kept.append(i)
    return kept


def analyze(csv_path: Path, answers: dict, threshold: float = 0.5, max_shared: int = 4) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"\n{csv_path.name}: empty")
        return
    df["prediction"] = df["prediction"].astype(str).str.lower() == "true"
    print(f"\n=== {csv_path.stem} ===")
    per_struct = df.groupby("structure").size()
    print(f"candidates per structure: median {per_struct.median():.0f}, max {per_struct.max()} "
          f"({per_struct.idxmax()}), total {len(df)}")

    buckets = Counter()
    site_stats = Counter()
    nms_stats = Counter()
    unrelated_examples = []
    for name, sub in df.groupby("structure"):
        known = answers.get(str(name).lower(), set())
        pos = sub[sub["prediction"]]
        keys = [parse_ids(t) for t in pos["residue_ids"]]
        for k, row in zip(keys, pos.itertuples()):
            if k in known:
                buckets["TP (exact)"] += 1
                continue
            best = max((len(k & m) for m in known), default=0)
            if best >= 6:
                buckets["near-miss (6-7/8 shared)"] += 1
            elif best >= 1:
                buckets["partial (1-5/8 shared)"] += 1
            else:
                buckets["unrelated (0/8 shared)"] += 1
                if len(unrelated_examples) < 10:
                    unrelated_examples.append((name, row.residue_ids, getattr(row, "probability", None)))

        # Sites: does each site's best candidate hit a known motif?
        probs = pos["probability"].tolist() if "probability" in pos else [1.0] * len(keys)

        # After non-maximum suppression
        picked = {keys[i] for i in nms(keys, probs, threshold, max_shared)}
        nms_stats["TP"] += len(picked & known)
        nms_stats["FN"] += len(known - picked)
        for k in picked - known:
            best = max((len(k & m) for m in known), default=0)
            nms_stats["FP on a known site" if best else "FP, no annotated motif nearby"] += 1
        for group in sites(keys):
            best_i = max(group, key=lambda i: probs[i])
            touches = any(keys[i] & m for i in group for m in known)
            if keys[best_i] in known:
                site_stats["site, top candidate = exact motif"] += 1
            elif touches:
                site_stats["site on a known motif, top candidate wrong"] += 1
            else:
                site_stats["site with no annotated motif"] += 1

    total_fp = sum(v for k, v in buckets.items() if not k.startswith("TP"))
    print("predicted positives:")
    for k in ("TP (exact)", "near-miss (6-7/8 shared)", "partial (1-5/8 shared)", "unrelated (0/8 shared)"):
        print(f"  {k:<28} {buckets[k]:>8}")
    print(f"  {'all FP':<28} {total_fp:>8}")
    print("sites (groups of overlapping positives):")
    for k, v in site_stats.items():
        print(f"  {k:<44} {v:>6}")
    tp, fn = nms_stats["TP"], nms_stats["FN"]
    fp = nms_stats["FP on a known site"] + nms_stats["FP, no annotated motif nearby"]
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"after NMS (p >= {threshold}, max {max_shared} shared residues between picks):")
    print(f"  TP {tp}  FP {fp}  FN {fn}  precision {prec:.3f}  recall {rec:.3f}")
    print(f"    FP on a known site: {nms_stats['FP on a known site']}, "
          f"FP with no annotated motif nearby: {nms_stats['FP, no annotated motif nearby']}")
    if unrelated_examples:
        print("examples of unrelated FPs (check these in a viewer):")
        for name, ids, p in unrelated_examples:
            print(f"  {name}  p={p}  {ids}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-results", type=Path, required=True, help="result folder of _13_test_all_models.py")
    p.add_argument("-answers", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--nms-max-shared", type=int, default=4,
                   help="picks may share at most this many residues (default 4 = one tetrad)")
    args = p.parse_args()
    answers = load_answers(args.answers)
    for csv_path in sorted(args.results.glob("*.csv")):
        if csv_path.name == "summary.csv":
            continue
        analyze(csv_path, answers, args.threshold, args.nms_max_shared)


if __name__ == "__main__":
    main()