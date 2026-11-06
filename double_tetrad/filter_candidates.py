#!/usr/bin/env python3
"""
Filter candidate rows using a false-positive-count table.

Keeps rows whose (structure, candidate_index) appears in the FP table
with false_positive_count >= threshold, and writes them all to one CSV.

Usage:
    python filter_candidates.py fp_counts.csv out.csv predictions1.csv [predictions2.csv ...]
    python filter_candidates.py fp_counts.csv out.csv "predictions/*.csv" --threshold 16

    python filter_candidates.py falsePositives.csv FPtoAdd.csv models_test/approximate-affinity-propagation-decision-tree.csv --threshold 16
"""
import argparse
import glob
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fp_csv", help="CSV with structure,candidate_index,false_positive_count")
    p.add_argument("output", help="Path of the combined output CSV")
    p.add_argument("inputs", nargs="+", help="Input CSV(s) or glob patterns")
    p.add_argument("--threshold", type=int, default=16,
                   help="Minimum false_positive_count to keep (default: 16)")
    args = p.parse_args()

    # Expand globs (useful if the shell didn't, or when quoted)
    paths = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    fp = pd.read_csv(args.fp_csv)
    fp = fp[fp["false_positive_count"] >= args.threshold]
    keys = fp[["structure", "candidate_index"]].drop_duplicates()
    print(f"{len(keys)} (structure, candidate_index) pairs with "
          f"false_positive_count >= {args.threshold}")

    kept_total = 0
    first = True
    # Process in chunks so big files don't need to fit in memory
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=500_000):
            # Inner merge = drops rows not present in the FP table
            filtered = chunk.merge(keys, on=["structure", "candidate_index"], how="inner")
            if filtered.empty:
                continue
            filtered.to_csv(args.output, mode="w" if first else "a",
                            header=first, index=False)
            first = False
            kept_total += len(filtered)
        print(f"Processed {path}")

    if first:  # nothing matched; still write an empty file with the header
        pd.read_csv(paths[0], nrows=0).to_csv(args.output, index=False)

    print(f"Wrote {kept_total} rows to {args.output}")


if __name__ == "__main__":
    sys.exit(main())