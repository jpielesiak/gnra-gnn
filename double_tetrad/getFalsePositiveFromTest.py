import csv
import glob
import os
from collections import defaultdict

folder = "./models_test"   # <-- change this
output = "falsePositives.csv"

# --- diagnostics ---
print("script started", flush=True)
print("absolute folder:", os.path.abspath(folder), flush=True)
print("folder exists:", os.path.isdir(folder), flush=True)

csvs = glob.glob(os.path.join(folder, "*.csv"))
print("csv files found:", len(csvs), flush=True)
for f in csvs[:5]:
    print("  -", f, flush=True)
# -------------------

counts = defaultdict(int)
for filepath in csvs:
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prediction = row["prediction"].strip().lower() == "true"
            ground_truth = row["ground_truth"].strip().lower() == "true"
            if prediction and not ground_truth:
                key = (row["structure"], row["candidate_index"])
                counts[key] += 1

with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["structure", "candidate_index", "false_positive_count"])
    for (structure, candidate_index), count in sorted(
        counts.items(), key=lambda x: x[1], reverse=True
    ):
        writer.writerow([structure, candidate_index, count])

print(f"Wrote {len(counts)} (structure, candidate_index) pairs to {output}", flush=True)
