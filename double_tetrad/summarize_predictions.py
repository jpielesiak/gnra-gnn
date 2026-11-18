#!/usr/bin/env python3
"""
Summarize binary-classification results stored in CSV files.

For every *.csv file in the given folder that has the expected columns
    structure, candidate_index, residue_ids, prediction, probability, ground_truth
two text files are written:
    false_<name>.txt    - all false positives and false negatives
    summary_<name>.txt  - confusion matrix, classification metrics,
                          probability-based metrics, per-structure breakdown

Usage:
    python summarize_predictions.py /path/to/folder
    uv run summarize_predictions.py ./models_test_11_11 -o ./models_test_11_11_summary

Only the Python standard library is used.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REQUIRED_COLUMNS = [
    "structure",
    "candidate_index",
    "residue_ids",
    "prediction",
    "probability",
    "ground_truth",
]

TRUE_VALUES = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_VALUES = {"false", "f", "no", "n", "0", "0.0"}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_bool(value):
    v = str(value).strip().lower()
    if v in TRUE_VALUES:
        return True
    if v in FALSE_VALUES:
        return False
    raise ValueError(f"cannot interpret {value!r} as True/False")


def parse_probability(value):
    v = str(value).strip()
    if v == "" or v.lower() in {"nan", "none", "null", "na"}:
        return None
    p = float(v)
    if math.isnan(p):
        return None
    return p


def load_csv(path):
    """Return (rows, bad_lines, error_message). error_message is None on success."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            return None, None, f"missing column(s): {', '.join(missing)}"
        reader.fieldnames = header

        rows, bad_lines = [], []
        for line_no, raw in enumerate(reader, start=2):
            try:
                rows.append({
                    "structure": raw["structure"].strip(),
                    "candidate_index": raw["candidate_index"].strip(),
                    "residue_ids": raw["residue_ids"].strip(),
                    "prediction": parse_bool(raw["prediction"]),
                    "probability": parse_probability(raw["probability"]),
                    "ground_truth": parse_bool(raw["ground_truth"]),
                    "raw": [raw[c] for c in REQUIRED_COLUMNS],
                })
            except (ValueError, AttributeError, TypeError) as e:
                bad_lines.append((line_no, str(e)))
    return rows, bad_lines, None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def safe_div(a, b):
    return a / b if b else None


def confusion(rows):
    tp = fp = fn = tn = 0
    for r in rows:
        p, g = r["prediction"], r["ground_truth"]
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def f_beta(precision, recall, beta):
    if precision is None or recall is None:
        return None
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else None


def classification_metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    balanced_acc = (recall + specificity) / 2 if None not in (recall, specificity) else None
    youden = recall + specificity - 1 if None not in (recall, specificity) else None

    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else None

    # Cohen's kappa
    acc = safe_div(tp + tn, n)
    if n:
        p_exp = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        kappa = (acc - p_exp) / (1 - p_exp) if p_exp != 1 else None
    else:
        kappa = None

    return [
        ("Accuracy", acc),
        ("Balanced accuracy", balanced_acc),
        ("Precision (PPV)", precision),
        ("Recall / Sensitivity (TPR)", recall),
        ("Specificity (TNR)", specificity),
        ("Negative predictive value (NPV)", safe_div(tn, tn + fn)),
        ("F1 score", safe_div(2 * tp, 2 * tp + fp + fn)),
        ("F0.5 score", f_beta(precision, recall, 0.5)),
        ("F2 score", f_beta(precision, recall, 2.0)),
        ("Matthews correlation coefficient (MCC)", mcc),
        ("Cohen's kappa", kappa),
        ("Jaccard index / IoU", safe_div(tp, tp + fp + fn)),
        ("Youden's J", youden),
        ("False positive rate (FPR)", safe_div(fp, fp + tn)),
        ("False negative rate (FNR)", safe_div(fn, fn + tp)),
        ("False discovery rate (FDR)", safe_div(fp, fp + tp)),
        ("False omission rate (FOR)", safe_div(fn, fn + tn)),
        ("Prevalence (share of GT positives)", safe_div(tp + fn, n)),
        ("Predicted positive rate", safe_div(tp + fp, n)),
    ]


def roc_auc(probs, labels):
    """ROC AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(r for r, l in zip(ranks, labels) if l)
    return (sum_pos_ranks - pos * (pos + 1) / 2) / (pos * neg)


def average_precision(probs, labels):
    """Area under the precision-recall curve (step-wise, same as sklearn)."""
    pos = sum(labels)
    if pos == 0:
        return None
    pairs = sorted(zip(probs, labels), key=lambda x: -x[0])
    tp = fp = 0
    ap = prev_recall = 0.0
    i = 0
    while i < len(pairs):
        threshold = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == threshold:
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / pos
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
    return ap


def probability_metrics(rows):
    usable = [r for r in rows if r["probability"] is not None]
    probs = [r["probability"] for r in usable]
    labels = [1 if r["ground_truth"] else 0 for r in usable]
    n = len(usable)
    out_of_range = sum(1 for p in probs if p < 0 or p > 1)

    if n == 0:
        return [], 0, len(rows), out_of_range

    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    eps = 1e-15
    log_loss = -sum(
        y * math.log(min(max(p, eps), 1 - eps)) + (1 - y) * math.log(1 - min(max(p, eps), 1 - eps))
        for p, y in zip(probs, labels)
    ) / n

    pos_probs = [p for p, y in zip(probs, labels) if y]
    neg_probs = [p for p, y in zip(probs, labels) if not y]
    pred_true = [r["probability"] for r in usable if r["prediction"]]
    pred_false = [r["probability"] for r in usable if not r["prediction"]]

    metrics = [
        ("ROC AUC", roc_auc(probs, labels)),
        ("Average precision (PR AUC)", average_precision(probs, labels)),
        ("Brier score", brier),
        ("Log loss", log_loss),
        ("Mean probability, GT positives", safe_div(sum(pos_probs), len(pos_probs))),
        ("Mean probability, GT negatives", safe_div(sum(neg_probs), len(neg_probs))),
        ("Min probability among predicted True", min(pred_true) if pred_true else None),
        ("Max probability among predicted False", max(pred_false) if pred_false else None),
    ]
    return metrics, n, len(rows) - n, out_of_range


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def fmt(value):
    if value is None:
        return "N/A (undefined)"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def write_false_file(path, source_name, rows):
    fps = [r for r in rows if r["prediction"] and not r["ground_truth"]]
    fns = [r for r in rows if not r["prediction"] and r["ground_truth"]]

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"False positives and false negatives for: {source_name}\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        writer = csv.writer(f, lineterminator="\n")
        for title, subset in (
            ("FALSE POSITIVES (prediction=True, ground_truth=False)", fps),
            ("FALSE NEGATIVES (prediction=False, ground_truth=True)", fns),
        ):
            f.write(f"=== {title}: {len(subset)} ===\n")
            if subset:
                writer.writerow(REQUIRED_COLUMNS)
                for r in subset:
                    writer.writerow(r["raw"])
            else:
                f.write("(none)\n")
            f.write("\n")
    return len(fps), len(fns)


def write_summary_file(path, source_name, rows, bad_lines):
    tp, fp, fn, tn = confusion(rows)
    structures = sorted({r["structure"] for r in rows})
    lines = []
    add = lines.append

    add(f"Summary for: {source_name}")
    add(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("=" * 60)
    add(f"Rows evaluated:           {len(rows)}")
    add(f"Rows skipped (bad data):  {len(bad_lines)}")
    add(f"Unique structures:        {len(structures)}")
    add("")

    add("CONFUSION MATRIX")
    add("-" * 60)
    w = max(len(str(x)) for x in (tp, fp, fn, tn, "Pred False"))
    add(f"{'':22}{'Pred True':>{w + 2}}{'Pred False':>{w + 2}}")
    add(f"{'Ground truth True':22}{tp:>{w + 2}}{fn:>{w + 2}}")
    add(f"{'Ground truth False':22}{fp:>{w + 2}}{tn:>{w + 2}}")
    add("")

    add("COUNTS")
    add("-" * 60)
    add(f"True positives  (TP): {tp}")
    add(f"False positives (FP): {fp}")
    add(f"False negatives (FN): {fn}")
    add(f"True negatives  (TN): {tn}")
    add(f"Ground-truth positives: {tp + fn}")
    add(f"Ground-truth negatives: {fp + tn}")
    add(f"Predicted positives:    {tp + fp}")
    add(f"Predicted negatives:    {fn + tn}")
    add("")

    add("CLASSIFICATION METRICS (based on the 'prediction' column)")
    add("-" * 60)
    metrics = classification_metrics(tp, fp, fn, tn)
    name_w = max(len(n) for n, _ in metrics)
    for name, value in metrics:
        add(f"{name:<{name_w}} : {fmt(value)}")
    add("")

    add("PROBABILITY-BASED METRICS (based on the 'probability' column)")
    add("-" * 60)
    prob_metrics, n_used, n_missing, out_of_range = probability_metrics(rows)
    add(f"Rows with a usable probability: {n_used} (missing: {n_missing})")
    if out_of_range:
        add(f"WARNING: {out_of_range} probabilities lie outside [0, 1]")
    if prob_metrics:
        name_w = max(len(n) for n, _ in prob_metrics)
        for name, value in prob_metrics:
            add(f"{name:<{name_w}} : {fmt(value)}")
    add("")

    add("PER-STRUCTURE BREAKDOWN")
    add("-" * 60)
    by_struct = defaultdict(list)
    for r in rows:
        by_struct[r["structure"]].append(r)
    s_w = max([len("structure")] + [len(s) for s in structures])
    add(f"{'structure':<{s_w}}  {'n':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>6}  "
        f"{'precision':>10} {'recall':>10} {'F1':>10}")
    structures_with_tp = 0
    for s in structures:
        stp, sfp, sfn, stn = confusion(by_struct[s])
        if stp:
            structures_with_tp += 1
        prec = safe_div(stp, stp + sfp)
        rec = safe_div(stp, stp + sfn)
        f1 = safe_div(2 * stp, 2 * stp + sfp + sfn)
        cell = lambda v: "N/A" if v is None else f"{v:.4f}"
        add(f"{s:<{s_w}}  {len(by_struct[s]):>6} {stp:>5} {sfp:>5} {sfn:>5} {stn:>6}  "
            f"{cell(prec):>10} {cell(rec):>10} {cell(f1):>10}")
    structures_with_gt = sum(1 for s in structures if any(r["ground_truth"] for r in by_struct[s]))
    add("")
    add(f"Structures with >=1 ground-truth positive: {structures_with_gt}")
    add(f"Structures with >=1 true positive found:   {structures_with_tp}")
    add("")

    if bad_lines:
        add("SKIPPED LINES")
        add("-" * 60)
        for line_no, reason in bad_lines:
            add(f"line {line_no}: {reason}")
        add("")

    add("Note: 'N/A (undefined)' means the metric's denominator was zero")
    add("(e.g. precision when the model predicted no positives).")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return tp, fp, fn, tn


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Summarize ML prediction CSV files in a folder.")
    parser.add_argument("folder", help="Folder containing the CSV files")
    parser.add_argument("-o", "--output-dir",
                        help="Where to write the .txt files (default: same folder as the CSVs)")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Error: '{folder}' is not a folder.")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return

    processed = skipped = 0
    for csv_path in csv_files:
        stem = csv_path.stem
        try:
            rows, bad_lines, error = load_csv(csv_path)
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            print(f"[SKIP] {csv_path.name}: could not read file ({e})")
            skipped += 1
            continue
        if error:
            print(f"[SKIP] {csv_path.name}: {error}")
            skipped += 1
            continue

        n_fp, n_fn = write_false_file(out_dir / f"false_{stem}.txt", csv_path.name, rows)
        tp, fp, fn, tn = write_summary_file(out_dir / f"summary_{stem}.txt", csv_path.name, rows, bad_lines)
        warn = f", {len(bad_lines)} bad line(s) skipped" if bad_lines else ""
        print(f"[OK]   {csv_path.name}: {len(rows)} rows  TP={tp} FP={fp} FN={fn} TN={tn}{warn}")
        processed += 1

    print(f"\nDone: {processed} file(s) summarized, {skipped} skipped. Output in: {out_dir}")


if __name__ == "__main__":
    main()