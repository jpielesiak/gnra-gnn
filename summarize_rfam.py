#!/usr/bin/env python3
"""
Summarize the CSVs made by clusters_to_rfam.py across several clusterings.

Finds every *_representatives_rfam.csv and every other *_rfam.csv under the
given folders (searched recursively) and writes:
    representatives_rfam_summary.txt  - from the *_representatives_rfam.csv files
    rfam_summary.txt                  - from the *_rfam.csv files

Each motif .cif filename is counted once, however many clusterings it shows
up in. If the same motif got different Rfam results in different CSVs, the
chain-level results are preferred over entry-level ones; otherwise the
families from all appearances are combined.

Usage:
    uv run summarize_rfam.py                       # search the current folder
    python summarize_rfam.py results1 results2     # search these folders
    python summarize_rfam.py results --out-dir summaries
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REP_SUFFIX = "_representatives_rfam.csv"
ALL_SUFFIX = "_rfam.csv"
NO_MATCH = "(no Rfam match)"


# ---------------------------------------------------------------- statistics
def gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), used for the chi-square p-value."""
    if x <= 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:  # series expansion
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300  # continued fraction (Lentz)
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(log_prefix) * h


def evenness_lines(counts):
    """Describe how close a list of family counts is to an equal split."""
    k, n = len(counts), sum(counts)
    if k < 2:
        return ["Only one family present, so there is nothing to compare."]
    expected = n / k
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    df = k - 1
    p = gamma_q(df / 2, chi2 / 2)
    shannon = -sum((c / n) * math.log(c / n) for c in counts if c)
    pielou = shannon / math.log(k)
    mean = n / k
    cv = math.sqrt(sum((c - mean) ** 2 for c in counts) / k) / mean

    lines = [
        f"Families compared:                 {k}",
        f"Expected per family if equal:      {expected:.2f}",
        f"Observed range:                    {min(counts)} - {max(counts)}",
        f"Largest / smallest family ratio:   {max(counts) / min(counts):.2f}",
        f"Coefficient of variation:          {cv:.3f}   (0 = perfectly equal)",
        f"Pielou's evenness J:               {pielou:.3f}   (1 = perfectly equal)",
        f"Chi-square vs. equal split:        X2 = {chi2:.2f}, df = {df}, p = {p:.3g}",
    ]
    if p < 0.05:
        lines.append("-> The distribution differs significantly from an equal split (p < 0.05).")
    else:
        lines.append("-> No significant difference from an equal split (p >= 0.05).")
    if expected < 5:
        lines.append("   Note: fewer than 5 motifs expected per family, so the chi-square "
                     "p-value is only approximate.")
    return lines


# ---------------------------------------------------------------- reading CSVs
def find_csvs(folders):
    reps, alls = [], []
    for folder in folders:
        for path in sorted(folder.rglob("*" + ALL_SUFFIX)):
            (reps if path.name.endswith(REP_SUFFIX) else alls).append(path)
    return reps, alls


def collect_motifs(csv_paths):
    """motif filename -> {rfam_acc: (rfam_id, description)}, deduplicated across CSVs."""
    # per motif: match level -> families seen at that level
    seen = defaultdict(lambda: defaultdict(dict))
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                motif = Path(row.get("motif_file") or row.get("motif_id", "")).name
                if not motif:
                    continue
                level = "chain" if row.get("match_level") == "chain" else "other"
                acc = row.get("rfam_acc", "").strip()
                fams = seen[motif][level]
                if acc:
                    fams[acc] = (row.get("rfam_id", ""), row.get("rfam_description", ""))

    motifs = {}
    for motif, levels in seen.items():
        if "chain" in levels:
            motifs[motif] = levels["chain"]
        else:
            motifs[motif] = levels["other"]
    return motifs


# ---------------------------------------------------------------- report
def build_report(title, csv_paths, motifs):
    family_members = defaultdict(set)
    family_info = {}
    for motif, fams in motifs.items():
        if not fams:
            family_members[NO_MATCH].add(motif)
        for acc, info in fams.items():
            family_members[acc].add(motif)
            family_info[acc] = info
    family_info[NO_MATCH] = ("", "")

    n_motifs = len(motifs)
    n_matched = sum(1 for f in motifs.values() if f)
    n_multi = sum(1 for f in motifs.values() if len(f) > 1)
    real_families = [a for a in family_members if a != NO_MATCH]

    out = [title, "=" * len(title), "", f"Source CSV files ({len(csv_paths)}):"]
    out += [f"  {p}" for p in csv_paths]
    out += [
        "",
        f"Unique motif files:        {n_motifs}",
        f"  with an Rfam family:     {n_matched}",
        f"  without an Rfam family:  {n_motifs - n_matched}",
        f"  in more than one family: {n_multi}",
        f"Rfam families:             {len(real_families)}",
        "",
    ]

    # --- distribution
    out += ["FAMILY DISTRIBUTION", "-------------------"]
    ordered = sorted(real_families, key=lambda a: (-len(family_members[a]), a))
    if NO_MATCH in family_members:
        ordered.append(NO_MATCH)
    max_count = max((len(m) for m in family_members.values()), default=1)
    id_w = max([len(family_info[a][0]) for a in ordered] + [7])
    acc_w = max([len(a) for a in ordered] + [8])
    out.append(f"{'rfam_acc':<{acc_w}}  {'rfam_id':<{id_w}}  {'motifs':>6}  {'%':>6}  description")
    for acc in ordered:
        count = len(family_members[acc])
        rid, desc = family_info[acc]
        bar = "#" * max(1, round(30 * count / max_count))
        out.append(f"{acc:<{acc_w}}  {rid:<{id_w}}  {count:>6}  "
                   f"{100 * count / n_motifs:>5.1f}%  {desc}")
        out.append(f"{'':<{acc_w}}  {bar}")
    if n_multi:
        out.append("Note: some motifs belong to more than one family, so the percentages "
                   "add up to more than 100%.")
    out.append("")

    out += ["EVENNESS (motifs without an Rfam family excluded)",
            "-------------------------------------------------"]
    out += evenness_lines([len(family_members[a]) for a in real_families])
    out.append("")

    # --- per motif
    out += ["MOTIFS", "------",
            "others_in_family = how many other unique motif files share that family", ""]
    name_w = max([len(m) for m in motifs] + [10])
    out.append(f"{'motif_file':<{name_w}}  {'rfam_acc':<{acc_w}}  "
               f"{'rfam_id':<{id_w}}  others_in_family")
    for motif in sorted(motifs):
        fams = motifs[motif] or {NO_MATCH: ("", "")}
        for acc in sorted(fams):
            others = len(family_members[acc]) - 1
            out.append(f"{motif:<{name_w}}  {acc:<{acc_w}}  "
                       f"{family_info[acc][0]:<{id_w}}  {others}")
    return "\n".join(out) + "\n"


def write_summary(title, csv_paths, out_path):
    if not csv_paths:
        print(f"No CSV files found for {out_path.name}, skipping.", file=sys.stderr)
        return
    motifs = collect_motifs(csv_paths)
    out_path.write_text(build_report(title, csv_paths, motifs), encoding="utf-8")
    print(f"Wrote {out_path} ({len(motifs)} unique motifs from {len(csv_paths)} files)",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", type=Path, nargs="*", default=[Path.cwd()],
                    help="Folders to search recursively for the CSVs (default: current folder).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write the summaries (default: the first folder).")
    args = ap.parse_args()

    reps, alls = find_csvs(args.folders)
    out_dir = args.out_dir or args.folders[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary("Rfam summary - cluster representatives", reps,
                  out_dir / "representatives_rfam_summary.txt")
    write_summary("Rfam summary - all motifs", alls, out_dir / "rfam_summary.txt")


if __name__ == "__main__":
    main()