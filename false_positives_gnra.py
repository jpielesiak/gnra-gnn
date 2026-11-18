#!/usr/bin/env python3
"""
Filter false-positive windows, check their sequence for a GNRA motif and,
optionally, look up their secondary structure in a folder of .dbn files.

A row is a false positive when prediction == True and ground_truth == False.
GNRA = G, any nucleotide (N), purine (A/G), A.

Columns added:
    motif_seq         True if the sequence contains a GNRA match, else False
    motif_nt_start    1-based position in the sequence string where the match
                      starts (empty if there is no match)
    motif_contiguous  True if the four matched residues are on the same chain
                      with consecutive residue numbers, False if the match
                      spans a numbering jump or chain change, empty if there
                      is no match or it can't be checked

With --dbn-dir, also:
    dbn_structure     dot-bracket of the fragment, one character per residue,
                      aligned with the `sequence` column
    dbn_status        found / ambiguous / not_found / chain_not_in_dbn /
                      no_dbn_file / bad_residue_ids
    dbn_matches       number of places in the .dbn the fragment matched
    dbn_positions     1-based positions in the .dbn chain (only for a single match)
    dbn_file          .dbn file(s) the structure came from

How the lookup works: .dbn files carry no residue numbers, so the fragment is
located by sequence. It is split into stretches of consecutive residue
numbers; each stretch is searched in the sequence of its chain, and the
stretches must appear in order with a gap no larger than the numbering gap
(the .dbn may lack unmodelled residues, never add residues). If every match
gives the same dot-bracket the status is "found"; if they differ, all
alternatives are listed separated by ";" and the status is "ambiguous".

Expected .dbn names: <pdb>-assembly<N>-<chain>_<chain>....dbn,
e.g. 1a0a-assembly1-C_D.dbn, containing a ">" header line, the sequence and
the dot-bracket. Chains inside a file may be separated by "&".

A .dbn whose dot-bracket has no pairs at all (only dots) is ignored, so
fragments that only occur in such files get dbn_status "not_found".

Output (one set of files per input CSV, in the output folder):
  without --dbn-dir:  <input name>_fp_gnra.csv        all false positives
  with --dbn-dir:     <input name>_fp_hairpin.csv     hairpin-like structure
                      <input name>_fp_no_hairpin.csv  structure found, not hairpin-like
                      <input name>_fp_mixed.csv       only if needed: ambiguous rows whose
                                                      alternatives disagree on hairpin-like
  False positives without a secondary structure are dropped in --dbn-dir mode.

Hairpin-like (column hairpin_like): the counts of "(" and ")" in the fragment
differ by at most --max-diff (default 1, to allow a hairpin shifted in the
window), and at least one "(" closes with a later ")" inside the fragment
(so "........" and "))((" are not hairpin-like). Only round brackets count;
pseudoknot brackets ([] {} <>) are ignored.

Input can be a folder of CSVs or a single CSV.

Usage:
    python false_positives_gnra.py /path/to/csv_folder /path/to/output_folder
    python false_positives_gnra.py /path/to/csv_folder out/ --dbn-dir /path/to/dbn_folder
    python false_positives_gnra.py /path/to/csv_folder out/ --dbn-dir dbn/ --max-diff 0

    uv run false_positives_gnra.py ./split_test ./split_test_dbns --dbn-dir ./consensus
    uv run false_positives_gnra.py ./split_test ./split_test_dbns --dbn-dir ./consensus --max-diff 0
"""
import argparse
import csv
import re
from pathlib import Path

# Lookahead so overlapping matches are also found (e.g. "GGAGA").
GNRA = re.compile(r"(?=(G[ACGUTN][AG]A))", re.IGNORECASE)
RES_RE = re.compile(r"^\s*([^:]+):(-?\d+)([A-Za-z]?)\s*$")
DBN_NAME = re.compile(r"^(?P<pdb>[^-]+)-(?P<asm>assembly\d+)-(?P<chains>.+)$", re.IGNORECASE)
STRUCT_NAME = re.compile(r"^(?P<pdb>[^-]+)(?:-(?P<asm>assembly\d+))?", re.IGNORECASE)

OUT_SUFFIXES = {"all": "_fp_gnra", "hairpin": "_fp_hairpin",
                "no_hairpin": "_fp_no_hairpin", "mixed": "_fp_mixed"}
REQUIRED = {"prediction", "ground_truth", "sequence", "residue_ids"}
GNRA_COLS = ["motif_seq", "motif_nt_start", "motif_contiguous"]
DBN_COLS = ["dbn_structure", "dbn_status", "dbn_matches", "dbn_positions", "dbn_file",
            "hairpin_like"]
MAX_HITS = 1000  # cap on placements per fragment, to keep short fragments fast


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes", "t"}


def parse_ids(residue_ids):
    """Parse "D:582,D:583,..." into [(chain, number), ...]; None if unparsable."""
    parsed = []
    for token in (residue_ids or "").split(","):
        if not token.strip():
            continue
        m = RES_RE.match(token)
        if not m:
            return None
        parsed.append((m.group(1), int(m.group(2))))
    return parsed


# ----------------------------------------------------------------- GNRA ----

def is_contiguous(residues, start0, length=4):
    """True if residues[start0:start0+length] are same-chain, consecutive numbers."""
    window = residues[start0:start0 + length]
    if len(window) < length:
        return None
    return all(b[0] == a[0] and b[1] == a[1] + 1 for a, b in zip(window, window[1:]))


def find_gnra(sequence, residues):
    """Return (1-based start, contiguous) for the best match, or (None, None)."""
    starts = [m.start() for m in GNRA.finditer(sequence or "")]
    if not starts:
        return None, None
    if residues is None or len(residues) != len(sequence):
        return starts[0] + 1, None  # can't map sequence positions to residues
    flags = [is_contiguous(residues, s) for s in starts]
    for s, ok in zip(starts, flags):
        if ok:
            return s + 1, True
    return starts[0] + 1, flags[0]


# ------------------------------------------------------------------ DBN ----

def norm(seq):
    return seq.upper().replace("T", "U")


def index_dbn_folder(dbn_dir):
    """{pdb_id: [(assembly, [chains], path), ...]}"""
    index = {}
    for f in Path(dbn_dir).glob("*.dbn"):
        m = DBN_NAME.match(f.stem)
        if m:
            index.setdefault(m["pdb"].lower(), []).append(
                (m["asm"].lower(), m["chains"].split("_"), f))
    return index


def read_dbn(path, chains):
    """Return {chain: (seq, dotbracket)}, or {None: (...)} if chains can't be split."""
    lines = [l.strip() for l in open(path) if l.strip()]
    body = [l for l in lines if not l.startswith(">")]
    if len(body) < 2:
        print(f"[warn] {path.name}: expected sequence and dot-bracket lines")
        return {}
    seq, db = body[0], body[1]
    amp_seq = [i for i, c in enumerate(seq) if c == "&"]
    amp_db = [i for i, c in enumerate(db) if c == "&"]
    if len(seq) != len(db) or amp_seq != amp_db:
        print(f"[warn] {path.name}: sequence and dot-bracket don't line up, skipped")
        return {}
    if not any(c not in ".&" for c in db):
        return {}  # no pairs anywhere in the molecule: treat as not found
    seq_parts, db_parts = seq.split("&"), db.split("&")
    if len(seq_parts) == len(chains):
        return {c: (norm(s), d) for c, s, d in zip(chains, seq_parts, db_parts)}
    if len(seq_parts) == 1 and len(chains) > 1:
        print(f"[warn] {path.name}: no '&' chain separators, searching the whole sequence")
    return {None: (norm(seq.replace("&", "")), db.replace("&", ""))}


def split_segments(residues, sequence):
    """Split a fragment into stretches of consecutive residue numbers."""
    segs = []
    start = 0
    for i in range(1, len(residues) + 1):
        if i == len(residues) or not (residues[i][0] == residues[i - 1][0]
                                      and residues[i][1] == residues[i - 1][1] + 1):
            segs.append((residues[start][0], residues[start][1], residues[i - 1][1],
                         norm(sequence[start:i])))
            start = i
    return segs


def occurrences(pattern, text):
    rx = "".join("." if c == "N" else re.escape(c) for c in pattern)
    return [m.start() for m in re.finditer(f"(?=({rx}))", text)]


def locate(segments, chains):
    """All placements of the segments in the .dbn; each is [(key, pos, len), ...]."""
    keys = []
    for chain, *_ in segments:
        key = chain if chain in chains else (None if None in chains else "missing")
        if key == "missing":
            return []
        keys.append(key)
    occ = [occurrences(seg[3], chains[k][0]) for seg, k in zip(segments, keys)]

    results, placed = [], []

    def rec(k):
        if len(results) >= MAX_HITS:
            return
        if k == len(segments):
            results.append(list(placed))
            return
        chain, first, _, s = segments[k]
        for pos in occ[k]:
            if placed:
                pchain, pkey, ppos, plen, plast = placed[-1][3], placed[-1][0], \
                    placed[-1][1], placed[-1][2], placed[-1][4]
                if pkey == keys[k] and pchain == chain and first > plast:
                    gap = pos - (ppos + plen)
                    if gap < 0 or gap > first - plast - 1:
                        continue
            placed.append((keys[k], pos, len(s), chain, segments[k][2]))
            rec(k + 1)
            placed.pop()

    rec(0)
    return results


def find_structure(row, dbn_index, dbn_cache):
    out = {c: "" for c in DBN_COLS}
    m = STRUCT_NAME.match(row.get("structure", ""))
    pdb, asm = (m["pdb"].lower(), (m["asm"] or "").lower()) if m else ("", "")
    files = [(a, ch, f) for a, ch, f in dbn_index.get(pdb, []) if not asm or a == asm]
    if not files:
        out["dbn_status"] = "no_dbn_file"
        return out

    residues = parse_ids(row["residue_ids"])
    seq = row["sequence"] or ""
    if not residues or len(residues) != len(seq):
        out["dbn_status"] = "bad_residue_ids"
        return out
    segments = split_segments(residues, seq)
    frag_chains = {c for c, _ in residues}

    found = {}       # dot-bracket -> list of (file name, positions)
    any_chain = False
    for _, chains, f in sorted(files, key=lambda x: x[2].name):
        if not frag_chains <= set(chains):
            continue
        any_chain = True
        if f not in dbn_cache:
            dbn_cache[f] = read_dbn(f, chains)
        parsed = dbn_cache[f]
        if not parsed:
            continue
        for placement in locate(segments, parsed):
            db = "".join(parsed[k][1][p:p + n] for k, p, n, *_ in placement)
            positions = [p + j + 1 for _, p, n, *_ in placement for j in range(n)]
            found.setdefault(db, []).append((f.name, positions))

    if not found:
        out["dbn_status"] = "not_found" if any_chain else "chain_not_in_dbn"
        return out

    hits = [h for v in found.values() for h in v]
    out["dbn_structure"] = ";".join(found)
    out["dbn_status"] = "found" if len(found) == 1 else "ambiguous"
    out["dbn_matches"] = len(hits)
    out["dbn_file"] = ";".join(sorted({name for name, _ in hits}))
    if len(hits) == 1:
        out["dbn_positions"] = ",".join(map(str, hits[0][1]))
    return out


# -------------------------------------------------------------- hairpin ----

def is_hairpin_like(db, max_diff=1):
    """Balanced-ish round brackets with at least one pair closed inside the fragment."""
    opens, closes = db.count("("), db.count(")")
    if abs(opens - closes) > max_diff:
        return False
    depth = closed_inside = 0
    for c in db:
        if c == "(":
            depth += 1
        elif c == ")" and depth:
            depth -= 1
            closed_inside += 1
    return closed_inside > 0


def classify(dbn_structure, max_diff):
    """'hairpin', 'no_hairpin' or 'mixed' (ambiguous alternatives disagree)."""
    flags = {is_hairpin_like(db, max_diff) for db in dbn_structure.split(";")}
    if len(flags) > 1:
        return "mixed"
    return "hairpin" if flags.pop() else "no_hairpin"


# ----------------------------------------------------------------- main ----

def process_file(in_path, out_dir, dbn_index, dbn_cache, max_diff):
    """Filter one CSV; return a dict of counts, or None if skipped."""
    counts = {"fp": 0, "gnra": 0, "contig": 0, "hairpin": 0, "no_hairpin": 0,
              "mixed": 0, "dropped": 0}
    with open(in_path, newline="") as fin:
        reader = csv.DictReader(fin)
        needed = REQUIRED | ({"structure"} if dbn_index is not None else set())
        missing = needed - set(reader.fieldnames or [])
        if missing:
            print(f"[skip] {in_path.name}: missing columns {sorted(missing)}")
            return None
        fieldnames = reader.fieldnames + GNRA_COLS + (DBN_COLS if dbn_index is not None else [])

        handles, writers = {}, {}

        def writer_for(kind):
            if kind not in writers:
                path = out_dir / f"{in_path.stem}{OUT_SUFFIXES[kind]}.csv"
                handles[kind] = open(path, "w", newline="")
                writers[kind] = csv.DictWriter(handles[kind], fieldnames=fieldnames)
                writers[kind].writeheader()
            return writers[kind]

        # Always create the main output file(s), even if they end up empty.
        for kind in (["hairpin", "no_hairpin"] if dbn_index is not None else ["all"]):
            writer_for(kind)

        try:
            for row in reader:
                if not (is_true(row["prediction"]) and not is_true(row["ground_truth"])):
                    continue
                counts["fp"] += 1
                start, contiguous = find_gnra(row["sequence"], parse_ids(row["residue_ids"]))
                row["motif_seq"] = start is not None
                row["motif_nt_start"] = start if start is not None else ""
                row["motif_contiguous"] = "" if contiguous is None else contiguous
                counts["gnra"] += start is not None
                counts["contig"] += contiguous is True

                if dbn_index is None:
                    writer_for("all").writerow(row)
                    continue
                row.update(find_structure(row, dbn_index, dbn_cache))
                if row["dbn_status"] not in {"found", "ambiguous"}:
                    counts["dropped"] += 1
                    continue
                kind = classify(row["dbn_structure"], max_diff)
                row["hairpin_like"] = {"hairpin": True, "no_hairpin": False}.get(kind, "mixed")
                counts[kind] += 1
                writer_for(kind).writerow(row)
        finally:
            for h in handles.values():
                h.close()
    return counts


def summary(c, with_dbn):
    msg = f"{c['fp']} false positives, {c['gnra']} with GNRA, {c['contig']} contiguous"
    if with_dbn:
        msg += (f"; {c['hairpin']} hairpin-like, {c['no_hairpin']} not, "
                f"{c['mixed']} mixed, {c['dropped']} dropped (no structure)")
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Folder containing the evaluation CSVs (or a single CSV)")
    ap.add_argument("output_dir", help="Folder to write the false-positive CSVs into")
    ap.add_argument("--dbn-dir", help="Folder with <pdb>-assembly<N>-<chains>.dbn files")
    ap.add_argument("--max-diff", type=int, default=1,
                    help="Max difference between '(' and ')' counts for hairpin-like (default 1)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        # Skip earlier outputs in case the output folder is the input folder.
        files = sorted(f for f in in_path.glob("*.csv")
                       if not any(f.stem.endswith(sfx) for sfx in OUT_SUFFIXES.values()))
    elif in_path.is_file():
        files = [in_path]
    else:
        raise SystemExit(f"Input not found: {in_path}")
    if not files:
        raise SystemExit(f"No CSV files found in {in_path}")

    dbn_index = None
    if args.dbn_dir:
        if not Path(args.dbn_dir).is_dir():
            raise SystemExit(f"DBN folder not found: {args.dbn_dir}")
        dbn_index = index_dbn_folder(args.dbn_dir)
        print(f"Indexed .dbn files for {len(dbn_index)} PDB entries")
    dbn_cache = {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = None
    for f in files:
        c = process_file(f, out_dir, dbn_index, dbn_cache, args.max_diff)
        if c is None:
            continue
        totals = c if totals is None else {k: totals[k] + c[k] for k in totals}
        print(f"{f.name}: {summary(c, dbn_index is not None)}")

    if totals:
        print(f"Total: {summary(totals, dbn_index is not None)}")
    print(f"Output written to {out_dir}")


if __name__ == "__main__":
    main()