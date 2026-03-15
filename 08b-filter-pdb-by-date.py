import requests
import csv
import time
import pandas as pd
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_URL = "https://data.rcsb.org/rest/v1/core/entry/"

# ------------------------------------
# 1. Get all PDB IDs containing RNA
# ------------------------------------
def get_rna_pdb_ids():
    all_ids = []
    start = 0
    page_size = 10000

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "RNA"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": start,
                "rows": page_size
            }
        }
    }

    while True:
        query["request_options"]["paginate"]["start"] = start

        response = requests.post(SEARCH_URL, json=query)
        if response.status_code != 200:
            print(response.text)
            response.raise_for_status()

        data = response.json()
        result_set = data.get("result_set", [])

        if not result_set:
            break

        ids = [entry["identifier"] for entry in result_set]
        all_ids.extend(ids)

        print(f"Fetched {len(all_ids)} RNA entries so far...")
        start += page_size

        if len(result_set) < page_size:
            break

    return all_ids

# ------------------------------------
# 2. Fetch release dates
# ------------------------------------
def fetch_release_dates(pdb_ids):
    results = []

    for i, pdb_id in enumerate(pdb_ids):
        try:
            response = requests.get(DATA_URL + pdb_id)
            response.raise_for_status()
            entry = response.json()

            release_date = entry["rcsb_accession_info"]["initial_release_date"]
            results.append((pdb_id, release_date))

        except Exception as e:
            print(f"Error fetching {pdb_id}: {e}")

        if i % 100 == 0:
            print(f"Processed {i} entries")
            time.sleep(0.1)

    return results


# ------------------------------------
# 3. Main for dates download and csv writing
# ------------------------------------
def download_dates_and_save_csv():
    print("Getting RNA-containing PDB IDs...")
    pdb_ids = get_rna_pdb_ids()

    print(f"Total RNA entries: {len(pdb_ids)}")
    print("Fetching release dates...")

    data = fetch_release_dates(pdb_ids)

    print("Writing CSV...")
    with open("rna_pdb_release_dates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdbid", "release_date"])
        writer.writerows(data)

    print("Done!")

def filter_csv_by_date(input_csv,dates_csv,output_csv_pre,output_csv_after, cutoff_date):
    # Load cutoff date
    cutoff_date = time.strptime(cutoff_date, "%Y-%m-%d")

    # Load release dates into a dictionary
    release_dates = {}
    with open(dates_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            release_dates[row["pdbid"]] = time.strptime(row["release_date"], "%Y-%m-%d")

    # Filter input CSV and write to output files. If pdbid is older than cutoff, get the entire row to pre, otherwise, to after.
    with open(input_csv, "r") as infile, open(output_csv_pre, "w", newline="") as pre_outfile, open(output_csv_after, "w", newline="") as after_outfile:
        reader = csv.DictReader(infile)
        pre_writer = csv.DictWriter(pre_outfile, fieldnames=reader.fieldnames)
        after_writer = csv.DictWriter(after_outfile, fieldnames=reader.fieldnames)

        pre_writer.writeheader()
        after_writer.writeheader()

        for row in reader:
            pdbid = row["pdbid"]
            if pdbid in release_dates:
                if release_dates[pdbid] < cutoff_date:
                    pre_writer.writerow(row)
                else:
                    after_writer.writerow(row)
def filter_pandas_dataframe_by_date(df, dates_csv, cutoff_date):
      # Load cutoff date
    cutoff_date = pd.to_datetime(cutoff_date)

    # Load release dates into a dictionary
    release_dates = {}
    with open(dates_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            release_dates[row["pdbid"]] = pd.to_datetime(row["release_date"])

    #creating two dfs identical to the df, but empty, with rows left to be filled. named pre_df and after_df
    pre_df = df.copy().iloc[0:0]
    after_df = df.copy().iloc[0:0]
    pre_rows = []
    after_rows = []
    lost_ids = []
    # the dataframe's index is named "source_file" with values like "HL_1G1X_001"
    for index, row in df.iterrows():
        # extracting the pdbid as the second component when split by underscores.
        # e.g. "HL_1G1X_001" -> pdbid "1G1X".
        src = str(index)
        parts = src.split("_")
        if len(parts) >= 2:
            pdbid = parts[1]
            #some entries have a two part prefix, like single_strands_2il9_0003
            if len(parts) == 4:
                pdbid = parts[2]
            
            pdbid_upper = pdbid.upper()
            pdbid_lower = pdbid.lower()
            if pdbid in release_dates:
                if release_dates[pdbid] < cutoff_date:
                    pre_rows.append(row)
                else:
                    after_rows.append(row)
            elif pdbid_upper in release_dates:
                if release_dates[pdbid_upper] < cutoff_date:
                    pre_rows.append(row)
                else:
                    after_rows.append(row)
            elif pdbid_lower in release_dates:
                if release_dates[pdbid_lower] < cutoff_date:
                    pre_rows.append(row)
                else:
                    after_rows.append(row)
            else:
                lost_ids.append(pdbid)
    pre_df = pd.concat([pre_df, pd.DataFrame(pre_rows)], ignore_index=True)
    after_df = pd.concat([after_df, pd.DataFrame(after_rows)], ignore_index=True)
    print(f"Total entries in dataframe: {len(df)}")
    print(f"Entries before cutoff date: {len(pre_df)}")
    print(f"Entries after cutoff date: {len(after_df)}")
    print(f"Lost IDs (not found in release dates): {lost_ids}")
    return pre_df, after_df

def filter_pandas_dataframe_by_date_old(df, dates_csv, cutoff_date):

    # the dataframe's index is named "source_file" with values like "HL_1G1X_001"
    # extracting the pdbid as the second component when split by underscores.
    # e.g. "HL_1G1X_001" -> pdbid "1G1X".

    # Load cutoff date
    cutoff_date = pd.to_datetime(cutoff_date)

    # Load release dates into a dictionary
    release_dates = {}
    with open(dates_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            release_dates[row["pdbid"]] = pd.to_datetime(row["release_date"])

    # extract pdbid from index (source_file)
    def extract_pdbid(src: str) -> str:
        # assume the format structure_pdbid_other; guard against unexpected
        parts = src.split("_")
        if len(parts) >= 2:
            return parts[1]
        return ""  # fallback empty

    df = df.copy()
    df["pdbid"] = df.index.astype(str).map(extract_pdbid)

    # Add release date to dataframe and filter
    df["release_date"] = df["pdbid"].map(release_dates)
    print(f"Total entries in dataframe: {len(df)}")
    pre_df = df[df["release_date"] < cutoff_date].copy()
    print(f"Entries before cutoff date: {len(pre_df)}")
    after_df = df[df["release_date"] >= cutoff_date].copy()
    print(f"Entries before cutoff date: {len(after_df)}")

    # drop the temporary pdbid column from all frames
    for frame in (df, pre_df, after_df):
        if "pdbid" in frame.columns:
            frame.drop(columns=["pdbid"], inplace=True)

    return pre_df, after_df
if __name__ == "__main__":
    download_dates_and_save_csv()
