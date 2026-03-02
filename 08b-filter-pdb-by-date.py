import requests
import csv
import time

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
if __name__ == "__main__":
    download_dates_and_save_csv()
