"""
download.py
Downloads MIT-BIH Arrythmia Database and INCART Database from PhysioNet.
To run:
    python -m src.data.download
"""

import os
from pathlib import Path
from tqdm import tqdm
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

MITBIH = "mitdb"
INCART = "incartdb"

# 48 recordings in MIT-BIH
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# 75 recordings in INCART
INCART_RECORDS = [f"I{str(i).zfill(2)}" for i in range(1, 76)]

def make_dirs() -> None:
    (RAW_DATA_DIR / MITBIH).mkdir(parents=True, exist_ok=True)
    (RAW_DATA_DIR / INCART).mkdir(parents=True, exist_ok=True)

def download_record(record_name: str, db_name: str, dest_dir: Path) -> None:
    """
    Download a single PhysioNet record (signal + annotation files).

    Args:
        record_name : name in record e.g. "100" or "I01"
        db_name     : "mitdb" or "incartdb"
        dest_dir    : folder to save into
    """

    try:
        wfdb.dl_database(db_name, str(dest_dir), records=[record_name])
    except Exception as e:
        print("Warning: could not download {record_name} from {db_name}: {e}")

def download_mitbih() -> None:
    """
    Download all 48 MIT-BIH recordings.
    """

    dest = RAW_DATA_DIR / MITBIH
    print(f"\nDownloading MIT-BIH ({len(MITBIH_RECORDS)} records) to {dest}")

    for record in tqdm(MITBIH_RECORDS, desc="MIT-BIH"):
        download_record(record, MITBIH, dest)
        
def download_incart() -> None:
    """
    Download all 75 INCART recordings.
    """

    dest = RAW_DATA_DIR / INCART
    print(f"\nDownloading INCART ({len(INCART_RECORDS)} recoreds) to {dest}")

    for record in tqdm(INCART_RECORDS, desc="INCART"):
        download_record(record, "incartdb", dest)

def verify_download(db_name: str, records: list, dest_dir: Path) -> None:
    """
    Check that every record has at least one header file.
    Prints summary of missing records if any.

    Args:
        db_name  : used for summary label
        records  : list of expected record names
        dest_dir : folder where files should exist
    """

    missing = []
    for record in records:
        if not (dest_dir / f"{record}.hea").exists():
            missing.append(record)

    if not missing:
        print(f"{db_name}: All {len(records)} records were downloaded successfully.")
    else:
        print(f"{db_name}: {len(missing)} records are missing: {missing}")

def main() -> None:
    make_dirs()
    download_mitbih()
    download_incart()

    verify_download(MITBIH, MITBIH_RECORDS, RAW_DATA_DIR / MITBIH)
    verify_download(INCART, INCART_RECORDS, RAW_DATA_DIR / INCART)
    print("\nData is in:", RAW_DATA_DIR)

if __name__ == "__main__":
    main()
        




