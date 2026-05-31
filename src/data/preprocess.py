"""
preprocess.py
Converts raw PhysioNet recordings into segmented, normalized beat arrays.

Run:
    python -m src.data.preprocess
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROJECT_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MITBIH = "mitdb"
INCART = "incartdb"

MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

INCART_RECORDS = [f"I{str(i).zfill(2)}" for i in range(1, 76)]

# Maps raw wfdb annotation symbols to 5 AAMI classes (0-4)
# N=0, S=1, V=2, F=3, Q=4
AAMI_MAP = {
    "N": 0, ".": 0, "L":0, "R":0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "Q": 4, "/": 4, "f": 4,
}

BEAT_WINDOW = 187   # total samples per beat
BEAT_BEFORE = 90    # samples before R-peak
BEAT_AFTER = BEAT_WINDOW - BEAT_BEFORE - 1  # samples after R-peak

def load_record(record_path: str):
    """
    Load ECG signal and annotations for one record.

    Returns:
        signal      : np.ndarray (n_samples, n_leads)
        annotations : wfdb.Annotation object
    """

    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, "atr")
    return record.p_signal, annotations

def extract_beats(signal: np.ndarray, annotations) -> tuple:
    """
    Slice fixed-length windows around each annotated R-peak.
    Skips beats too close to signal boundaries.

    Args:
        signal      : np.ndarray (n_samples, n_leads)
        annotations : wfdb.Annotation object

    Returns:
        beats   : np.ndarray (n_valid_beats, BEAT_WINDOW)
        labels  : np.ndarray (n_valid_beats,) dtype int
    """

    # Use lead 0 only (MLII in MIT-BIH)
    signal_1d = signal[:, 0]
    n_samples = len(signal_1d)

    beats, labels = [], []

    for idx, symbol in zip(annotations.sample, annotations.symbol):
        # Skip beats too close to start or end of recording
        if idx < BEAT_BEFORE or idx + BEAT_AFTER >= n_samples:
            continue

        # Skip symbols not in our AAMI map (noise, non-beat markers)
        if symbol not in AAMI_MAP:
            continue

        window = signal_1d[idx - BEAT_BEFORE: idx + BEAT_AFTER + 1]
        beats.append(window)
        labels.append(AAMI_MAP[symbol])

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)

def normalize_beats(beats: np.ndarray) -> np.ndarray:
    """
    Per-beat z-score normalization.
    Each beat independently normalized to zero mean, unit variance.

    Args:
        beats : (N, BEAT_WINDOW)
    
    Returns:
        normalized beats : (N, BEAT_WINDOW)
    """

    mean = beats.mean(axis=1, keepdims=True)    # shape (N, 1)
    std = beats.std(axis=1, keepdims=True)      # shape (N, 1)
    return (beats - mean) / (std + 1e-8)

def process_data(db_name: str, records: list) -> tuple:
    """
    Process all records in one database.

    Returns:
        all_beats   : np.ndarray (N, BEAT_WINDOW)
        all_labels  : np.ndarray (N, )
        all_ids     : np.ndarray (N, )
    """

    all_beats, all_labels, all_ids = [], [], []

    for idx, name in enumerate(tqdm(records, desc=db_name)):
        record_path = str(RAW_DATA_DIR / db_name / name)

        try:
            signal, annotations = load_record(record_path)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue

        beats, labels = extract_beats(signal, annotations)

        if len(beats) == 0:
            continue

        beats = normalize_beats(beats)

        ids = np.full(len(beats), idx, dtype=np.int64)

        all_beats.append(beats)
        all_labels.append(labels)
        all_ids.append(ids)

    if not all_beats:
        return np.array([]), np.array([]), np.array([])
    
    return (np.concatenate(all_beats), np.concatenate(all_labels), np.concatenate(all_ids))

def save_arrays(db_name: str, beats, labels, ids) -> None:
    """
    Save processed arrays to data/processed/
    """

    PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prefix = db_name.replace("db", "")
    np.save(PROJECT_DATA_DIR / f"{db_name}_beats.npy", beats)
    np.save(PROJECT_DATA_DIR / f"{db_name}_labels.npy", labels)
    np.save(PROJECT_DATA_DIR / f"{db_name}_ids.npy", ids)
    print(f"Saved {len(beats):,} beats to data/processed/{db_name}_*.npy")

def main():
    print("Processing MIT-BIH")
    beats, labels, ids = process_data(MITBIH, MITBIH_RECORDS)
    save_arrays(MITBIH, beats, labels, ids)

    print("\nProcessing INCART")
    beats, labels, ids = process_data(INCART, INCART_RECORDS)
    save_arrays(INCART, beats, labels, ids)

if __name__ == "__main__":
    main()



