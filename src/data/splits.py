"""
splits.py
Train/validation/test splitting for ECG beat classification.

Two splitting strategies:
    1. Inter-patient split (clinically correct)
       Beats from the same patient never appear in both train and test.
       
    2. Random split (for comparison/ablation only)
       Randomly assigns beats regardless of patient identity.
       Included to demonstrate the gap.

Usage:
    from src.data.splits import get_inter_patient_splits, get_random_splits
"""

import numpy as np
from typing import Tuple

MITBIH_TEST_INDICES = set(range(24, 48))
INCART_TEST_INDICES = set(range(60, 75))
RANDOM_SEED = 42

def get_inter_patient_split(beats: np.ndarray, labels: np.ndarray,
                            rec_ids: np.ndarray, test_indices: set,
                            val_fraction: float = 0.15,) -> Tuple:
    """
    Split beats by patient identity.

    Args:
        beats       : (N, 187) preprocessed beat segments
        labels      : (N,) AAMI class labels
        rec_ids     : (N,) integer record index per beat
        test_indices: set of record indices designated for test
        val_fraction: fraction of training beats to use for validation

    Returns:
        X_train, y_train,
        X_val, y_val,
        X_test, y_test - all np.ndarrays
    """

    test_mask = np.isin(rec_ids, list(test_indices))
    train_mask = ~test_mask

    X_test, y_test = beats[test_mask], labels[test_mask]
    X_train_full, y_train_full = beats[train_mask], labels[train_mask]

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_idx = rng.permutation(len(X_train_full))
    val_size = int(len(X_train_full) * val_fraction)

    val_idx = shuffled_idx[:val_size]
    train_idx = shuffled_idx[val_size:]

    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_random_split(beats: np.ndarray, labels: np.ndarray, test_frac: float = 0.20,
                     val_frac: float = 0.15,) -> Tuple:
    """
    Random train/val/test split ignoring patient identity.

    Args:
        beats       : (N, 187)
        labels      : (N,)
        test_frac   : fraction of all beats for test set
        val_frac    : fraction of remaining beats for validation

    Returns:
        X_train, y_train, 
        X_val, y_val,
        X_test, y_test
    """

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(beats)
    indices = rng.permutation(n)

    test_size = int(n * test_frac)
    test_indices = indices[:test_size]
    remain_indices = indices[test_size:]

    val_size = int(len(remain_indices) * val_frac)

    val_indices = remain_indices[:val_size]
    train_indices = remain_indices[val_size:]

    X_test, y_test = beats[test_indices], labels[test_indices]
    X_val, y_val = beats[val_indices], labels[val_indices]
    X_train, y_train = beats[train_indices], labels[train_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test

def print_split_summary(y_train: np.ndarray, y_val: np.ndarray,
                        y_test: np.ndarray, split_name: str = "Split",
                        ) -> None:
    """
    Print beat counts and class distribution for each split.

    Args:
        y_train, y_val, y_test  : label arrays
        split_name              : label for the printed summary
    """

    class_names = ["N", "S", "V", "F", "Q"]
    print(f"\n-- {split_name} Summary --")
    print(f"Train : {len(y_train)} beats")
    print(f"Val : {len(y_val)} beats")
    print(f"Test : {len(y_test)} beats")
    print("\n Class distribution:")
    print(f"{'Class':<6} {'Train':>8} {'Val':>8} {'Test':>8}")
    for i, name in enumerate(class_names):
        tr = int((y_train == i).sum())
        va = int((y_val == i).sum())
        te = int((y_test == i).sum())
        print(f"{name:<6} {tr:>8,} {va:>8,} {te:>8,}")


