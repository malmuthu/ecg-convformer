"""
dataset.py
PyTorch Dataset class for ECG beat classification.
Provides indivisual beats + labels to the DataLoader.
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA_DIR = PROJECT_ROOT / "data" / "processed"

class ECGDataset(Dataset):
    """
    Dataset for ECG beat classification.

    Each item is a single heartbeat segment of shape (1, 187)
    and an integer class label in {0, 1, 2, 3, 4}.

    The channel dimension (1,) is added for Conv1D layers

    Args:
        beats       : np.ndarray shape (N, 187) - preprocessed beat segments
        labels      : np.ndarray shape (N,) - AAMI class labels
        transform   : optional applied to each beat tensor
    """

    def __init__(self, beats: np.ndarray, labels: np.ndarray, transform=None,):
        self.beats = torch.tensor(beats, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the number of beats in the dataset.
        """
        return len(self.beats)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Return one beat and its label.

        Args:
            idx : integer index

        Returns:
            beat    : torch.Tensor shape (1, 187)
            label   : torch.Tensor scalar (long)
        """
        beat = self.beats[idx]
        if self.transform is not None:
            beat = self.transform(beat)
        return beat, self.labels[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss

        Classes with fewer samples get higher weight so the model
        does not ignore minority classes.

        Returns:
            weights : torch.Tensor shape (5,) - one weight per class
        """

        labels_np = self.labels.numpy()
        total_beats = len(labels_np)
        weights = np.zeros(5, dtype=np.float32)

        for cls in range(5):
            count = int((labels_np == cls).sum())
            if count > 0:
                weight = total_beats / (5 * count)
            else:
                weight = 0.0
            weights[cls] = weight

        return torch.tensor(weights)
    
def load_dataset(db: str) -> tuple:
    """
    Load preprocessed arrays from disk for a given database.

    Args:
        db  : "mit for MIT-BIH or "incart" for INCART

    Returns:
        beats   : np.ndarray (N, 187)
        labels  : np.ndarray (N,)
        rec_ids : np.ndarray (N,)
    """

    beats = np.load(PROJECT_DATA_DIR / f"{db}_beats.npy")
    labels = np.load(PROJECT_DATA_DIR / f"{db}_labels.npy")
    rec_ids = np.load(PROJECT_DATA_DIR / f"{db}_ids.npy")
    return beats, labels, rec_ids

def get_class_distribution(labels: np.ndarray) -> dict:
    """
    Print and return the count of each AAMI class in a label array.

    Args:
        labels  : np.ndarray of integer class labels

    Returns:
        dict mapping class name -> count
    """

    class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
    distribution = {}
    for c, name in class_names.items():
        count = int(np.sum(labels == c))
        distribution[name] = count
        print(f"Class {name}: {count} beats")
    return distribution
    


