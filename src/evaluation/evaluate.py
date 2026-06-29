"""
evaluate.py
Standalone evaluation script for trained ECG-ConvFormer checkpoints.

Loads a saved checkpoint, runs inference on the test set, computes all metrics, prints a summary,
and saves results to results/metrics/.

Usage:
    python -m src.evaluation.evaluate --checkpoint results/checkpoints/best.pt --split inter_patient
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch

from src.models.convformer import ECGConvFormer
from src.models.baseline import ConvOnlyBaseline
from src.data.dataset import ECGDataset, load_dataset
from src.data.splits import get_inter_patient_split, get_random_split, MITBIH_TEST_INDICES
from src.evaluation.metrics import compute_all_metrics, bootstrap_ci, print_metrics, collect_predictions

PROJ_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJ_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load a saved checkpoint and reconstruct the model.

    Args:
        checkpoint_path : path to .pt checkpoint file
        device          : torch.device

    Returns:
        model   : nn.Module loaded with saved weights
        config  : training config dict saved inside checkpoint
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    if config["model"] == "convformer":
        model = ECGConvFormer(d_model=config["d_model"], n_heads=config["n_heads"], d_ff=config["d_ff"], n_layers=config["n_layers"], dropout=config["dropout"])
    elif config["model"] == "baseline":
        model = ConvOnlyBaseline(d_model=config["d_model"], dropout=config["dropout"])
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)

    return model, config

def get_test_loader(config: dict, batch_size: int = 256):
    """
    Build the test DataLoader from saved config split settings.

    Args:
        config      : training config dict
        batch_size  : inference batch size 

    Returns:
        test_loader : DataLoader
        y_test      : np.ndarray of true labels
    """

    from torch.utils.data import DataLoader

    beats, labels, ids = load_dataset(db="mit")

    if config["split"] == "inter_patient":
        _, _, _, _, X_test, y_test = get_inter_patient_split(beats, labels, ids, MITBIH_TEST_INDICES)
    else:
        _, _, _, _, X_test, y_test = get_random_split(beats, labels)

    test_dataset = ECGDataset(X_test, y_test)

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False), y_test

def save_metrics(metrics: dict, ci: tuple, run_name: str) -> None:
    """
    Save metrics to a JSON file in results/metrics/.

    Converts numpy arrays to lists for JSON serialization.

    Args:
        metrics     : dict from compute_all_metrics()
        ci          : tuple from bootstrap_ci
        run_name    : used as filename
    """

    results = {}

    results["macro_f1"] = float(metrics["macro_f1"])
    results["micro_f1"] = float(metrics["micro_f1"])
    results["auroc"] = float(metrics["auroc"]) if metrics["auroc"] else None
    results["per_class_f1"] = metrics["per_class_f1"].tolist()
    results["bootstrap_ci"] = {"mean": ci[0], "lower": ci[1], "upper": ci[2]}
    results["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    path = METRICS_DIR / f"{run_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="eval")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, config = load_checkpoint(args.checkpoint, device)
    print(f"Loaded: {config.get('run_name', 'unknown')} | "
          f"Model: {config['model']} | "
          f"Split: {config['split']}")
    
    test_loader, y_test = get_test_loader(config)

    print("Running inference...")
    labels, preds, probs = collect_predictions(model, test_loader, device)

    metrics = compute_all_metrics(labels, preds, probs)
    ci = bootstrap_ci(labels, preds)

    print_metrics(metrics, ci)
    save_metrics(metrics, ci, args.run_name)

if __name__ == "__main__":
    main()