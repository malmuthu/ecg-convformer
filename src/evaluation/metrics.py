"""
metrics.py
Evaluation metrics for ECG beat classification.

Computes:
    - Per-class F1, precision, recall
    - Macro F1 
    - AUROC
    - Confusion matrix
    - Bootstrap confidence intervals on macro F1

Usage:
    from src.evaluation.metrics import compute_all_metrics
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import torch

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

def compute_all_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """
    Compute full evaluation metric suite.

    Args:
        labels  : shape (N,) - true integer class labels
        preds   : shape (N,) - predicted integer class labels
        probs   : shape (N, 5) - softmax probabilities per class

    Returns:
        dict containig all computed metrics
    """

    metrics = {}

    metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    metrics["micro_f1"] = f1_score(labels, preds, average="micro", zero_division=0)
    metrics["per_class_f1"] = f1_score(labels, preds, average=None, zero_division=0)
    metrics["per_class_precision"] = precision_score(labels, preds, average=None, zero_division=0)
    metrics["per_class_recall"] = recall_score(labels, preds, average=None, zero_division=0)

    # Compute AUROC
    try:
        metrics["auroc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"Warning: AUROC failed.")
        metrics["auroc"] = None

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(labels, preds)

    return metrics

def bootstrap_ci(labels: np.ndarray, preds: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple:
    """
    Bootstrap 95% confidence interval on macro F1.

    Resamples the test set with replacement n_bootstrap times, computes macro F1 on each resample

    Args:
        labels      : true labels
        preds       : predicted labels
        n_bootstrap : number of bootstrap resamples
        ci          : confidence level (default 0.95)
        seed        : random seed

    Returns:
        (mean_f1, lower_bound, upper_bound)
    """

    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = []

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        score = f1_score(labels[idx], preds[idx], average="macro", zero_division=0)
        scores.append(score)

    scores = np.array(scores)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)

    return float(scores.mean()), float(lower), float(upper)

def print_metrics(metrics: dict, ci: tuple = None) -> None:
    """
    Print a formatted metrics summary table.

    Args:
        metrics : dict from compute_all_metrics()
        ci      : optional tuple from bootstrap_ci
    """

    print("\n" + "="*55)
    print("Evaluation Results")
    print("="*55)
    print(f"Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"Micro F1    : {metrics['micro_f1']:.4f}")
    if metrics.get("auroc"):
        print(f"AUROC   : {metrics['auroc']:.4f}")
    if ci:
        print(f"95% CI  : [{ci[1]:.4f}, {ci[2]:.4f}]")
    print("\n Per-Class Results:")
    print(f"{'Class':<6} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    print(" " + "-"*35)
    for i, name in enumerate(CLASS_NAMES):
        f1 = metrics["per_class_f1"][i]
        pr = metrics["per_class_precision"][i]
        rec = metrics["per_class_recall"][i]
        print(f" {name:<6} {f1:>7.4f} {pr:>10.4f} {rec:>8.4f}")
    print("\n Confusion Matrix:")
    print("Predicted ->")
    header = " " + "".join(f"{n:>6}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "".join(f"{v:>6}" for v in row)
        print(f"{CLASS_NAMES[i]} {row_str}")
    print("="*55)

def collect_predictions(model: torch.nn.Module, loader, device: torch.device) -> tuple:
    """
    Run model on a DataLoader and collect all predictions and probabilities.

    Args:
        model   : trained nn.Module in eval mode
        loader  : DataLoader
        device  : torch.device

    Returns:
        labels  : np.ndarray (N,)
        preds   : np.ndarray (N,)
        probs   : np.ndarray (N, 5)
    """

    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for beats, labels in loader:
            beats = beats.to(device)
            logits = model(beats)

            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            all_labels.append(labels.numpy())
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
    
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    return labels, preds, probs