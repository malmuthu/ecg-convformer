"""
transfer.py
Cross-dataset transfer evaluation: MIT-BIH -> INCART.

Two evaluation modes:
    1. Zero-shot    : use MIT-BIH weights directly on INCART (no retraining)
    2. Fine-tuned   : freeze conv stem, fine-tune transformer + classifier on 10% of INCART, evaluate on remaining 90%

This tests whether the covnolutional stem learned transferable ECG features, or just dataset-specific noise.

Usage:
    python -m src.evaluation.transfer --checkpoint results/checkpoints/best.pt --mode both
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.models.convformer import ECGConvFormer
from src.data.dataset import ECGDataset, load_dataset
from src.evaluation.metrics import compute_all_metrics, bootstrap_ci, print_metrics, collect_predictions
from src.training.losses import get_weighted_cross_entropy
from src.training.scheduler import get_optimizer, get_cosine_schedule_with_warmup

PROJ_ROOT = Path(__file__).resolve().parents[2]

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_incart_splits(finetune_fraction: float = 0.10) -> tuple:
    """
    Load INCART data and split into fine-tune and test sets.

    Args:
        finetune_fraction   : fraction of INCART used for fine-tuning

    Returns:
        finetune_dataset    : ECGDataset - 10% of INCART for fine-tuning
        test_dataset        : ECGDataset - 90% of INCART for evaluation
        class_weights       : torch.Tensor for weighted loss
    """

    beats, labels, ids = load_dataset(db="incart")

    X_ft, X_test, y_ft, y_test = train_test_split(beats, labels, test_size=(1 - finetune_fraction), random_state=42, stratify=labels)

    ft_dataset = ECGDataset(X_ft, y_ft)
    test_dataset = ECGDataset(X_test, y_test)

    class_weights = ft_dataset.get_class_weights()

    return ft_dataset, test_dataset, class_weights

def zero_shot_eval(model: nn.Module, test_dataset: ECGDataset, device: torch.device) -> dict:
    """
    Evaluate MIT-BIH trained model directly on INCART test set.

    Args:
        model       : trained ECGConvformer (MIT-BIH weights)
        test_dataset: INCART test ECGDataset
        device      : torch.device

    Returns:
        metrics dict from compute_all_metrics()
    """

    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    labels, preds, probs = collect_predictions(model, test_loader, device)

    metrics = compute_all_metrics(labels, preds, probs)

    print("\n-- Zero-Shot Transfer (MIT-BIH -> INCART) --")
    print_metrics(metrics=metrics)

    return metrics

def fine_tune(model: nn.Module, finetune_dataset: ECGDataset, class_weights: torch.Tensor, device: torch.device, epochs: int = 10, lr: float = 1e-4) -> nn.Module:
    """
    Fine-tune only the transfer encoder and classifer head on INCART.

    Args:
        model               : pretrained ECGConvFormer
        finetune_dataset    : 10% of INCART
        class_weights       : for weighted loss
        device              : torch.device
        epochs              : fine-tuning epochs
        lr                  : fine-tuning learning rate

    Returns:
        fine-tuned model
    """

    for param in model.conv_stem.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Fine-tuning {trainable:,} parameteres (transformer + classifier)")

    loader = DataLoader(finetune_dataset, batch_size=64, shuffle=True)
    criterion = get_weighted_cross_entropy(class_weights.to(device))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=2, max_epochs=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for beats, labels in loader:
            beats = beats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(beats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()

        epoch_loss = total_loss / len(loader)
        print(f"Fine-tune epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    return model

def fine_tune_eval(model: nn.Module, test_dataset: ECGDataset, device: torch.device) -> dict:
    """
    Evaluate fine-tuned model on INCART test set.

    Args:
        model       : fine-tuned ECGConvFormer
        test_dataset: INCART test ECGDataset
        device      : torch.device

    Returns:
        metrics dict
    """

    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    labels, preds, probs = collect_predictions(model, test_loader, device)

    metrics = compute_all_metrics(labels, preds, probs)

    print("\n-- Fine-Tuned Transfer Transfer (MIT-BIH -> INCART, 10% fine-tune) --")
    print_metrics(metrics=metrics)

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both", choices=["zero-shot", "fine-tune", "both"])
    parser.add_argument("--finetune_fraction", type=float, default=0.10)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = get_device()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = ECGConvFormer(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    # Load INCART
    ft_dataset, test_dataset, class_weights = load_incart_splits(args.finetune_fraction)

    if args.mode in ("zero_shot", "both"):
        zero_shot_metrics = zero_shot_eval(model, test_dataset, device)

    if args.mode in ("fine_tune", "both"):
        model = fine_tune(model, ft_dataset, class_weights, device, epochs=args.finetune_epochs, lr=args.finetune_lr)
        ft_metrics = fine_tune_eval(model, test_dataset, device)

    if args.mode == "both":
        print("\n-- Transfer Summary --")
        print(f"Zero-shot Macro F1: {zero_shot_metrics['macro_f1']:.4f}")
        print(f"Fine-tuned Macro F1: {ft_metrics['macro_f1']:.4f}")
        improvement = ft_metrics['macro_f1'] - zero_shot_metrics['macro_f1']
        print(f"Improvement : +{improvement:.4f}")

if __name__ == "__main__":
    main()