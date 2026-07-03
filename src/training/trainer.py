"""
trainer.py
Main training loop for ECG-ConvFormer.

Usage:
    python -m src.training.trainer --config configs/default.yaml

"""

import argparse
import os
import time
import torch
import torch.nn as nn
import wandb
import yaml

from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.convformer import ECGConvFormer
from src.models.baseline import ConvOnlyBaseline
from src.data.dataset import ECGDataset, load_dataset
from src.data.splits import get_inter_patient_split, get_random_split, print_split_summary, MITBIH_TEST_INDICES
from src.training.losses import get_weighted_cross_entropy, FocalLoss
from src.training.scheduler import get_optimizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def get_device() -> torch.device:
    """
    Return best available device - CUDA, MPS, or CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_dataloaders(config: dict) -> tuple:
    """
    Load data, split, build ECGDatasets and DataLoaders.

    Args:
        config  : training config dictionary from YAML

    Returns:
        train_loader, val_loader, test_Loader, class_weights
    """

    beats, labels, rec_ids = load_dataset(db="mit")

    if config["split"] == "inter_patient":
        X_train, y_train, X_val, y_val, X_test, y_test = get_inter_patient_split(beats, labels, rec_ids, MITBIH_TEST_INDICES)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = get_random_split(beats, labels)

    print_split_summary(y_train, y_val, y_test, config["split"])

    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    class_weights = train_dataset.get_class_weights()

    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights

def build_model(config: dict) -> nn.Module:
    """
    Instantiate model based on config.

    Args:
        config  : training config dictionary

    Returns:
        nn.Module - either ECGConvFormer or ConvOnlyBaseline
    """

    if config["model"] == "convformer":
        return ECGConvFormer(d_model=config["d_model"], n_heads=config["n_heads"], d_ff=config["d_ff"], n_layers=config["n_layers"], dropout=config["dropout"])
    elif config["model"] == "baseline":
        return ConvOnlyBaseline(d_model=config["d_model"], dropout=config["dropout"])
    else:
        raise ValueError(f"Unknown model: {config['model']}")

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, scaler: torch.cuda.amp.GradScaler, device: torch.device) -> tuple:
    """
    Run one full training epoch.

    Args:
        model       : the model being trained
        loader      : training DataLoader
        optimizer   : AdamW optimizer
        criterion   : loss function
        scaler      : GradScaler for mixed precision
        device      : torch.device

    Returns:
        avg_loss    : float - mean loss over all batches
        macro_f1    : float - macro F1 over all batches
    """   

    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    use_amp = (device.type == "cuda")

    for beats, labels in loader:
        beats = beats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(beats)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, macro_f1

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    """
    Evaluate model on val or test set.

    Args:
        model       : the model to evaluate
        loader      : val or test DataLoader
        criterion   : loss function
        device      : torch.device

    Returns:
        avg_loss    : float
        macro_f1    : float
    """

    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for beats, labels in loader:
        beats = beats.to(device)
        labels = labels.to(device)

        logits = model(beats)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, macro_f1

def train(config: dict) -> None:
    """
    Full training loop with early stopping and checkpointing.

    Args:
        config  : training configuration dictionary
    """

    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(config)
    class_weights = class_weights.to(device)

    # Model
    model = build_model(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Loss
    if config["loss"] == "focal":
        criterion = FocalLoss(gamma=2.0, class_weights=class_weights)
    else:
        criterion = get_weighted_cross_entropy(class_weights)

    # Optimizer and scheduler
    optimizer = get_optimizer(model, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=config["warmup_epochs"], max_epochs=config["max_epochs"])

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # W&B
    wandb.init(project="ecg-convformer", config=config, name=config.get("run_name", "run"))

    # Early stopping state
    best_val_f1 = 0.0
    patience_count = 0
    best_ckpt_path = CHECKPOINT_DIR / f"{config.get('run_name', 'best')}.pt"

    for epoch in range(config["max_epochs"]):
        train_loss, train_f1 = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, scaler=scaler, device=device)
        val_loss, val_f1 = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{config['max_epochs']} | "
            f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        wandb.log({"train/loss": train_loss, "train/f1": train_f1,
                   "val/loss": val_loss, "val/f1": val_f1,
                   "lr": current_lr}, step=epoch)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_f1": val_f1, "config": config}, best_ckpt_path)
        else:
            patience_count += 1

        if patience_count >= config["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("\nLoading best checkpoint for test evaluation...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_f1 = evaluate(model=model, loader=test_loader, criterion=criterion, device=device)
    print(f"Test Macro F1: {test_f1:.4f}")
    wandb.log({"test/f1": test_f1, "test/loss": test_loss})
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)

if __name__ == "__main__":
    main()


    


