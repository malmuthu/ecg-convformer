"""
integrated_gradients.py
Integrated Gradients interpretability for ECG-ConvFormer.

Computes per-sample attribution scores using Captum's IntegratedGradients, showing which
timesteps in the ECG beat most influenced the prediction.

Uses a flat-line (zero) baseline.

Usage:
    from src.interpretability.integrated_gradients import compute_attributions
"""

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

def compute_attributions(model: nn.Module, beat: torch.Tensor, target_class: int, device: torch.device, n_steps: int = 50) -> np.ndarray:
    """
    Compute Integrated Gradients attribution for a single beat.

    Args:
        model       : trained model in eval mode
        beat        : shape (1, 1, 187) - single beat, batch dim included
        target_class: the class to explain
        device      : torch.device
        n_steps     : number of interpolation steps between baseline and input

    Returns:
        attributions: np.ndarray shape (187,) - one score per timestep
    """

    model.eval()
    beat = beat.to(device)

    ig = IntegratedGradients(model)

    baseline = torch.zeros_like(beat)

    attributions = ig.attribute(beat, baselines=baseline, target=target_class, n_steps=n_steps)

    attributions = attributions.squeeze().cpu().detach().numpy()

    return attributions

def compute_attributions_batch(model: nn.Module, beats: torch.Tensor, targets: torch.Tensor, device: torch.device, n_steps: int = 50) -> np.ndarray:
    """
    Compute Integrated Gradients for a batch of beats at once.
    More efficient than looping compute_attributions() one beat at a time.

    Args:
        model   : trained model in eval mode
        beats   : shape (batch, 1, 187)
        targets : shape (batch,)
        device  : torch.device
        n_steps : interpolation steps

    Returns:
        attributions    : np.ndarray shape (batch, 187)
    """

    model.eval()
    beats = beats.to(device)
    targets = targets.to(device)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(beats)

    attributions = ig.attribute(beats, baselines=baseline, target=targets, n_steps=n_steps)
    attributions = attributions.squeeze(1).cpu().detach().numpy()

    return attributions

def select_correctly_classified_examples(model: nn.Module, test_dataset, device: torch.device, n_per_class: int = 5) -> dict:
    """
    Find n_per_class correctly classified beats for each AAMI class.

    Args:
        model       : trained model in eval mode
        test_dataset: ECGDataset
        device      : torch.device
        n_per_class : how many correct examples to find per class

    Returns:
        dict mapping class index
    """

    model.eval()
    selected = {c: [] for c in range(5)}

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            beat, label = test_dataset[idx]
            label = label.item()

            if len(selected[label]) >= n_per_class:
                continue

            beat_batch = beat.unsqueeze(0).to(device)
            logits = model(beat_batch)
            pred = logits.argmax(dim=1).item()

            if pred == label:
                selected[label].append(idx)

            if all(len(v) >= n_per_class for v in selected.values()):
                break

    return selected