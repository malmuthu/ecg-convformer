# ECG-ConvFormer

A hybrid convolutional-transformer model for inter-patient ECG arrhythmia classification, class imbalance analysis, Integrated Gradients interpretability, and cross-dataset transfer evaluation.

---

## Overview

ECG-ConvFormer classifies individual heartbeats from raw ECG signal into one of five AAMI EC57 standard arrhythmia categories (Normal, Supraventricular, Ventricular, Fusion, Unknown). The model combines a residual convolutional stem which extracts local waveform morphology such as QRS shape and P wave structure — with a transformer encoder that captures global relationships across the beat.

Combining convolutional and transformer components for ECG classification has been explored in prior work (see [References](#references)). This project is a fully reproducible implementation with a controlled ablation against a convolution-only baseline, clinically-validated interpretability via Integrated Gradients, and a cross-dataset transfer evaluation.

---

## Dataset

**[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)** (PhysioNet) — 48 two-channel ECG recordings from 47 patients, sampled at 360 Hz, with beat-level annotations from two independent cardiologists.

**[INCART Database](https://physionet.org/content/incartdb/1.0.0/)** (PhysioNet) — 75 recordings used for cross-dataset transfer evaluation. Same AAMI beat-class labels as MIT-BIH, different patient population and recording hardware.

---

## Architecture

```
Input: raw ECG beat (1, 187)
        │
        ▼
┌───────────────────────────┐
│   CONVOLUTIONAL STEM      │
│   3× Residual Conv1D      │   ← local morphology: QRS shape, P/T wave structure
│   blocks (kernel 7→5→3)   │
│   187 → 94 → 47 timesteps │
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│   TRANSFORMER ENCODER     │
│   Sinusoidal pos. encoding│   ← global context: relationships between
│   2× Pre-LN encoder blocks│     P wave, QRS, and T wave regions
│   4-head self-attention   │
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│   Global Average Pooling  │
│   LayerNorm → FC → GELU   │
│   → Dropout → FC          │
└───────────┬───────────────┘
            ▼
      5-class logits
   (N, S, V, F, Q)
```

Full implementation: [`src/models/convformer.py`](src/models/convformer.py), [`src/models/conv_stem.py`](src/models/conv_stem.py), [`src/models/transformer_encoder.py`](src/models/transformer_encoder.py)

---

## Results

### Primary model — ECG-ConvFormer (inter-patient split)

**Per-class results:**

| Class | F1 | Precision | Recall |
|---|---|---|---|
| N | `0.8080` | `0.9247` | `0.7175` |
| S | `0.0604` | `0.0464` | `0.0864` |
| V | `0.6473` | `0.5008` | `0.9153` | 
| F | `0.0109` | `0.0385` | `0.0064` | 
| Q | `0.4061` | `0.2552` | `0.9945` |

### Ablation — does the transformer add value?

| Model | Parameters | Macro F1 (test) |
|---|---|---|
| ConvOnly baseline (conv stem + classifier, no transformer) | `0.405` |
| ECG-ConvFormer (full) | `0.387` |

> `The transformer encoder did not meaningfully improve over the conv-only baseline (macro F1: 0.387 vs 0.405), suggesting that the convolutional stem captures most of the discriminative signal and the transformer's global context adds limited value at this sequence length.`

### Cross-dataset transfer — MIT-BIH → INCART

| Evaluation mode | Macro F1 |
|---|---|
| Zero-shot (no INCART training) | `0.2313` |
| Fine-tuned (10% of INCART, transformer + classifier only) | `0.4283` |
| Improvement | `+0.1970` |

The convolutional stem was frozen during fine-tuning, testing whether it learned transferable low-level ECG features versus dataset-specific noise. 

---

## Interpretability

Integrated Gradients ([Sundararajan et al., 2017](https://arxiv.org/abs/1703.01365)) was used to attribute each prediction to specific timesteps in the input signal, using a zero-baseline (flat line) reference. Unlike Grad-CAM, IG is architecture-agnostic and satisfies completeness and sensitivity axioms, making it appropriate for a transformer-containing model.

---

## Repository Structure

```
ecg-convformer/
├── src/
│   ├── data/              # download, preprocessing, dataset, splits
│   ├── models/             # ConvStem, TransformerEncoder, ConvFormer, baseline
│   ├── training/            # losses, LR scheduler, training loop
│   ├── evaluation/          # metrics, evaluation, cross-dataset transfer
│   └── interpretability/    # Integrated Gradients, visualization
├── configs/                 # YAML experiment configurations
├── results/
│   ├── figures/              # interpretability visualizations
│   ├── checkpoints/           # trained model weights (gitignored)
│   └── metrics/                # saved JSON evaluation results
└── requirements.txt
```

---

## Reproduction

```bash
# 1. Environment
conda create -n ecg-convformer python=3.11
conda activate ecg-convformer
pip install -r requirements.txt

# 2. Data
python -m src.data.download        # downloads MIT-BIH and INCART from PhysioNet
python -m src.data.preprocess      # segments beats, applies AAMI mapping, normalizes

# 3. Train
python -m src.training.trainer --config configs/default.yaml          # full ConvFormer
python -m src.training.trainer --config configs/baseline.yaml         # ablation baseline
python -m src.training.trainer --config configs/random_split.yaml     # random-split comparison

# 4. Evaluate
python -m src.evaluation.evaluate --checkpoint results/checkpoints/convformer_interpatient_v1.pt

# 5. Cross-dataset transfer
python -m src.evaluation.transfer --checkpoint results/checkpoints/convformer_interpatient_v1.pt --mode both
```

---

## References

1. Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
2. He et al., ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385), CVPR 2016.
3. Sundararajan et al., ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/abs/1703.01365), ICML 2017.
4. de Chazal et al., ["Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features"](https://ieeexplore.ieee.org/document/1306572), IEEE TBME 2004 — establishes the inter-patient evaluation paradigm and DS1/DS2 split convention used here.
5. AAMI, ANSI/AAMI EC57:2012, *Testing and Reporting Performance Results of Cardiac Rhythm and ST Segment Measurement Algorithms.*
6. Moody & Mark, ["The Impact of the MIT-BIH Arrhythmia Database"](https://ieeexplore.ieee.org/document/932724), IEEE Eng. in Medicine and Biology 2001.

