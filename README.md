<div align="center">

# 🤖 Deep Multi-Scale Heatmap Prediction for Robotic Grasping

### *DeepLabV3+ with Atrous Spatial Pyramid Pooling for Cluttered Environments*

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![GPU](https://img.shields.io/badge/GPU-RTX_4060-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Dataset](https://img.shields.io/badge/Dataset-ivalab%2Fgrasp__multiObject-blue?style=for-the-badge)](https://github.com/ivalab/grasp_multiObject)

<br>

**A dense, pixel-wise grasp detection system that predicts graspability heatmaps, grasp angles, and gripper widths at every pixel — eliminating the spatial hallucination problem of coordinate regression models.**

---

<img src="paper_results.png" width="95%" alt="Qualitative Results">

</div>

<br>

## 📊 Results at a Glance

<div align="center">

| Metric | Score | Description |
|:------:|:-----:|:------------|
| **mIoU** | `0.426` | Spatial overlap between predicted and true grasp regions |
| **Success Rate** | `63.3%` | Grasps within 10px center error AND < 15° angle error |
| **F1-Score** | `0.625` | Balanced precision (0.633) and recall (0.617) |

</div>

<br>

<div align="center">
<img src="paper_metrics.png" width="95%" alt="Training Metrics">
<p><em>Training dynamics over 60 epochs — loss convergence, mIoU growth, success rate improvement, and F1/precision/recall balance.</em></p>
</div>

---

## 🏗️ Architecture

This project uses a **DeepLabV3+** encoder-decoder architecture with **Atrous Spatial Pyramid Pooling (ASPP)** for multi-scale feature extraction.

```
Input (3 × 320 × 320)  ──►  ResNet-50 Encoder (pretrained ImageNet)
                                    │
                              ┌─────┴─────┐
                              │  Layer 1   │ ── Low-level features (80×80×256)
                              │  Layer 2   │        │
                              │  Layer 3   │        │
                              │  Layer 4   │        │
                              └─────┬─────┘        │
                                    ▼               │
                            ┌──── ASPP ────┐        │
                            │ 1×1  rate=1  │        │
                            │ 3×3  rate=6  │        │
                            │ 3×3  rate=12 │        │
                            │ 3×3  rate=18 │        │
                            │ Global Pool  │        │
                            └──────┬───────┘        │
                                   ▼                │
                           256ch @ 10×10            │
                           Upsample 8×              │
                                   ▼                ▼
                            ┌── Concat + Refine ──┐
                            │   304ch @ 80×80     │
                            │   → 256ch @ 80×80   │
                            └─────────┬───────────┘
                                      │
                               Upsample 4×
                              256ch @ 320×320
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                     Heatmap    Angle (2ch)     Width
                     Sigmoid      Tanh        Sigmoid
                      [0,1]     [-1,1]         [0,1]
                         │            │            │
                         └──── Output (4 × 320 × 320) ────┘
```

### Why ASPP?

Objects in cluttered scenes vary dramatically in size — a small pen (~20px) vs. large pliers (~100px). ASPP's parallel dilated convolutions at rates **[6, 12, 18]** give the model four different "zoom levels" plus global scene context, enabling accurate detection across all scales simultaneously.

---

## 🔬 The Problem We Solved

<div align="center">

| ❌ Before: Coordinate Regression | ✅ After: Heatmap Prediction |
|:---:|:---:|
| ResNet-50 → FC → 6 values | DeepLabV3+ → 4ch × 320×320 |
| All grasps collapse to corner (11, 39) | Grasps land on actual objects |
| No spatial awareness | Full pixel-wise spatial correspondence |
| Confidence: N/A | Confidence: 0.69 |

</div>

<div align="center">
<img src="inference_result.png" width="38%" alt="Grasp Detection">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="inference_result_heatmap_overlay.png" width="38%" alt="Heatmap Overlay">
<p><em>Left: Predicted grasp box on pliers handle. Right: Heatmap overlay showing per-object activation.</em></p>
</div>

---

## 📁 Project Structure

```
Grasp_-CNN/
├── model.py                  # DeepLabV3+ with ASPP (40.3M params)
├── dataset.py                # Heatmap ground truth generation (Gaussian σ=5)
├── train.py                  # Training loop + mIoU / SR / F1 metrics
├── inference.py              # Peak detection + grasp visualization
├── process_grasps.py         # RG-D preprocessing pipeline
├── visualize_results.py      # Paper figure generation (3×4 grid + metrics)
├── report.tex                # Full LaTeX conference report
├── paper_results.png/pdf     # Qualitative results figure
├── paper_metrics.png/pdf     # Training metrics figure
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Prerequisites

```bash
pip install torch torchvision numpy opencv-python matplotlib
```

### 2. Prepare Dataset

Download the [ivalab/grasp_multiObject](https://github.com/ivalab/grasp_multiObject) dataset, then preprocess:

```bash
python process_grasps.py
```

This creates RG-D images (Red-Green-Depth) at 320×320 with grasp labels in `processed_data/`.

### 3. Train

```bash
python train.py
```

Training runs for **60 epochs** on an RTX 4060 (~8 min). Metrics are logged every epoch:

```
Epoch 37/60 | Loss: 0.0784 / 0.2406 (train/val) | mIoU: 0.364 | SR: 0.517 | F1: 0.613 | LR: 1.25e-06
  >>> Best model saved to best_grasp_heatmap.pth
```

### 4. Inference

```bash
python inference.py
```

Outputs `inference_result.png` (grasp box) and `inference_result_heatmap_overlay.png` (heatmap glow).

### 5. Generate Paper Figures

```bash
python visualize_results.py
```

Creates publication-ready `paper_results.png/pdf` and `paper_metrics.png/pdf` at 300 DPI.

---

## 🔧 Training Configuration

| Parameter | Value |
|:----------|:------|
| GPU | NVIDIA RTX 4060 (8 GB VRAM) |
| Input Resolution | 320 × 320 × 3 (RG-D) |
| Batch Size | 4 |
| Epochs | 60 |
| Optimizer | Adam |
| Encoder LR | 1 × 10⁻⁵ (fine-tune) |
| Decoder LR | 1 × 10⁻³ (train from scratch) |
| Mixed Precision | FP16 via GradScaler |
| Frozen Layers | Stem + Layer1 + Layer2 (1.4M params) |
| Total Parameters | 40.3M (38.9M trainable) |

---

## 📐 Output Format

The model produces a **4-channel dense prediction** at every pixel:

| Channel | Meaning | Activation | Range |
|:-------:|:--------|:----------:|:-----:|
| 0 | Graspability heatmap | Sigmoid | [0, 1] |
| 1 | sin(2θ) | Tanh | [-1, 1] |
| 2 | cos(2θ) | Tanh | [-1, 1] |
| 3 | Normalized gripper width | Sigmoid | [0, 1] |

**Angle recovery:** θ = ½ · arctan2(sin2θ, cos2θ)

The trigonometric encoding eliminates angle discontinuities and naturally handles the 180° grasp symmetry.

---

## 📄 Paper

The full LaTeX report is included as [`report.tex`](report.tex). Compile with:

```bash
pdflatex report.tex
```

Or upload to [Overleaf](https://www.overleaf.com) along with the figure PNGs for instant compilation.

---

## 📚 References

- Chen et al., *"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"*, ECCV 2018
- He et al., *"Deep Residual Learning for Image Recognition"*, CVPR 2016
- Morrison et al., *"Closing the Loop for Robotic Grasping"*, RSS 2018
- Kumra & Kanan, *"Robotic Grasp Detection Using Deep CNNs"*, IROS 2017
- [ivalab/grasp_multiObject Dataset](https://github.com/ivalab/grasp_multiObject)

---

<div align="center">

**Built with ❤️ for the NMIT Bangalore Conference**

</div>
