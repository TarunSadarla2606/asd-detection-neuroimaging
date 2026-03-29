# ASD Detection using sMRI and Deep Learning

> *Detecting Autism Spectrum Disorder from structural MRI brain scans using CNNs and Vision Transformers.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Institution](https://img.shields.io/badge/MIT%20Manipal-B.Tech%20Final%20Year-8B0000)]()

---

## Overview

This repository contains the complete code and documentation for my **B.Tech final year project** (2022–2023) at Manipal Institute of Technology, submitted in partial fulfilment of the degree of Bachelor of Technology in Computer Science & Engineering, under the guidance of **Dr. J. Andrew**.

It was my **first major deep learning project** and my **first project in medical imaging**. The goal was to develop and compare deep learning approaches for detecting Autism Spectrum Disorder (ASD) from structural MRI (sMRI) brain scans, addressing the limitations of current subjective, behaviour-based diagnostic methods.

The project explored three model families across two frameworks, and continued developing beyond the formal submission — including pretrained ViT experiments and an XAI (LIME) explainability attempt.

---

## Problem Statement

Autism Spectrum Disorder (ASD) is a complex neurological condition currently diagnosed through subjective behavioural assessments. This process is time-consuming, inconsistent across clinicians, and often delayed — particularly in early childhood when timely intervention matters most. Neuroimaging-based approaches, coupled with deep learning, offer a path toward objective, reproducible, and biologically grounded diagnosis.

---

## Dataset — ABIDE-I

- **Source:** Autism Brain Imaging Data Exchange ([ABIDE-I](https://fcon_1000.projects.nitrc.org/indi/abide/)), accessed via COINS
- **Scale:** 1,112 subjects from 17 international research centres; median age 14.7 years (range 7–64)
- **After cleaning** (removed missing/unclear sMRI and "Unknown" label subjects): **1,067 subjects**
- **Format pipeline:** NIfTI (.nii.gz) → DICOM → PNG slices
- **After slice extraction** (removing empty/malformed slices): **100,510 usable DICOM slices**
- **Splits:** Train: 2,400 | Validation: 300 | Test: 300 (slice-level)

---

## Preprocessing Pipelines

### CNN Preprocessing
```
NIfTI → DICOM → PNG → Resize (224×224) → 1-channel to 3-channel → Mean/Std Normalization
```

### ViT Preprocessing
```
NIfTI → DICOM → PNG → Canny Edge Detection → Edge-based Crop → Resize (224×224) →
1-channel to 3-channel → Min-Max Normalization → Patchify (16×16 patches → 196 patches/image)
```

The ViT pipeline adds edge-based cropping to remove empty background and patchification to match the Vision Transformer's input structure.

---

## Models

### 1. Custom 5-Layer CNN

A purpose-built convolutional network for binary ASD/TC (Typical Control) classification:

```
Conv(16) + LeakyReLU → MaxPool(2×2) → Dropout(20%)
Conv(32) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(20%)
Conv(64) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(20%)
Conv(128) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(20%)
Conv(256) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(20%)
Flatten → FC(100) → LeakyReLU → Output(2)
```

Trained with three optimizers: **Adam**, **NAdam**, and **RMSprop** — each with LR=0.001, 50 epochs, batch size 64.

### 2. Skip-Connected CNN

The same 5-layer architecture with a **residual skip connection** added between the output of layer 1 and the input of layer 3. This addresses the vanishing gradient problem and encourages feature reuse across layers. Trained with the same three optimizers.

### 3. Vision Transformer — Keras (TensorFlow)

A custom ViT implemented in TensorFlow/Keras, following the standard architecture:
- 6 Transformer blocks with multi-head self-attention
- 128 projection dimensions, 4 attention heads
- MLP: 512 → 256 units with GELU activation and Dropout
- Classification head: LayerNorm → Global Average Pooling → Dense → Dropout → Output
- Patch size: 16×16 (196 patches per 224×224 image)
- Trained with RMSprop, LR=0.001, 100 epochs

### 4. Vision Transformer — Custom PyTorch (from scratch)

A fully custom ViT implementation in PyTorch, built without any pretrained weights:
- `PatchEncoder`: Conv2d-based patch projection + learnable positional embeddings + CLS token
- `ViTBlock` / `TransformerEncoderBlock`: LayerNorm → Multi-Head Self-Attention → Skip connection → MLP (GELU) → Skip connection
- `ViT` / `VisionTransformer`: stacked encoder blocks + MLP classification head
- Multiple hyperparameter configurations explored (see `notebooks/vit_hyperparameter_search/`)

### 5. Pretrained ViT (Post-submission exploration)

After the formal submission, pretrained ViT models were fine-tuned on the ABIDE-I data:
- `vit_base_patch16_224` and `vit_small_patch16_224` via [timm](https://github.com/huggingface/pytorch-image-models)
- `google/vit-base-patch16-224` via [HuggingFace Transformers](https://huggingface.co/google/vit-base-patch16-224)

These experiments were run on paid GPU sessions (Paperspace/DigitalOcean) and the final trained weights were not retained. The notebooks reflect the setup and training loop.

---

## Results

### CNN Model

| Optimizer | Train Acc | Test Acc | AUC  | Precision | Recall | Specificity | F1   |
|-----------|-----------|----------|------|-----------|--------|-------------|------|
| Adam      | 0.92      | 0.90     | 0.97 | 0.87      | 0.93   | 0.87        | 0.90 |
| NAdam     | 0.93      | 0.91     | 0.97 | 0.90      | 0.90   | 0.91        | 0.90 |
| RMSprop   | 0.94      | 0.91     | 0.97 | 0.90      | 0.90   | 0.91        | 0.90 |

### Skip-Connected CNN

| Optimizer | Train Acc | Test Acc | AUC  | Precision | Recall | Specificity | F1   |
|-----------|-----------|----------|------|-----------|--------|-------------|------|
| Adam      | 0.91      | 0.90     | 0.97 | 0.90      | 0.88   | 0.91        | 0.89 |
| NAdam     | 0.93      | 0.91     | 0.98 | 0.93      | 0.88   | 0.94        | 0.90 |
| RMSprop   | 0.93      | 0.91     | 0.97 | 0.93      | 0.87   | 0.94        | 0.90 |

Both CNN variants consistently achieved **90–94% training accuracy, 90–91% test accuracy, and AUC of 0.97–0.98** across all three optimizers, demonstrating strong and stable classification performance.

### Vision Transformer (Keras, RMSprop)

| Metric    | Score  |
|-----------|--------|
| Train Accuracy | 0.5973 |
| Test Accuracy  | 0.5975 |
| AUC            | 0.5757 |
| Precision      | 0.5810 |
| Recall         | 0.4652 |
| F1-Score       | 0.5167 |

The ViT model trained from scratch underperformed significantly — essentially near-random classification. This is consistent with the known data-hunger of Vision Transformers: training from scratch on a dataset of ~100K slices from 1,067 subjects is insufficient. The model was constrained to 100 epochs due to GPU and session time limitations.

**Key insight:** CNNs are far better suited to this scale of medical imaging data when training from scratch. ViTs require either much larger datasets or pretrained weights to be effective.

---

## Explainability — LIME (Post-submission)

A LIME (Local Interpretable Model-Agnostic Explanations) explainability notebook was developed after the formal submission to probe what the CNN was learning from the MRI slices.

The notebook (`notebooks/xai/lime_explainability.ipynb`):
1. Loads a trained CNN model from saved weights (`SXAI_weights.pth`)
2. Loads a raw MRI PNG slice from the ABIDE dataset
3. Runs `LimeImageExplainer.explain_instance()` to generate a superpixel-based explanation
4. Visualizes the explanation using `mark_boundaries` to highlight which regions of the MRI contributed most to the ASD/TC classification decision

The LIME setup ran and the explainer was invoked, but visualization results were incomplete — the notebook encountered an error before producing the final annotated image output. **GradCAM** was also explored after submission but those notebooks were not retained.

---

## Repository Structure

```
asd-detection-neuroimaging/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/                                 ← Standalone Python modules
│   ├── preprocess.py    — CNN and ViT preprocessing pipelines; edge-based crop; patchify
│   ├── dataset.py       — ASDDataset (PyTorch Dataset); build_loaders factory
│   ├── models.py        — ASD_CNN, ASD_SkipCNN, ViT_PyTorch architectures + get_model()
│   ├── train.py         — Training loop; Adam/NAdam/RMSprop; early stopping; weight saving
│   ├── evaluate.py      — Metrics, ROC curve, confusion matrix, training curve plots
│   └── lime_explain.py  — LIME explainability; superpixel visualization on MRI slices
│
├── notebooks/
│   │
│   ├── README.md                        ← Detailed description of every notebook
│   │
│   ├── cnn/                             ← Custom 5-layer CNN (PyTorch)
│   │   ├── cnn_adam.ipynb               — Plain CNN, Adam optimizer
│   │   ├── cnn_nadam.ipynb              — Plain CNN, NAdam optimizer
│   │   ├── cnn_nadam_and_rmsprop.ipynb  — Plain CNN, NAdam (half-dataset) + RMSprop (full)
│   │   ├── cnn_skip_adam.ipynb          — Skip-connected CNN, Adam
│   │   ├── cnn_skip_nadam.ipynb         — Skip-connected CNN, NAdam
│   │   └── cnn_skip_rmsprop.ipynb       — Skip-connected CNN, RMSprop
│   │
│   ├── vit_custom_keras/                ← Custom ViT, TensorFlow/Keras
│   │   └── vit_keras_rmsprop.ipynb      — Full ViT from scratch; 6 blocks, 4 heads, 128 dim
│   │
│   ├── vit_custom_pytorch/              ← Custom ViT, PyTorch from scratch
│   │   ├── vit_pytorch_full.ipynb       — Full ViT with patchify visualization
│   │   └── vit_pytorch_compact.ipynb    — Compact ViT variant (GELU MLP, ViTBlock + CLS token)
│   │
│   ├── vit_pretrained/                  ← Pretrained ViT fine-tuning (post-submission)
│   │   ├── vit_pretrained_timm.ipynb         — ViT-B/16 + ViT-S/16 via timm
│   │   └── vit_pretrained_huggingface.ipynb  — google/vit-base-patch16-224 via Transformers
│   │
│   ├── vit_hyperparameter_search/       ← ViT configuration experiments (post-submission)
│   │   ├── vit_4heads_64dim_128ff_6layers.ipynb    — 4 heads, dim=64, ff=128, 6 layers
│   │   ├── vit_4heads_64dim_128ff_6layers_v2.ipynb — Same config, extended run
│   │   └── vit_pytorch_early_experiment.ipynb      — Early ViT prototype
│   │
│   └── xai/                             ← Explainable AI (post-submission)
│       └── lime_explainability.ipynb    — LIME superpixel explanations on MRI slices
│
├── results/
│   ├── README.md                        ← All metrics tables with sources
│   ├── experiment_results.csv           ← Full metrics for all 11 experimental runs
│   └── figures/
│       ├── accuracy_comparison.png      — Train vs test accuracy, all CNN variants
│       ├── metrics_all_models.png       — AUC/Precision/Recall/F1 across 6 CNN experiments
│       ├── cnn_vs_vit.png               — Direct CNN vs ViT performance comparison
│       ├── radar_cnn_optimizers.png     — Radar chart of all metrics, all optimizers
│       └── dataset_size_comparison.png — 533-subject vs 1,067-subject accuracy/AUC
│
├── docs/
│   ├── final_report_btech.pdf           ← Full B.Tech final year report (IEEE format)
│   ├── model_architecture_diagram.pdf  ← CNN and ViT architecture diagrams
│   ├── presentation.pptx               ← Project presentation slides
│   └── base_paper_reference.pdf        ← Primary reference paper (ABIDE-I basis)
│
└── data/
    └── README.md                        ← ABIDE-I download instructions and CSV format
```

---

## Quickstart

All notebooks were developed and run on **Kaggle** (free GPU tier) and **Paperspace/DigitalOcean** (paid GPU sessions for pretrained ViT experiments).

```bash
git clone https://github.com/TarunSadarla2606/asd-detection-neuroimaging.git
cd asd-detection-neuroimaging
pip install -r requirements.txt
# Open any notebook in Kaggle or Colab
```

**Data setup:** The notebooks expect data at Kaggle paths (`/kaggle/input/autism/`). See `data/README.md` for ABIDE-I download instructions and how to adapt paths for local or Colab use.

---

## Key Findings

**CNNs significantly outperform from-scratch ViTs on this dataset scale.** The custom 5-layer CNN and its skip-connected variant both reached ~91% test accuracy and AUC 0.97 across all three optimizers, with stable training curves and no signs of overfitting. The ViT trained from scratch plateaued near random chance (~60%), limited by dataset size and GPU constraints.

**Skip connections provide marginal but consistent improvement** — particularly with NAdam (AUC improves from 0.97 to 0.98) and improved specificity across all optimizers, suggesting better gradient flow helps on this task.

**Optimizer choice matters less than architecture at this scale** — all three optimizers produced comparable results within each architecture, with RMSprop edging out slightly on train accuracy (0.94) for the plain CNN.

**ViTs require pre-training to be competitive on small medical datasets.** The pretrained ViT experiments (run after submission) were set up to test this hypothesis directly, though final results from those runs were not retained.

---

## Future Work

- [ ] Complete GradCAM implementation to visualize which brain regions activate for ASD classification
- [ ] Fix LIME visualization pipeline and produce annotated MRI explanation figures
- [ ] Fine-tune pretrained ViT (ViT-B/16 ImageNet weights) with sufficient epochs — expected to close the gap with CNN
- [ ] Multi-site analysis: evaluate per-site accuracy to assess cross-scanner generalization
- [ ] Fuse sMRI with fMRI (functional connectivity) for multimodal classification
- [ ] Explore 3D convolutions on volumetric sMRI data rather than 2D slice-level classification

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
opencv-python>=4.8.0
Pillow>=10.0.0
tqdm>=4.65.0
timm>=0.9.0
transformers>=4.35.0
lime>=0.2.0.1
```

---

## Citation

```bibtex
@misc{sadarla2023asd,
  author      = {Sadarla, Tarun},
  title       = {Detection of Autism Spectrum Disorder using sMRI and Deep Learning Techniques},
  year        = {2023},
  institution = {Manipal Institute of Technology, Department of Computer Science and Engineering},
  note        = {B.Tech Final Year Project, Reg. No. 190905165},
  supervisor  = {Andrew, J.},
  url         = {https://github.com/TarunSadarla2606/asd-detection-neuroimaging}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*B.Tech Final Year Project — Manipal Institute of Technology, November 2023*
*Department of Computer Science & Engineering*
*Guide: Dr. J. Andrew, Assistant Professor (Senior Scale)*
