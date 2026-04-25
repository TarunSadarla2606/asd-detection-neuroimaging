# ASD Detection using sMRI and Deep Learning

> *Detecting Autism Spectrum Disorder from structural MRI brain scans using CNNs and Vision Transformers.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![CI](https://github.com/TarunSadarla2606/asd-detection-neuroimaging/actions/workflows/ci.yml/badge.svg)](https://github.com/TarunSadarla2606/asd-detection-neuroimaging/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Institution](https://img.shields.io/badge/MIT%20Manipal-B.Tech%20Final%20Year-8B0000)]()

---

## Overview

This repository contains the complete code and documentation for my **B.Tech final year project** (2022–2023) at Manipal Institute of Technology, submitted in partial fulfilment of the degree of Bachelor of Technology in Computer Science & Engineering, under the guidance of **Dr. J. Andrew**.

It was my **first major deep learning project** and my **first project in medical imaging**. The goal was to develop and compare deep learning approaches for detecting Autism Spectrum Disorder (ASD) from structural MRI (sMRI) brain scans, addressing the limitations of current subjective, behaviour-based diagnostic methods.

The project explored three model families across two frameworks, and continued developing beyond the formal submission — including pretrained ViT experiments and an XAI (LIME) explainability attempt.

---

## Phase 2 Continuation — Graduate Clinical AI System (2026)

This B.Tech project asked: *“Can a CNN detect ASD from sMRI slices?”*

The answer was yes — with AUC 0.97–0.98 and clear CNN > ViT superiority at this data scale.

That opened a harder question: *“What does it take to go from a model that works in a notebook to a system a clinician could actually interact with?”*

The graduate-level continuation is here: **[🔗 asd-detection-clinical-ai](https://github.com/TarunSadarla2606/asd-detection-clinical-ai)**

What Phase 2 adds on top of this work:

| Capability | Phase 1 (this repo) | Phase 2 (clinical-ai) |
|---|---|---|
| **Input format** | Pre-extracted PNG slices | Raw NIfTI volumes (.nii / .nii.gz) |
| **Slice quality** | Raw slices including blanks | 4-metric quality filter (~6.5% removed) |
| **Architecture** | CNN, Skip-CNN, ViT from scratch | ASDClassifierCNN (AUC 0.994) + **Hybrid CNN-ViT** (AUC 0.997) |
| **Explainability** | LIME attempt (incomplete) | Grad-CAM + LIME + CLS-token attention rollout |
| **Uncertainty** | None | MC-Dropout (30 stochastic passes) |
| **Deployment** | Kaggle notebook | Streamlit on Hugging Face Spaces (always-on) |
| **Governance** | None | Model Card, FDA SaMD Class II framing |

---

## Problem Statement

Autism Spectrum Disorder (ASD) is a complex neurological condition currently diagnosed through subjective behavioural assessments. This process is time-consuming, inconsistent across clinicians, and often delayed — particularly in early childhood when timely intervention matters most. Neuroimaging-based approaches, coupled with deep learning, offer a path toward objective, reproducible, and biologically grounded diagnosis.

---

## Dataset — ABIDE-I

- **Source:** Autism Brain Imaging Data Exchange ([ABIDE-I](https://fcon_1000.projects.nitrc.org/indi/abide/)), accessed via COINS
- **Scale:** 1,112 subjects from 17 international research centres; median age 14.7 years (range 7–64)
- **After cleaning** (removed missing/unclear sMRI and “Unknown” label subjects): **1,067 subjects**
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

The same 5-layer architecture with a **residual skip connection** from block1 to block3. Trained with the same three optimizers.

### 3. Vision Transformer — Keras (TensorFlow)

A custom ViT implemented in TensorFlow/Keras with 6 Transformer blocks. Trained with RMSprop, LR=0.001, 100 epochs.

### 4. Vision Transformer — Custom PyTorch (from scratch)

A fully custom ViT in PyTorch with Conv2d-based patch projection, learnable positional embeddings, CLS token, and stacked TransformerEncoderBlocks.

### 5. Pretrained ViT (Post-submission exploration)

`vit_base_patch16_224` and `vit_small_patch16_224` via [timm](https://github.com/huggingface/pytorch-image-models), and `google/vit-base-patch16-224` via HuggingFace. These experiments validated the hypothesis that pretrained ViTs can match CNNs — but the trained weights were not retained (Paperspace session ended).

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

### Vision Transformer (Keras, RMSprop)

| Metric         | Score  |
|----------------|--------|
| Train Accuracy | 0.5973 |
| Test Accuracy  | 0.5975 |
| AUC            | 0.5757 |
| Precision      | 0.5810 |
| Recall         | 0.4652 |
| F1-Score       | 0.5167 |

---

## Key Findings

**CNNs significantly outperform from-scratch ViTs on this dataset scale.** The custom 5-layer CNN and its skip-connected variant both reached ~91% test accuracy and AUC 0.97 across all three optimizers. The ViT trained from scratch plateaued near random chance (~60%).

**Skip connections provide marginal but consistent improvement** — particularly with NAdam (AUC improves from 0.97 to 0.98).

**Optimizer choice matters less than architecture at this scale** — all three optimizers produced comparable results within each architecture.

### ViT Failure Post-Mortem

The from-scratch ViT achieving only ~60% accuracy was the most important finding of this project. Three root causes:

1. **Data scale mismatch.** ViTs are sequence models that learn spatial relationships from global self-attention over all patch pairs. This requires seeing a very large number of diverse examples to develop useful attention patterns. ABIDE-I at 100K slices is large for medical imaging — but ~3 orders of magnitude below the datasets (JFT-300M, ImageNet-21K) on which ViTs show their strength.

2. **No inductive bias.** CNNs have translation equivariance and local receptive fields built in — effectively pre-loaded priors for image data. ViTs must learn these from data. On small datasets, the CNN’s structural prior wins decisively.

3. **Optimisation instability.** The ViT loss curves showed oscillation and failed to converge below ~40% training loss within 100 epochs. Standard LR schedules (warmup + cosine decay) were not applied due to session time constraints — this likely compounded the problem.

**The pretrained ViT experiments** (post-submission, fine-tuning ViT-B/16 ImageNet weights) were set up to directly test whether pre-training resolves these issues. The Phase 2 graduate work resolved this differently: a **Hybrid CNN-ViT** uses the CNN backbone to extract local spatial features first, then routes those 196 feature tokens through a 4-block Transformer for global context. This sidesteps the cold-start problem entirely and achieves AUC 0.997. See [asd-detection-clinical-ai](https://github.com/TarunSadarla2606/asd-detection-clinical-ai).

---

## Explainability — LIME (Post-submission)

A LIME explainability notebook was developed after the formal submission. The setup ran and the explainer was invoked, but visualisation results were incomplete — the notebook encountered an error before producing the final annotated image output. **GradCAM** was also explored but those notebooks were not retained.

Both GradCAM and LIME are fully implemented in Phase 2 — see [asd-detection-clinical-ai](https://github.com/TarunSadarla2606/asd-detection-clinical-ai).

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
├── tests/                               ← pytest test suite (CI)
│   ├── __init__.py
│   └── test_models.py                   ← architecture smoke-tests (no weights)
│
├── src/                                 ← Standalone Python modules
│   ├── preprocess.py
│   ├── dataset.py
│   ├── models.py                        ← ASD_CNN, ASD_SkipCNN, ViT_PyTorch + get_model()
│   ├── train.py
│   ├── evaluate.py
│   └── lime_explain.py
│
├── notebooks/
│   ├── README.md
│   ├── cnn/
│   │   ├── cnn_adam.ipynb
│   │   ├── cnn_nadam.ipynb
│   │   ├── cnn_nadam_and_rmsprop.ipynb
│   │   ├── cnn_skip_adam.ipynb
│   │   ├── cnn_skip_nadam.ipynb
│   │   └── cnn_skip_rmsprop.ipynb
│   ├── vit_custom_keras/
│   │   └── vit_keras_rmsprop.ipynb
│   ├── vit_custom_pytorch/
│   │   ├── vit_pytorch_full.ipynb
│   │   └── vit_pytorch_compact.ipynb
│   ├── vit_pretrained/
│   │   ├── vit_pretrained_timm.ipynb
│   │   └── vit_pretrained_huggingface.ipynb
│   ├── vit_hyperparameter_search/
│   │   ├── vit_4heads_64dim_128ff_6layers.ipynb
│   │   ├── vit_4heads_64dim_128ff_6layers_v2.ipynb
│   │   └── vit_pytorch_early_experiment.ipynb
│   └── xai/
│       └── lime_explainability.ipynb
│
├── results/
│   ├── README.md
│   ├── experiment_results.csv
│   └── figures/
│       ├── accuracy_comparison.png
│       ├── metrics_all_models.png
│       ├── cnn_vs_vit.png
│       ├── radar_cnn_optimizers.png
│       └── dataset_size_comparison.png
│
├── docs/
│   ├── final_report_btech.pdf
│   ├── model_architecture_diagram.pdf
│   ├── presentation.pptx
│   └── base_paper_reference.pdf
│
└── data/
    └── README.md
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

---

## Future Work

- [x] Complete GradCAM implementation — **done in Phase 2** ([asd-detection-clinical-ai](https://github.com/TarunSadarla2606/asd-detection-clinical-ai))
- [x] Fix LIME visualization pipeline — **done in Phase 2**
- [x] Multi-site analysis: per-site accuracy — **done in Phase 2** (7-site stratified analysis)
- [ ] Fine-tune pretrained ViT (ViT-B/16 ImageNet weights) with sufficient epochs
- [ ] Fuse sMRI with fMRI (functional connectivity) for multimodal classification
- [ ] Explore 3D convolutions on volumetric sMRI data

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
Pillow>=9.0.0
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
