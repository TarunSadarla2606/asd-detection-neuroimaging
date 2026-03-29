# Notebooks

All notebooks were run on **Kaggle** (free GPU tier) except the pretrained ViT experiments, which used paid GPU sessions on Paperspace/DigitalOcean.

---

## `cnn/` — Custom 5-Layer CNN (PyTorch)

All six notebooks share the same architecture and data pipeline, differing only in optimizer and whether a skip connection is present. Each notebook is self-contained: data loading → model definition → training → evaluation (accuracy, AUC, ROC curve, confusion matrix).

| Notebook | Architecture | Optimizer | Key hyperparameters |
|---|---|---|---|
| `cnn_adam.ipynb` | Plain CNN | Adam | lr=0.001, β₁=0.9, β₂=0.999, ε=1e-7 |
| `cnn_nadam.ipynb` | Plain CNN | NAdam | lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8 |
| `cnn_nadam_and_rmsprop.ipynb` | Plain CNN | NAdam (first half) → RMSprop (full dataset) | Two experiments in one notebook; second half marked `# COMPLETE DATA` |
| `cnn_skip_adam.ipynb` | Skip-CNN | Adam | Same as above |
| `cnn_skip_nadam.ipynb` | Skip-CNN | NAdam | Same as above |
| `cnn_skip_rmsprop.ipynb` | Skip-CNN | RMSprop | lr=0.001, ε=1e-7 |

**CNN architecture** (5 convolutional blocks):
```
Conv(3→16, k=3) + LeakyReLU → MaxPool(2×2) → Dropout(0.2)
Conv(16→32, k=3) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(0.2)
Conv(32→64, k=3) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(0.2)
Conv(64→128, k=3) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(0.2)
Conv(128→256, k=3) + BatchNorm + LeakyReLU → MaxPool(2×2) → Dropout(0.2)
Flatten → FC(100) + LeakyReLU → Output(2)
```

**Skip-CNN addition:** Output of Conv layer 1 is added to the input of Conv layer 3, addressing vanishing gradients and promoting feature reuse.

**Training:** 50 epochs, batch size 64, CrossEntropyLoss, early stopping (patience=5).

---

## `vit_custom_keras/` — Custom ViT, TensorFlow/Keras

| Notebook | Framework | Description |
|---|---|---|
| `vit_keras_rmsprop.ipynb` | TensorFlow/Keras | Full ViT from scratch; 6 transformer blocks, 4 heads, 128 dim, patch size 16 |

**Architecture details:**
- Patch embedding: `16×16` patches → 196 patches per 224×224 image → linear embedding to 128 dims
- Transformer encoder: 6 blocks × (LayerNorm → MultiHeadAttention(4 heads) → skip → LayerNorm → MLP(512→256, GELU) → Dropout → skip)
- Classification head: LayerNorm → GlobalAveragePooling → Dense(256) → Dropout → Dense(2)
- Optimizer: RMSprop, LR=0.001, ε=1e-7
- Trained for 100 epochs, batch size 64, SparseCategorialCrossEntropy

**Results:** ~60% accuracy, AUC ~0.58 — near-random performance, consistent with insufficient training data for a from-scratch ViT.

---

## `vit_custom_pytorch/` — Custom ViT, PyTorch from scratch

| Notebook | Description |
|---|---|
| `vit_pytorch_full.ipynb` | Complete ViT build with patchify function, patch visualization, full training loop |
| `vit_pytorch_compact.ipynb` | Alternative implementation using `ViTBlock` (GELU MLP) + `PatchEmbedding` with CLS token |

Both notebooks implement the core ViT components from scratch:
- `PatchEncoder` / `PatchEmbedding` — Conv2d-based patch projection + learnable positional embeddings
- `TransformerEncoderBlock` / `ViTBlock` — LayerNorm, Multi-Head Self-Attention, skip connections, MLP
- `VisionTransformer` / `ViT` — stacked blocks + classification MLP head

`vit_pytorch_full.ipynb` also includes manual `patchify()` function and patch grid visualization to validate preprocessing.

---

## `vit_pretrained/` — Pretrained ViT Fine-tuning (Post-submission)

These experiments were conducted after the formal B.Tech submission to test whether pretrained weights would close the performance gap with CNNs.

| Notebook | Pretrained Model | Library |
|---|---|---|
| `vit_pretrained_timm.ipynb` | `vit_base_patch16_224` and `vit_small_patch16_224` | [timm](https://github.com/huggingface/pytorch-image-models) |
| `vit_pretrained_huggingface.ipynb` | `google/vit-base-patch16-224` | [HuggingFace Transformers](https://huggingface.co/google/vit-base-patch16-224) |

Both replace the final classification head with a binary output layer and fine-tune on ABIDE-I slices. Trained on paid GPU sessions (Paperspace/DigitalOcean); final weights not retained. Notebooks show training setup and loop.

---

## `vit_hyperparameter_search/` — ViT Configuration Experiments (Post-submission)

Systematic exploration of ViT hyperparameters to find configurations that converged on this dataset.

| Notebook | Config |
|---|---|
| `vit_4heads_64dim_128ff_6layers.ipynb` | 4 attention heads, embed_dim=64, ff_dim=128, 6 layers |
| `vit_4heads_64dim_128ff_6layers_v2.ipynb` | Same config, extended training run |
| `vit_pytorch_early_experiment.ipynb` | Early prototype — embed_dim=128, 6 layers, Adam |

These are lightweight ViT configurations specifically designed to fit within Kaggle's free GPU memory limits while still having enough capacity to learn.

---

## `xai/` — Explainable AI (Post-submission)

| Notebook | Method | Status |
|---|---|---|
| `lime_explainability.ipynb` | LIME (Local Interpretable Model-Agnostic Explanations) | Partial — setup and explainer ran; visualization incomplete due to error |

**What it does:**
1. Loads a trained CNN model from `SXAI_weights.pth`
2. Loads a raw ABIDE MRI slice (`.png`)
3. Runs `lime_image.LimeImageExplainer().explain_instance()` to compute superpixel importances
4. Attempts to visualize with `skimage.segmentation.mark_boundaries()`

The LIME explainer ran and generated an explanation object, but the final visualization cell encountered an error before producing the annotated image output. The core LIME integration with the PyTorch model is complete and reusable.

**GradCAM** was also explored post-submission but those notebooks were not retained.
