# Results

All metrics below are sourced directly from notebook cell outputs. The full dataset runs use 1,067 subjects (72,367 train slices, 20,102 test slices). Early experiments used a 533-subject subset.

## CNN Model — Full Dataset (1,067 subjects)

| Optimizer | Train Acc | Test Acc | AUC  | Precision | Recall | Specificity | F1   |
|-----------|-----------|----------|------|-----------|--------|-------------|------|
| Adam      | 0.92      | 0.90     | 0.97 | 0.87      | 0.93   | 0.87        | 0.90 |
| NAdam     | 0.93      | 0.91     | 0.97 | 0.90      | 0.90   | 0.91        | 0.90 |
| RMSprop   | **0.94**  | 0.91     | 0.97 | 0.90      | 0.90   | 0.91        | 0.90 |

## Skip-Connected CNN — Full Dataset (1,067 subjects)

| Optimizer | Train Acc | Test Acc | AUC  | Precision | Recall | Specificity | F1   |
|-----------|-----------|----------|------|-----------|--------|-------------|------|
| Adam      | 0.91      | 0.90     | 0.97 | 0.90      | 0.88   | 0.91        | 0.89 |
| NAdam     | 0.93      | 0.91     | **0.98** | **0.93** | 0.88 | **0.94** | 0.90 |
| RMSprop   | 0.93      | 0.91     | 0.97 | **0.93**  | 0.87   | **0.94**    | 0.90 |

## CNN Early Runs — Half Dataset (533 subjects)

| Optimizer | Train Acc | Test Acc | AUC  | Precision | Recall | Specificity | F1   |
|-----------|-----------|----------|------|-----------|--------|-------------|------|
| Adam      | 0.94      | 0.92     | 0.98 | 0.91      | 0.94   | 0.91        | 0.92 |
| NAdam     | 0.94      | 0.92     | 0.98 | 0.93      | 0.91   | 0.94        | 0.92 |
| RMSprop   | 0.93      | 0.91     | 0.98 | 0.97      | 0.84   | 0.97        | 0.90 |

## Vision Transformer (Keras, from scratch) — Full Dataset

| Metric    | Score  |
|-----------|--------|
| Train Accuracy | 0.597 |
| Test Accuracy  | 0.616 |
| AUC            | 0.619 |
| Precision      | 0.587 |
| Recall         | 0.694 |
| F1-Score       | 0.636 |

Near-random performance; consistent with insufficient data for training ViTs from scratch.

## Figures

| File | Description |
|------|-------------|
| `accuracy_comparison.png` | Train vs test accuracy for all CNN variants and optimizers |
| `metrics_all_models.png` | AUC, Precision, Recall, F1 side-by-side for all 6 CNN experiments |
| `cnn_vs_vit.png` | Direct comparison of best CNN, best Skip-CNN, and ViT |
| `radar_cnn_optimizers.png` | Radar chart of all metrics across CNN optimizers |
| `dataset_size_comparison.png` | Test accuracy and AUC at 533 vs 1,067 subjects |
