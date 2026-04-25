# Changelog — ASD Detection from sMRI (B.Tech Baseline)

All notable changes to this repository are documented here.

---

## [2026-04] Portfolio Cleanup & CI

- Added GitHub Actions CI pipeline (`pytest`, CPU-only PyTorch)
- Added automated test suite (`tests/`) covering `ASD_CNN`, `ASD_SkipCNN`, `ViT_PyTorch`, and `get_model()` factory — no weights required
- Added `.github/ISSUE_TEMPLATE/bug_report.md`
- Updated README: CI badge, ViT failure post-mortem, Phase 2 cross-link table, Future Work corrections
- Fixed Future Work section to mark GradCAM, LIME, and multi-site analysis as completed in Phase 2

---

## [2023-05] B.Tech Capstone — Original Submission

- Trained and evaluated 11 model configurations on ABIDE-I (1,067 subjects, 17 sites)
- Architectures: 5-layer plain CNN, skip-connected CNN, from-scratch ViT
- Best result: CNN with NAdam — AUC 0.978, Sensitivity 90.2%, Specificity 91.4%
- ViT result: AUC 0.62 — failed to converge from scratch at ABIDE-I data scale
- Root causes documented: data scale mismatch, no inductive bias, optimisation instability
- Established CNN vs ViT baseline for graduate Phase 2 work
- Notebooks run on Kaggle (free GPU tier)
