"""Architecture smoke-tests for Phase 1 ASD detection models.
Verifies output shapes and factory behaviour without loading any weights.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import ASD_CNN, ASD_SkipCNN, ViT_PyTorch, get_model

BATCH = 2
X = torch.randn(BATCH, 3, 224, 224)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

def test_asd_cnn_output_shape():
    assert ASD_CNN().eval()(X).shape == (BATCH, 2)


def test_asd_skipcnn_output_shape():
    assert ASD_SkipCNN().eval()(X).shape == (BATCH, 2)


def test_vit_pytorch_output_shape():
    # Small embed_dim + 2 layers keeps the CPU test fast
    m = ViT_PyTorch(embed_dim=64, num_heads=2, num_layers=2).eval()
    assert m(X).shape == (BATCH, 2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_get_model_cnn():
    m = get_model("cnn").eval()
    assert m(X).shape == (BATCH, 2)


def test_get_model_skip_cnn():
    m = get_model("skip_cnn").eval()
    assert m(X).shape == (BATCH, 2)


def test_get_model_vit_small():
    m = get_model("vit", embed_dim=64, num_heads=2, num_layers=2).eval()
    assert m(X).shape == (BATCH, 2)


def test_get_model_unknown_raises():
    with pytest.raises(ValueError):
        get_model("nonexistent_model")
