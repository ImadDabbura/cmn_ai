"""Tests for vision learner utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import cmn_ai.vision.core as vision_core
from cmn_ai.utils.data import DataLoaders
from cmn_ai.vision.core import VisionLearner


def test_show_batch_displays_requested_inputs(monkeypatch):
    """Test VisionLearner.show_batch displays samples from one batch."""
    x = torch.arange(12, dtype=torch.float32).view(3, 1, 2, 2)
    y = torch.zeros(3, 1)
    dls = DataLoaders(
        DataLoader(TensorDataset(x, y), batch_size=3),
        DataLoader(TensorDataset(x, y), batch_size=3),
    )
    learner = VisionLearner(
        nn.Sequential(nn.Flatten(), nn.Linear(4, 1)),
        dls,
        loss_func=nn.MSELoss(),
    )
    shown = {}

    def fake_show_images(images, **kwargs):
        shown["images"] = images
        shown["kwargs"] = kwargs

    monkeypatch.setattr(vision_core, "show_images", fake_show_images)

    learner.show_batch(sample_sz=2, ncols=2)

    torch.testing.assert_close(shown["images"], x[:2])
    assert shown["kwargs"] == {"ncols": 2}
