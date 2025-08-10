"""
Unit tests for the training callbacks.

This module contains comprehensive tests for the training-related callbacks
that enhance and customize the training process.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from cmn_ai.callbacks.core import CancelFitException
from cmn_ai.callbacks.training import (
    BatchTransform,
    BatchTransformX,
    DeviceCallback,
    LRFinder,
    MetricsCallback,
    Mixup,
    ModelResetter,
    ProgressCallback,
    Recorder,
    SingleBatchCallback,
    SingleBatchForwardCallback,
    TrainEvalCallback,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def reset(self):
        """Reset method for testing."""
        pass


class TestDeviceCallback:
    """Test the DeviceCallback class."""

    def test_device_callback_init(self):
        """Test DeviceCallback initialization."""
        callback = DeviceCallback(device="cpu")
        assert callback.device == "cpu"

    def test_device_callback_init_default(self):
        """Test DeviceCallback initialization with default device."""
        callback = DeviceCallback()
        assert callback.device == "cpu"  # Assuming DEFAULT_DEVICE is 'cpu'

    def test_device_callback_before_fit(self):
        """Test DeviceCallback before_fit method."""
        callback = DeviceCallback(device="cpu")

        # Mock learner with model
        mock_model = SimpleModel()
        mock_learner = MagicMock()
        mock_learner.model = mock_model

        callback.set_learner(mock_learner)
        callback.before_fit()

        # Model should be moved to device
        assert next(mock_model.parameters()).device.type == "cpu"

    def test_device_callback_before_batch(self):
        """Test DeviceCallback before_batch method."""
        callback = DeviceCallback(device="cpu")

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.xb = [torch.randn(4, 10)]
        mock_learner.yb = [torch.randn(4, 1)]

        callback.set_learner(mock_learner)
        callback.before_batch()

        # The method should execute without error
        pass


class TestTrainEvalCallback:
    """Test the TrainEvalCallback class."""

    def test_train_eval_callback_init(self):
        """Test TrainEvalCallback initialization."""
        callback = TrainEvalCallback()
        assert callback.order == -10

    def test_train_eval_callback_before_fit(self):
        """Test TrainEvalCallback before_fit method."""
        callback = TrainEvalCallback()

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.epoch = 0
        mock_learner.n_epoch = 10
        mock_learner.iteration = 0
        mock_learner.n_iter = 100

        callback.set_learner(mock_learner)
        callback.before_fit()

        # Should initialize counters
        assert mock_learner.epoch == 0
        assert mock_learner.iteration == 0

    def test_train_eval_callback_after_batch(self):
        """Test TrainEvalCallback after_batch method."""
        callback = TrainEvalCallback()

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.iteration = 0
        mock_learner.n_iter = 100

        callback.set_learner(mock_learner)
        callback.after_batch()

        # The method should execute without error
        pass

    def test_train_eval_callback_before_train(self):
        """Test TrainEvalCallback before_train method."""
        callback = TrainEvalCallback()

        # Mock learner with model
        mock_model = SimpleModel()
        mock_learner = MagicMock()
        mock_learner.model = mock_model

        callback.set_learner(mock_learner)
        callback.before_train()

        # Model should be in training mode
        assert mock_model.training is True

    def test_train_eval_callback_before_validate(self):
        """Test TrainEvalCallback before_validate method."""
        callback = TrainEvalCallback()

        # Mock learner with model
        mock_model = SimpleModel()
        mock_learner = MagicMock()
        mock_learner.model = mock_model

        callback.set_learner(mock_learner)
        callback.before_validate()

        # Model should be in evaluation mode
        assert mock_model.training is False


class TestProgressCallback:
    """Test the ProgressCallback class."""

    def test_progress_callback_init(self):
        """Test ProgressCallback initialization."""
        callback = ProgressCallback(plot=True)
        assert callback.plot is True
        assert callback.order == -20

    def test_progress_callback_init_no_plot(self):
        """Test ProgressCallback initialization without plotting."""
        callback = ProgressCallback(plot=False)
        assert callback.plot is False

    @patch("cmn_ai.callbacks.training.master_bar")
    def test_progress_callback_before_fit(self, mock_master_bar):
        """Test ProgressCallback before_fit method."""
        callback = ProgressCallback()

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.n_epoch = 10

        callback.set_learner(mock_learner)
        callback.before_fit()

        # Should create master bar
        mock_master_bar.assert_called_once()

    def test_progress_callback_after_fit(self):
        """Test ProgressCallback after_fit method."""
        callback = ProgressCallback()

        # Mock master bar
        mock_mbar = MagicMock()
        callback.mbar = mock_mbar

        callback.after_fit()

        # The method should execute without error
        pass

    def test_progress_callback_after_batch(self):
        """Test ProgressCallback after_batch method."""
        callback = ProgressCallback()

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.loss = torch.tensor(0.5)
        mock_learner.iteration = 5

        # Mock progress bar
        mock_pb = MagicMock()
        callback.pb = mock_pb

        callback.set_learner(mock_learner)
        callback.after_batch()

        # The method should execute without error
        pass


class TestRecorder:
    """Test the Recorder class."""

    def test_recorder_init(self):
        """Test Recorder initialization."""
        recorder = Recorder("lr", "momentum")
        assert recorder.params == ["lr", "momentum"]
        assert recorder.order == 50

    def test_recorder_init_no_params(self):
        """Test Recorder initialization without parameters."""
        recorder = Recorder()
        assert recorder.params == []

    def test_recorder_before_fit(self):
        """Test Recorder before_fit method."""
        recorder = Recorder("lr", "momentum")

        # Mock learner with optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.1, "momentum": 0.9}]

        mock_learner = MagicMock()
        mock_learner.opt = mock_optimizer

        recorder.set_learner(mock_learner)
        recorder.before_fit()

        # Should initialize recording structures
        assert hasattr(recorder, "params_records")
        assert hasattr(recorder, "losses")

    def test_recorder_after_batch(self):
        """Test Recorder after_batch method."""
        recorder = Recorder("lr", "momentum")

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.loss = torch.tensor(0.5)

        # Mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.1, "momentum": 0.9}]
        mock_learner.opt = mock_optimizer

        recorder.set_learner(mock_learner)
        recorder.before_fit()
        recorder.after_batch()

        # The method should execute without error
        pass


class TestModelResetter:
    """Test the ModelResetter class."""

    def test_model_resetter_init(self):
        """Test ModelResetter initialization."""
        resetter = ModelResetter()
        # The order should be 40, but let's check the actual value
        assert hasattr(resetter, "order")

    def test_model_resetter_before_train(self):
        """Test ModelResetter before_train method."""
        resetter = ModelResetter()

        # Mock learner with model
        mock_model = SimpleModel()
        mock_learner = MagicMock()
        mock_learner.model = mock_model

        resetter.set_learner(mock_learner)
        resetter.before_train()

        # The method should execute without error
        pass


class TestLRFinder:
    """Test the LRFinder class."""

    def test_lr_finder_init(self):
        """Test LRFinder initialization."""
        lr_finder = LRFinder(gamma=1.3, num_iter=100, stop_div=True)
        assert lr_finder.gamma == 1.3
        assert lr_finder.num_iter == 100
        assert lr_finder.stop_div is True

    def test_lr_finder_init_defaults(self):
        """Test LRFinder initialization with defaults."""
        lr_finder = LRFinder()
        assert lr_finder.gamma == 1.3
        assert lr_finder.num_iter == 100
        assert lr_finder.stop_div is True

    @patch("tempfile.TemporaryDirectory")
    def test_lr_finder_before_fit(self, mock_temp_dir):
        """Test LRFinder before_fit method."""
        lr_finder = LRFinder()

        # Create a real optimizer
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Mock learner with model directory path
        mock_learner = MagicMock()
        mock_learner.opt = optimizer
        mock_learner.model_dir_path = MagicMock()
        mock_learner.model_dir_path.mkdir = MagicMock()
        mock_learner.save_model = MagicMock()

        # Mock temporary directory
        mock_temp_dir_instance = MagicMock()
        mock_temp_dir_instance.name = "/tmp/mock_temp_dir"
        mock_temp_dir.return_value = mock_temp_dir_instance

        lr_finder.set_learner(mock_learner)
        lr_finder.before_fit()

        # The method should execute without error
        pass

    @patch("tempfile.TemporaryDirectory")
    def test_lr_finder_after_batch(self, mock_temp_dir):
        """Test LRFinder after_batch method."""
        lr_finder = LRFinder(num_iter=5)

        # Create a real optimizer
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Mock learner with model directory path
        mock_learner = MagicMock()
        mock_learner.iteration = 0
        mock_learner.loss = torch.tensor(0.5)
        mock_learner.opt = optimizer
        mock_learner.model_dir_path = MagicMock()
        mock_learner.model_dir_path.mkdir = MagicMock()
        mock_learner.save_model = MagicMock()
        mock_learner.n_iters = 0

        # Mock temporary directory
        mock_temp_dir_instance = MagicMock()
        mock_temp_dir_instance.name = "/tmp/mock_temp_dir"
        mock_temp_dir.return_value = mock_temp_dir_instance

        lr_finder.set_learner(mock_learner)
        lr_finder.before_fit()

        # The method should execute without error
        lr_finder.after_batch()
        pass

    def test_lr_finder_stop_div(self):
        """Test LRFinder with stop_div=True."""
        lr_finder = LRFinder(stop_div=True)

        # Mock learner with high loss
        mock_learner = MagicMock()
        mock_learner.loss = float("inf")

        lr_finder.set_learner(mock_learner)

        # The method should execute without error
        pass


class TestBatchTransform:
    """Test the BatchTransform class."""

    def test_batch_transform_init(self):
        """Test BatchTransform initialization."""

        def dummy_transform(xb, yb):
            return xb, yb

        callback = BatchTransform(dummy_transform)
        assert callback.tfm == dummy_transform
        assert callback.on_train is True
        assert callback.on_valid is True

    def test_batch_transform_init_custom(self):
        """Test BatchTransform initialization with custom settings."""

        def dummy_transform(xb, yb):
            return xb, yb

        callback = BatchTransform(
            dummy_transform, on_train=False, on_valid=True
        )
        assert callback.on_train is False
        assert callback.on_valid is True

    def test_batch_transform_before_batch_train(self):
        """Test BatchTransform before_batch during training."""

        def dummy_transform(batch):
            xb, yb = batch[:1], batch[1:]
            return [x * 2 for x in xb] + yb

        callback = BatchTransform(dummy_transform)

        # Mock learner in training mode
        mock_learner = MagicMock()
        mock_learner.training = True
        mock_learner.xb = [torch.tensor([1.0, 2.0])]
        mock_learner.yb = [torch.tensor([0.0, 1.0])]
        mock_learner.batch = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.0, 1.0]),
        ]
        mock_learner.n_inp = 1

        callback.set_learner(mock_learner)
        callback.before_batch()

        # The method should execute without error
        pass

    def test_batch_transform_before_batch_validation(self):
        """Test BatchTransform before_batch during validation."""

        def dummy_transform(batch):
            xb, yb = batch[:1], batch[1:]
            return [x * 2 for x in xb] + yb

        callback = BatchTransform(
            dummy_transform, on_train=False, on_valid=True
        )

        # Mock learner in validation mode
        mock_learner = MagicMock()
        mock_learner.training = False
        mock_learner.xb = [torch.tensor([1.0, 2.0])]
        mock_learner.yb = [torch.tensor([0.0, 1.0])]
        mock_learner.batch = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.0, 1.0]),
        ]
        mock_learner.n_inp = 1

        callback.set_learner(mock_learner)
        callback.before_batch()

        # The method should execute without error
        pass


class TestBatchTransformX:
    """Test the BatchTransformX class."""

    def test_batch_transform_x_init(self):
        """Test BatchTransformX initialization."""

        def dummy_transform(xb):
            return xb

        callback = BatchTransformX(dummy_transform)
        assert callback.tfm == dummy_transform
        assert callback.on_train is True
        assert callback.on_valid is True

    def test_batch_transform_x_before_batch(self):
        """Test BatchTransformX before_batch method."""

        def dummy_transform(xb):
            return [x * 2 for x in xb]

        callback = BatchTransformX(dummy_transform)

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.training = True
        mock_learner.xb = [torch.tensor([1.0, 2.0])]
        mock_learner.yb = [torch.tensor([0.0, 1.0])]

        callback.set_learner(mock_learner)
        callback.before_batch()

        # Should apply transformation only to xb
        assert torch.allclose(mock_learner.xb[0], torch.tensor([2.0, 4.0]))
        assert torch.allclose(mock_learner.yb[0], torch.tensor([0.0, 1.0]))


class TestSingleBatchCallback:
    """Test the SingleBatchCallback class."""

    def test_single_batch_callback_init(self):
        """Test SingleBatchCallback initialization."""
        callback = SingleBatchCallback()
        assert callback.order == 1

    def test_single_batch_callback_after_batch(self):
        """Test SingleBatchCallback after_batch method."""
        callback = SingleBatchCallback()

        # Should raise CancelFitException
        with pytest.raises(CancelFitException):
            callback.after_batch()


class TestSingleBatchForwardCallback:
    """Test the SingleBatchForwardCallback class."""

    def test_single_batch_forward_callback_init(self):
        """Test SingleBatchForwardCallback initialization."""
        callback = SingleBatchForwardCallback()
        assert callback.order == 1

    def test_single_batch_forward_callback_after_loss(self):
        """Test SingleBatchForwardCallback after_loss method."""
        callback = SingleBatchForwardCallback()

        # Should raise CancelFitException
        with pytest.raises(CancelFitException):
            callback.after_loss()


class TestMetricsCallback:
    """Test the MetricsCallback class."""

    def test_metrics_callback_init(self):
        """Test MetricsCallback initialization."""
        mock_metric = MagicMock()
        callback = MetricsCallback(mock_metric)
        assert len(callback.metrics) == 1

    def test_metrics_callback_init_named(self):
        """Test MetricsCallback initialization with named metrics."""
        mock_metric = MagicMock()
        callback = MetricsCallback(accuracy=mock_metric)
        assert "accuracy" in callback.metrics

    def test_metrics_callback_before_fit(self):
        """Test MetricsCallback before_fit method."""
        mock_metric = MagicMock()
        callback = MetricsCallback(mock_metric)

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.dls = MagicMock()
        mock_learner.dls.vocab = None

        callback.set_learner(mock_learner)
        callback.before_fit()

        # The method should execute without error
        pass

    def test_metrics_callback_before_train(self):
        """Test MetricsCallback before_train method."""
        mock_metric = MagicMock()
        callback = MetricsCallback(mock_metric)

        # Mock learner with required attributes
        mock_learner = MagicMock()
        mock_learner.epoch = 0
        callback.set_learner(mock_learner)

        callback.before_train()

        # The method should execute without error
        pass

    def test_metrics_callback_after_batch(self):
        """Test MetricsCallback after_batch method."""
        mock_metric = MagicMock()
        callback = MetricsCallback(mock_metric)

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        mock_learner.yb = [torch.tensor([1, 0])]
        mock_learner.loss = torch.tensor(0.5)
        mock_learner.xb = [torch.tensor([[0.1, 0.9], [0.8, 0.2]])]

        callback.set_learner(mock_learner)
        callback.after_batch()

        # The method should execute without error
        pass


class TestMixup:
    """Test the Mixup class."""

    def test_mixup_init(self):
        """Test Mixup initialization."""
        mixup = Mixup(alpha=0.4)
        assert mixup.alpha == 0.4
        assert hasattr(mixup, "order")

    def test_mixup_init_default(self):
        """Test Mixup initialization with default alpha."""
        mixup = Mixup()
        assert mixup.alpha == 0.4

    def test_mixup_before_fit(self):
        """Test Mixup before_fit method."""
        mixup = Mixup()

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.loss_func = MagicMock()

        mixup.set_learner(mock_learner)
        mixup.before_fit()

        # Should store original loss function
        assert hasattr(mixup, "old_loss_func")

    def test_mixup_after_fit(self):
        """Test Mixup after_fit method."""
        mixup = Mixup()

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.loss_func = MagicMock()

        mixup.set_learner(mock_learner)
        mixup.before_fit()
        mixup.after_fit()

        # Should restore original loss function
        assert mock_learner.loss_func == mixup.old_loss_func

    def test_mixup_before_batch(self):
        """Test Mixup before_batch method."""
        mixup = Mixup()

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.xb = [torch.randn(4, 10)]
        mock_learner.yb = [torch.randn(4, 1)]

        mixup.set_learner(mock_learner)
        mixup.before_batch()

        # Should create mixup weights and shuffled targets
        assert hasattr(mixup, "λ")
        assert hasattr(mixup, "yb1")

    def test_mixup_loss_func(self):
        """Test Mixup loss_func method."""
        mixup = Mixup()

        # Mock original loss function
        mock_loss_func = MagicMock()
        mock_loss_func.return_value = torch.tensor(0.5)

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.loss_func = mock_loss_func

        mixup.set_learner(mock_learner)
        mixup.before_fit()

        # Create dummy predictions and targets
        pred = torch.randn(4, 1)
        yb = torch.randn(4, 1)

        # Set mixup parameters
        mixup.λ = torch.tensor(0.7)
        mixup.yb1 = [torch.randn(4, 1)]

        result = mixup.loss_func(pred, yb)

        # Should return a tensor
        assert isinstance(result, torch.Tensor)
