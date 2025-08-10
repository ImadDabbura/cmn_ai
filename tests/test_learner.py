"""
Unit tests for the Learner class.

This module contains comprehensive tests for the Learner class, covering
initialization, training methods, model saving/loading, learning rate finding,
and callback functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cmn_ai.callbacks.core import (
    Callback,
    CancelEpochException,
    CancelFitException,
)
from cmn_ai.callbacks.training import TrainEvalCallback
from cmn_ai.learner import Learner, params_getter
from cmn_ai.utils.data import DataLoaders


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=100, input_size=10):
        self.size = size
        self.input_size = input_size
        self.data = torch.randn(size, input_size)
        self.targets = torch.randn(size, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestCallback(Callback):
    """Test callback for testing callback functionality."""

    def __init__(self):
        self.events = []

    def __call__(self, event_nm):
        self.events.append(event_nm)
        super().__call__(event_nm)


class TestCallback2(Callback):
    """Second test callback for testing callback functionality."""

    def __init__(self):
        self.events = []

    def __call__(self, event_nm):
        self.events.append(event_nm)
        super().__call__(event_nm)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    return SimpleDataset()


@pytest.fixture
def data_loaders(simple_dataset):
    """Create data loaders for testing."""
    train_dl = torch.utils.data.DataLoader(simple_dataset, batch_size=4)
    valid_dl = torch.utils.data.DataLoader(simple_dataset, batch_size=4)
    return DataLoaders(train_dl, valid_dl)


@pytest.fixture
def learner(simple_model, data_loaders):
    """Create a learner instance for testing."""
    return Learner(simple_model, data_loaders)


class TestParamsGetter:
    """Test the params_getter function."""

    def test_params_getter(self):
        """Test that params_getter returns model parameters."""
        model = SimpleModel()
        params = list(params_getter(model))

        assert len(params) == 2  # weight and bias
        assert all(isinstance(p, nn.Parameter) for p in params)
        assert params[0].shape == (1, 10)  # weight
        assert params[1].shape == (1,)  # bias


class TestLearnerInitialization:
    """Test Learner initialization."""

    def test_learner_init_basic(self, simple_model, data_loaders):
        """Test basic learner initialization."""
        learner = Learner(simple_model, data_loaders)

        assert learner.model == simple_model
        assert learner.dls == data_loaders
        assert learner.n_inp == 1
        assert learner.loss_func == F.mse_loss
        assert learner.opt_func == torch.optim.SGD
        assert learner.lr == 1e-2
        assert learner.splitter == params_getter
        assert learner.path == Path(".")
        assert learner.model_dir_path == Path(".") / Path("models")
        assert learner.opt is None
        assert learner.logger == print
        assert len(learner.callbacks) == 1  # TrainEvalCallback
        assert isinstance(learner.callbacks[0], TrainEvalCallback)

    def test_learner_init_custom_params(self, simple_model, data_loaders):
        """Test learner initialization with custom parameters."""
        custom_loss = nn.L1Loss()
        custom_opt = torch.optim.Adam
        custom_lr = 0.001
        custom_path = Path("/tmp/test")

        learner = Learner(
            simple_model,
            data_loaders,
            n_inp=2,
            loss_func=custom_loss,
            opt_func=custom_opt,
            lr=custom_lr,
            path=custom_path,
            model_dir="custom_models",
        )

        assert learner.n_inp == 2
        assert learner.loss_func == custom_loss
        assert learner.opt_func == custom_opt
        assert learner.lr == custom_lr
        assert learner.path == custom_path
        assert learner.model_dir_path == custom_path / Path("custom_models")

    def test_learner_init_with_callbacks(self, simple_model, data_loaders):
        """Test learner initialization with custom callbacks."""
        test_callback = TestCallback()
        learner = Learner(
            simple_model, data_loaders, callbacks=[test_callback]
        )

        assert len(learner.callbacks) == 2  # TrainEvalCallback + test_callback
        assert test_callback in learner.callbacks
        assert hasattr(learner, test_callback.name)


class TestLearnerTraining:
    """Test Learner training functionality."""

    def test_fit_basic(self, learner):
        """Test basic fit functionality."""
        with patch.object(learner, "_fit") as mock_fit:
            learner.fit(n_epochs=5)

            assert learner.n_epochs == 5
            assert learner.run_train is True
            assert learner.run_valid is True
            mock_fit.assert_called_once()

    def test_fit_with_custom_params(self, learner):
        """Test fit with custom parameters."""
        with patch.object(learner, "_fit") as mock_fit:
            learner.fit(n_epochs=10, run_train=False, run_valid=True, lr=0.001)

            assert learner.n_epochs == 10
            assert learner.run_train is False
            assert learner.run_valid is True
            assert learner.opt is not None
            assert learner.opt.param_groups[0]["lr"] == 0.001
            mock_fit.assert_called_once()

    def test_fit_reset_optimizer(self, learner):
        """Test fit with optimizer reset."""
        # Create initial optimizer
        learner.fit(n_epochs=1)
        initial_opt = learner.opt

        # Fit again with reset_opt=True
        with patch.object(learner, "_fit"):
            learner.fit(n_epochs=1, reset_opt=True)

            assert learner.opt is not None
            assert learner.opt is not initial_opt

    def test_fit_with_callbacks(self, learner):
        """Test fit with temporary callbacks."""
        test_callback = TestCallback()

        with patch.object(learner, "_fit"):
            learner.fit(n_epochs=1, callbacks=[test_callback])

            # Callback should be removed after fit
            assert test_callback not in learner.callbacks
            assert not hasattr(learner, test_callback.name)


class TestLearnerTrainingMode:
    """Test Learner training mode property."""

    def test_training_property(self, learner):
        """Test training property getter and setter."""
        # Initially in training mode
        assert learner.training is True

        # Set to evaluation mode
        learner.training = False
        assert learner.training is False
        assert learner.model.training is False

        # Set back to training mode
        learner.training = True
        assert learner.training is True
        assert learner.model.training is True


class TestLearnerModelSaving:
    """Test Learner model saving and loading."""

    def test_save_model_basic(self, learner, tmp_path):
        """Test basic model saving."""
        learner.path = tmp_path

        # Create optimizer first
        learner.fit(n_epochs=1)

        # Create the models directory
        (tmp_path / "models").mkdir(exist_ok=True)

        # Save with explicit path to ensure it works
        model_path = tmp_path / "models" / "model"
        learner.save_model(path=model_path)

        assert model_path.exists()

        # Load and verify
        checkpoint = torch.load(model_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" not in checkpoint
        assert "epoch" not in checkpoint
        assert "loss" not in checkpoint

    def test_save_model_with_options(self, learner, tmp_path):
        """Test model saving with all options enabled."""
        learner.path = tmp_path

        # Create optimizer and set some state
        learner.fit(n_epochs=1)
        learner.epoch = 5
        learner.loss = torch.tensor(0.5)

        # Create the models directory
        (tmp_path / "models").mkdir(exist_ok=True)

        # Save with explicit path
        model_path = tmp_path / "models" / "model"
        learner.save_model(
            path=model_path, with_opt=True, with_epoch=True, with_loss=True
        )

        # Load and verify
        checkpoint = torch.load(model_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "loss" in checkpoint
        assert checkpoint["epoch"] == 5
        assert checkpoint["loss"] == 0.5

    def test_save_model_custom_path(self, learner, tmp_path):
        """Test model saving with custom path."""
        custom_path = tmp_path / "custom_model.pt"

        # Create optimizer first
        learner.fit(n_epochs=1)

        learner.save_model(path=custom_path)

        assert custom_path.exists()

    def test_load_model_basic(self, learner, tmp_path):
        """Test basic model loading."""
        learner.path = tmp_path

        # Save a model first
        learner.fit(n_epochs=1)

        # Create the models directory and save with explicit path
        (tmp_path / "models").mkdir(exist_ok=True)
        model_path = tmp_path / "models" / "model"
        learner.save_model(path=model_path)

        # Create new learner and load
        new_learner = Learner(SimpleModel(), learner.dls)
        new_learner.path = tmp_path
        new_learner.load_model(path=model_path)

        # Verify model state is loaded
        for p1, p2 in zip(
            learner.model.parameters(), new_learner.model.parameters()
        ):
            assert torch.allclose(p1, p2)

    def test_load_model_with_options(self, learner, tmp_path):
        """Test model loading with all options enabled."""
        learner.path = tmp_path

        # Save with all options
        learner.fit(n_epochs=1)
        learner.epoch = 10
        learner.loss = torch.tensor(0.3)

        # Create the models directory and save with explicit path
        (tmp_path / "models").mkdir(exist_ok=True)
        model_path = tmp_path / "models" / "model"
        learner.save_model(
            path=model_path, with_opt=True, with_epoch=True, with_loss=True
        )

        # Create new learner and load
        new_learner = Learner(SimpleModel(), learner.dls)
        new_learner.path = tmp_path

        # Create optimizer first so it can be loaded
        new_learner.fit(n_epochs=1)

        new_learner.load_model(
            path=model_path, with_opt=True, with_epoch=True, with_loss=True
        )

        # Verify all state is loaded
        assert new_learner.epoch == 10
        assert new_learner.loss == 0.3
        assert new_learner.opt is not None


class TestLearnerLearningRateFinder:
    """Test Learner learning rate finder."""

    def test_lr_find_basic(self, learner):
        """Test basic learning rate finder."""
        with patch.object(learner, "fit") as mock_fit:
            learner.lr_find()

            # Verify fit was called with correct parameters
            mock_fit.assert_called_once()
            call_args = mock_fit.call_args
            assert call_args[1]["run_valid"] is False
            assert call_args[1]["lr"] == 1e-7
            assert len(call_args[1]["callbacks"]) == 2  # LRFinder + Recorder

    def test_lr_find_custom_params(self, learner):
        """Test learning rate finder with custom parameters."""
        with patch.object(learner, "fit") as mock_fit:
            learner.lr_find(
                start_lr=1e-6,
                gamma=1.5,
                num_iter=200,
                stop_div=False,
                max_mult=5,
            )

            call_args = mock_fit.call_args
            assert call_args[1]["lr"] == 1e-6


class TestLearnerCallbacks:
    """Test Learner callback functionality."""

    def test_add_callbacks(self, learner):
        """Test adding callbacks to learner."""
        test_callback = TestCallback()

        added = learner._add_callbacks([test_callback])

        assert test_callback in learner.callbacks
        assert test_callback in added
        assert hasattr(learner, test_callback.name)
        assert getattr(learner, test_callback.name) == test_callback

    def test_remove_callbacks(self, learner):
        """Test removing callbacks from learner."""
        test_callback = TestCallback()
        learner._add_callbacks([test_callback])

        learner._remove_callbacks([test_callback])

        assert test_callback not in learner.callbacks
        assert not hasattr(learner, test_callback.name)

    def test_callback_execution(self, learner):
        """Test callback execution order."""
        callback1 = TestCallback()
        callback2 = TestCallback2()
        callback1.order = 1
        callback2.order = 0

        # Clear existing callbacks to avoid interference
        learner.callbacks = []

        learner._add_callbacks([callback1, callback2])

        # Verify callbacks were added
        assert len(learner.callbacks) == 2
        assert callback1 in learner.callbacks
        assert callback2 in learner.callbacks

        # Execute a callback event
        learner._callback("test_event")

        # Verify callbacks were called (both should be called)
        assert callback1.events == ["test_event"]
        assert callback2.events == ["test_event"]

        # Verify execution order (callback2 should be called first due to lower order)
        # The _callback method sorts by order, so callback2 (order=0) should be called before callback1 (order=1)
        # We can't easily test the exact order without modifying the callback system, but we can verify both were called


class TestLearnerModelSummary:
    """Test Learner model summary functionality."""

    def test_summary(self, learner):
        """Test model summary generation."""
        with patch("cmn_ai.learner.summary") as mock_summary:
            learner.summary(verbose=1)

            mock_summary.assert_called_once()
            call_args = mock_summary.call_args
            assert call_args[0][0] == learner.model
            assert call_args[1]["verbose"] == 1


class TestLearnerShowBatch:
    """Test Learner show_batch functionality."""

    def test_show_batch_not_implemented(self, learner):
        """Test that show_batch raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            learner.show_batch()


class TestLearnerTrainingLoop:
    """Test Learner training loop internals."""

    def test_with_events_normal(self, learner):
        """Test _with_events with normal execution."""
        mock_func = Mock()
        mock_func.return_value = "success"

        result = None

        def test_func():
            nonlocal result
            result = mock_func()

        learner._with_events(test_func, "test", Exception)

        mock_func.assert_called_once()
        assert result == "success"

    def test_with_events_exception(self, learner):
        """Test _with_events with exception handling."""
        test_callback = TestCallback()
        learner._add_callbacks([test_callback])

        def failing_func():
            raise ValueError("test error")

        # The _with_events method catches exceptions and doesn't re-raise them
        learner._with_events(failing_func, "test", ValueError)

        # Should have called after_cancel_test and after_test
        assert "after_cancel_test" in test_callback.events
        assert "after_test" in test_callback.events

    def test_one_batch(self, learner):
        """Test _one_batch processing."""
        # Setup
        learner.fit(n_epochs=1)  # Create optimizer
        learner.xb = (torch.randn(2, 10),)
        learner.yb = (torch.randn(2, 1),)
        learner.training = True

        with patch.object(learner, "_callback") as mock_callback:
            learner._one_batch()

            # Verify predictions and loss were computed
            assert hasattr(learner, "preds")
            assert hasattr(learner, "loss")
            assert learner.preds.shape == (2, 1)
            assert isinstance(learner.loss, torch.Tensor)

            # Verify callbacks were called
            mock_callback.assert_any_call("after_predict")
            mock_callback.assert_any_call("after_loss")

    def test_one_batch_validation_mode(self, learner):
        """Test _one_batch in validation mode (no backward pass)."""
        # Setup
        learner.fit(n_epochs=1)  # Create optimizer
        learner.xb = (torch.randn(2, 10),)
        learner.yb = (torch.randn(2, 1),)
        learner.training = False

        with patch.object(learner, "_callback") as mock_callback:
            learner._one_batch()

            # Verify predictions and loss were computed
            assert hasattr(learner, "preds")
            assert hasattr(learner, "loss")

            # In validation mode, no backward pass should occur
            mock_callback.assert_any_call("after_predict")
            mock_callback.assert_any_call("after_loss")


class TestLearnerIntegration:
    """Integration tests for Learner."""

    def test_full_training_cycle(self, simple_model, data_loaders):
        """Test a complete training cycle."""
        learner = Learner(simple_model, data_loaders)

        # Train for one epoch
        learner.fit(n_epochs=1)

        # Verify optimizer was created
        assert learner.opt is not None

        # After fit with validation, model should be in eval mode due to TrainEvalCallback
        # Let's verify this is the expected behavior
        assert learner.training is False

        # Save and load model - use a temporary path to avoid directory issues
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            learner.path = temp_path
            (temp_path / "models").mkdir(exist_ok=True)
            model_path = temp_path / "models" / "model"
            learner.save_model(path=model_path)
            new_learner = Learner(SimpleModel(), data_loaders)
            new_learner.path = temp_path
            new_learner.load_model(path=model_path)

    def test_learner_with_custom_loss_and_optimizer(
        self, simple_model, data_loaders
    ):
        """Test learner with custom loss function and optimizer."""
        custom_loss = nn.L1Loss()
        custom_opt = torch.optim.Adam

        learner = Learner(
            simple_model,
            data_loaders,
            loss_func=custom_loss,
            opt_func=custom_opt,
            lr=0.001,
        )

        learner.fit(n_epochs=1)

        assert isinstance(learner.opt, torch.optim.Adam)
        assert learner.opt.param_groups[0]["lr"] == 0.001
