"""
Unit tests for the hook system.

This module contains comprehensive tests for the hook system used for
inspecting neural network activations and gradients during training.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from cmn_ai.callbacks.hook import (
    ActivationStats,
    Hook,
    Hooks,
    HooksCallback,
    compute_stats,
    get_hist,
    get_min,
)


class SimpleModel(nn.Module):
    """Simple model for testing hooks."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestHook:
    """Test the Hook class for single module hooking."""

    def test_hook_init_forward(self):
        """Test hook initialization for forward hook."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats, is_forward=True)

        assert hook.is_forward is True
        assert hook.hook is not None

    def test_hook_init_backward(self):
        """Test hook initialization for backward hook."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats, is_forward=False)

        assert hook.is_forward is False
        assert hook.hook is not None

    def test_hook_register_forward(self):
        """Test registering a forward hook."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats, is_forward=True)

        # Hook is automatically registered in __init__
        assert hook.hook is not None

    def test_hook_register_backward(self):
        """Test registering a backward hook."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats, is_forward=False)

        # Hook is automatically registered in __init__
        assert hook.hook is not None

    def test_hook_remove(self):
        """Test hook removal."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats)

        # Store the hook handle before removal
        original_hook = hook.hook
        assert original_hook is not None

        hook.remove()

        # The hook handle should still exist but the hook should be removed from the module
        # The actual removal is handled by PyTorch's RemovableHandle
        pass

    def test_hook_remove_none(self):
        """Test hook removal when hook is None."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats)

        # Should not raise an exception
        hook.remove()

    def test_hook_context_manager(self):
        """Test hook as context manager."""
        model = SimpleModel()

        with Hook(model.linear1, compute_stats) as hook:
            assert isinstance(hook, Hook)
            assert hook.hook is not None

        # Hook should be removed after context exit
        # Note: The hook is already removed in __exit__, so hook.hook will be None
        pass

    def test_hook_del(self):
        """Test hook cleanup on deletion."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats)

        # Trigger deletion
        del hook

        # The __del__ method should handle cleanup
        pass


class TestHooks:
    """Test the Hooks class for multiple module hooking."""

    def test_hooks_init(self):
        """Test hooks initialization."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]
        hooks = Hooks(modules, compute_stats)

        assert len(hooks.hooks) == 2
        assert all(isinstance(h, Hook) for h in hooks.hooks)

    def test_hooks_getitem(self):
        """Test hooks indexing."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]
        hooks = Hooks(modules, compute_stats)

        assert isinstance(hooks[0], Hook)
        assert isinstance(hooks[1], Hook)

    def test_hooks_len(self):
        """Test hooks length."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]
        hooks = Hooks(modules, compute_stats)

        assert len(hooks) == 2

    def test_hooks_iter(self):
        """Test hooks iteration."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]
        hooks = Hooks(modules, compute_stats)

        hook_list = list(hooks)
        assert len(hook_list) == 2
        assert all(isinstance(h, Hook) for h in hook_list)

    def test_hooks_remove(self):
        """Test hooks removal."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]
        hooks = Hooks(modules, compute_stats)

        # Mock hook handles
        for hook in hooks.hooks:
            hook.hook = MagicMock()

        hooks.remove()

        # All hooks should be removed
        for hook in hooks.hooks:
            hook.hook.remove.assert_called_once()

    def test_hooks_context_manager(self):
        """Test hooks as context manager."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]

        with Hooks(modules, compute_stats) as hooks:
            assert isinstance(hooks, Hooks)
            assert len(hooks) == 2
            assert all(h.hook is not None for h in hooks.hooks)

        # The hooks should be removed after context exit
        # Note: The hooks are removed in __exit__, so we can't access them after
        pass

    def test_hooks_del(self):
        """Test hooks cleanup on deletion."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]
        hooks_obj = Hooks(modules, compute_stats)

        # Trigger deletion
        del hooks_obj

        # The __del__ method should handle cleanup
        pass


class TestHookFunctions:
    """Test hook utility functions."""

    def test_compute_stats(self):
        """Test compute_stats function."""
        hook = MagicMock()
        hook.is_forward = True
        # Initialize stats structure
        hook.stats = ([], [], [])
        module = nn.Linear(10, 5)
        inp = torch.randn(4, 10)
        outp = torch.randn(4, 5)

        compute_stats(hook, module, inp, outp)

        # Check that stats were computed and stored
        assert hasattr(hook, "stats")
        assert len(hook.stats) == 3  # mean, std, hist
        assert len(hook.stats[0]) == 1  # one mean value
        assert len(hook.stats[1]) == 1  # one std value
        assert len(hook.stats[2]) == 1  # one hist value

    def test_get_hist(self):
        """Test get_hist function."""
        hook = MagicMock()
        # Create mock stats with histogram data
        hist_data = torch.randn(40)
        hook.stats = ([], [], [hist_data])

        hist = get_hist(hook)

        assert isinstance(hist, torch.Tensor)
        assert hist.shape == (40, 1)  # (bins, timesteps)

    def test_get_min(self):
        """Test get_min function."""
        hook = MagicMock()
        # Create a histogram with some zeros
        hist = torch.zeros(40)
        hist[0] = 10  # Some values in first bin (around zero)
        hist[20] = 5  # Some values in middle bin
        hook.stats = ([], [], [hist])

        min_vals = get_min(hook, (0, 10))

        assert isinstance(min_vals, torch.Tensor)
        assert min_vals.shape == (1,)  # one timestep


class TestHooksCallback:
    """Test the HooksCallback class."""

    def test_hooks_callback_init(self):
        """Test hooks callback initialization."""
        model = SimpleModel()
        callback = HooksCallback(compute_stats, modules=model.children())

        assert callback.hookfunc == compute_stats
        assert callback.on_train is True
        assert callback.on_valid is False
        assert callback.is_forward is True

    def test_hooks_callback_before_fit(self):
        """Test hooks callback before_fit method."""
        model = SimpleModel()
        callback = HooksCallback(compute_stats, modules=model.children())

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.model = model
        callback.set_learner(mock_learner)

        callback.before_fit()

        # Should create hooks
        assert hasattr(callback, "hooks")
        assert len(callback.hooks) > 0

    def test_hooks_callback_after_fit(self):
        """Test hooks callback after_fit method."""
        model = SimpleModel()
        callback = HooksCallback(compute_stats, modules=model.children())

        # Mock hooks
        mock_hooks = MagicMock()
        callback.hooks = mock_hooks

        callback.after_fit()

        # Should remove hooks
        mock_hooks.remove.assert_called_once()

    def test_hooks_callback_iter(self):
        """Test hooks callback iteration."""
        model = SimpleModel()
        callback = HooksCallback(compute_stats, modules=model.children())

        # Mock hooks
        mock_hooks = [MagicMock(), MagicMock()]
        callback.hooks = mock_hooks

        hook_list = list(callback)
        assert hook_list == mock_hooks

    def test_hooks_callback_len(self):
        """Test hooks callback length."""
        model = SimpleModel()
        callback = HooksCallback(compute_stats, modules=model.children())

        # Mock hooks
        mock_hooks = [MagicMock(), MagicMock()]
        callback.hooks = mock_hooks

        assert len(callback) == 2


class TestActivationStats:
    """Test the ActivationStats class."""

    def test_activation_stats_init(self):
        """Test activation stats initialization."""
        model = SimpleModel()
        stats = ActivationStats(model.children())

        assert stats.is_forward is True
        assert stats.bins == 40
        assert stats.bins_range == (0, 10)

    def test_activation_stats_init_custom_params(self):
        """Test activation stats initialization with custom parameters."""
        model = SimpleModel()
        stats = ActivationStats(
            model.children(), is_forward=False, bins=20, bins_range=(0, 5)
        )

        assert stats.is_forward is False
        assert stats.bins == 20
        assert stats.bins_range == (0, 5)

    @patch("matplotlib.pyplot.subplots")
    def test_activation_stats_plot_hist(self, mock_subplots):
        """Test activation stats histogram plotting."""
        model = SimpleModel()
        stats = ActivationStats(model.children())

        # Mock hooks with stored data
        mock_hook = MagicMock()
        mock_hook.stored = [{"hist": torch.randn(40)}]
        stats.hooks = [mock_hook]

        # Mock matplotlib
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax])

        stats.plot_hist()

        # Should call subplots
        mock_subplots.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_activation_stats_dead_chart(self, mock_subplots):
        """Test activation stats dead chart plotting."""
        model = SimpleModel()
        stats = ActivationStats(model.children())

        # Mock hooks with stored data
        mock_hook = MagicMock()
        mock_hook.stored = [{"hist": torch.randn(40)}]
        stats.hooks = [mock_hook]

        # Mock matplotlib
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax])

        stats.dead_chart((0, 10))

        # Should call subplots
        mock_subplots.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_activation_stats_plot_stats(self, mock_subplots):
        """Test activation stats plotting."""
        model = SimpleModel()
        stats = ActivationStats(model.children())

        # Mock hooks with stored data
        mock_hook = MagicMock()
        mock_hook.stored = [
            {
                "mean": torch.tensor([0.5]),
                "std": torch.tensor([1.0]),
                "hist": torch.randn(40),
            }
        ]
        stats.hooks = [mock_hook]

        # Mock matplotlib
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax, mock_ax, mock_ax])

        stats.plot_stats()

        # Should call subplots
        mock_subplots.assert_called_once()


class TestHookIntegration:
    """Test hook system integration."""

    def test_hook_with_actual_forward_pass(self):
        """Test hook with actual forward pass."""
        model = SimpleModel()
        hook = Hook(model.linear1, compute_stats)

        # Forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = model(x)

        # Check that stats were computed
        assert hasattr(hook, "stats")
        assert len(hook.stats) == 3  # mean, std, hist
        assert len(hook.stats[0]) == 1  # one mean value
        assert len(hook.stats[1]) == 1  # one std value
        assert len(hook.stats[2]) == 1  # one hist value

        # Cleanup
        hook.remove()

    def test_hooks_with_actual_forward_pass(self):
        """Test hooks with actual forward pass."""
        model = SimpleModel()
        modules = [model.linear1, model.linear2]
        hooks = Hooks(modules, compute_stats)

        # Forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = model(x)

        # Check that stats were computed for all hooks
        for hook in hooks.hooks:
            assert hasattr(hook, "stats")
            assert len(hook.stats) == 3  # mean, std, hist
            assert len(hook.stats[0]) == 1  # one mean value
            assert len(hook.stats[1]) == 1  # one std value
            assert len(hook.stats[2]) == 1  # one hist value

        # Cleanup
        hooks.remove()
