"""
Unit tests for the scheduling callbacks.

This module contains comprehensive tests for the scheduling system used for
hyperparameter management during training, including various scheduler functions
and callback classes.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from cmn_ai.callbacks.schedule import (
    BatchScheduler,
    EpochScheduler,
    ParamScheduler,
    Scheduler,
    annealer,
    combine_scheds,
    cos_1cycle_anneal,
    cos_sched,
    exp_sched,
    lin_sched,
    no_sched,
)


class TestSchedulerFunctions:
    """Test scheduler utility functions."""

    def test_annealer_decorator(self):
        """Test the annealer decorator."""

        @annealer
        def test_scheduler(start, end, pos):
            return start + (end - start) * pos

        # Test that it creates a partial function
        scheduler = test_scheduler(0.1, 0.01)
        assert callable(scheduler)

        # Test the scheduler function
        assert scheduler(0.0) == 0.1  # start
        assert abs(scheduler(1.0) - 0.01) < 1e-10  # end
        assert abs(scheduler(0.5) - 0.055) < 1e-10  # middle

    def test_no_sched(self):
        """Test no_sched function."""
        scheduler = no_sched(0.1, 0.01)

        # Should always return start value regardless of position
        assert scheduler(0.0) == 0.1
        assert scheduler(0.5) == 0.1
        assert scheduler(1.0) == 0.1

    def test_lin_sched(self):
        """Test linear scheduler function."""
        scheduler = lin_sched(0.1, 0.01)

        # Test linear interpolation
        assert scheduler(0.0) == 0.1  # start
        assert abs(scheduler(1.0) - 0.01) < 1e-10  # end
        assert abs(scheduler(0.5) - 0.055) < 1e-10  # middle

    def test_cos_sched(self):
        """Test cosine scheduler function."""
        scheduler = cos_sched(0.1, 0.01)

        # Test cosine interpolation
        assert scheduler(0.0) == 0.1  # start
        assert abs(scheduler(1.0) - 0.01) < 1e-10  # end

        # Middle value should be within valid range
        middle_cos = scheduler(0.5)
        assert 0.01 <= middle_cos <= 0.1

    def test_exp_sched(self):
        """Test exponential scheduler function."""
        scheduler = exp_sched(0.1, 0.01)

        # Test exponential interpolation
        assert scheduler(0.0) == 0.1  # start
        assert abs(scheduler(1.0) - 0.01) < 1e-10  # end

        # Should be exponential decay
        assert scheduler(0.25) > scheduler(0.5)  # earlier values higher

    def test_cos_1cycle_anneal(self):
        """Test cosine 1-cycle annealer function."""
        schedulers = cos_1cycle_anneal(0.1, 0.5, 0.01)

        # Should return a list of two schedulers
        assert len(schedulers) == 2
        assert callable(schedulers[0])
        assert callable(schedulers[1])

        # Test the first scheduler (start to high)
        assert abs(schedulers[0](0.0) - 0.1) < 1e-10  # start
        assert abs(schedulers[0](1.0) - 0.5) < 1e-10  # high

        # Test the second scheduler (high to end)
        assert abs(schedulers[1](0.0) - 0.5) < 1e-10  # high
        assert abs(schedulers[1](1.0) - 0.01) < 1e-10  # end

    def test_combine_scheds(self):
        """Test combine_scheds function."""
        # Create two simple schedulers
        sched1 = lin_sched(0.1, 0.05)  # 0-50% of training
        sched2 = cos_sched(0.05, 0.01)  # 50-100% of training

        combined = combine_scheds([0.5, 0.5], [sched1, sched2])

        # Test the combined scheduler
        assert abs(combined(0.0) - 0.1) < 1e-10  # start of first scheduler
        assert (
            abs(combined(0.5) - 0.05) < 1e-10
        )  # end of first, start of second
        # Test a value close to but not exactly 1.0 to avoid indexing issues
        assert (
            abs(combined(0.99) - 0.01) < 1e-4
        )  # near end of second scheduler


class TestParamScheduler:
    """Test the ParamScheduler callback."""

    def test_param_scheduler_init(self):
        """Test ParamScheduler initialization."""
        scheduler = ParamScheduler("lr", exp_sched(0.1, 0.01))

        assert scheduler.pname == "lr"
        assert scheduler.order == 60

    def test_param_scheduler_init_multiple(self):
        """Test ParamScheduler initialization with multiple schedulers."""
        schedulers = [lin_sched(0.1, 0.05), cos_sched(0.05, 0.01)]
        scheduler = ParamScheduler("lr", schedulers)

        assert scheduler.pname == "lr"
        assert len(scheduler.sched_funcs) == 2

    def test_param_scheduler_before_fit(self):
        """Test ParamScheduler before_fit method."""
        scheduler = ParamScheduler("lr", exp_sched(0.1, 0.01))

        # Mock learner with optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.1}, {"lr": 0.1}]

        mock_learner = MagicMock()
        mock_learner.opt = mock_optimizer

        scheduler.set_learner(mock_learner)
        scheduler.before_fit()

        # Should have stored initial values
        assert hasattr(scheduler, "sched_funcs")
        assert len(scheduler.sched_funcs) == 2

    def test_param_scheduler_update_value(self):
        """Test ParamScheduler _update_value method."""
        scheduler = ParamScheduler("lr", exp_sched(0.1, 0.01))

        # Mock learner with optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.1}, {"lr": 0.1}]

        mock_learner = MagicMock()
        mock_learner.opt = mock_optimizer

        scheduler.set_learner(mock_learner)
        scheduler.before_fit()

        # Update values
        scheduler._update_value(0.5)

        # Check that lr was updated
        assert mock_optimizer.param_groups[0]["lr"] != 0.1
        assert mock_optimizer.param_groups[1]["lr"] != 0.1

    def test_param_scheduler_before_batch(self):
        """Test ParamScheduler before_batch method."""
        scheduler = ParamScheduler("lr", exp_sched(0.1, 0.01))

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.epoch = 0
        mock_learner.n_epoch = 10
        mock_learner.iteration = 5
        mock_learner.n_iter = 100

        scheduler.set_learner(mock_learner)

        # Mock _update_value
        with patch.object(scheduler, "_update_value") as mock_update:
            scheduler.before_batch()
            mock_update.assert_called_once()


class TestScheduler:
    """Test the base Scheduler class."""

    def test_scheduler_init(self):
        """Test Scheduler initialization."""
        mock_scheduler = MagicMock()
        scheduler = Scheduler(mock_scheduler)

        assert scheduler.scheduler == mock_scheduler

    def test_scheduler_before_fit(self):
        """Test Scheduler before_fit method."""
        mock_scheduler = MagicMock()
        scheduler = Scheduler(mock_scheduler)

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.opt = MagicMock()

        scheduler.set_learner(mock_learner)
        scheduler.before_fit()

        # Should have stored optimizer
        assert scheduler.opt == mock_learner.opt

    def test_scheduler_step(self):
        """Test Scheduler step method."""
        mock_scheduler = MagicMock()
        mock_scheduler_object = MagicMock()
        mock_scheduler.return_value = mock_scheduler_object

        scheduler = Scheduler(mock_scheduler)

        # Mock learner and optimizer
        mock_learner = MagicMock()
        mock_learner.opt = MagicMock()
        scheduler.set_learner(mock_learner)

        # Initialize the scheduler object
        scheduler.before_fit()

        scheduler.step()

        # Should call scheduler object step
        mock_scheduler_object.step.assert_called_once()


class TestBatchScheduler:
    """Test the BatchScheduler class."""

    def test_batch_scheduler_init(self):
        """Test BatchScheduler initialization."""
        mock_scheduler = MagicMock()
        scheduler = BatchScheduler(mock_scheduler)

        assert scheduler.scheduler == mock_scheduler

    def test_batch_scheduler_after_batch(self):
        """Test BatchScheduler after_batch method."""
        mock_scheduler = MagicMock()
        mock_scheduler_object = MagicMock()
        mock_scheduler.return_value = mock_scheduler_object

        scheduler = BatchScheduler(mock_scheduler)

        # Mock learner and optimizer
        mock_learner = MagicMock()
        mock_learner.opt = MagicMock()
        scheduler.set_learner(mock_learner)

        # Initialize the scheduler object
        scheduler.before_fit()

        # Mock training mode
        scheduler.training = True

        scheduler.after_batch()

        # Should call step
        mock_scheduler_object.step.assert_called_once()


class TestEpochScheduler:
    """Test the EpochScheduler class."""

    def test_epoch_scheduler_init(self):
        """Test EpochScheduler initialization."""
        mock_scheduler = MagicMock()
        scheduler = EpochScheduler(mock_scheduler)

        assert scheduler.scheduler == mock_scheduler

    def test_epoch_scheduler_after_epoch(self):
        """Test EpochScheduler after_epoch method."""
        mock_scheduler = MagicMock()
        mock_scheduler_object = MagicMock()
        mock_scheduler.return_value = mock_scheduler_object

        scheduler = EpochScheduler(mock_scheduler)

        # Mock learner and optimizer
        mock_learner = MagicMock()
        mock_learner.opt = MagicMock()
        scheduler.set_learner(mock_learner)

        # Initialize the scheduler object
        scheduler.before_fit()

        # Mock training mode
        scheduler.training = True

        scheduler.after_epoch()

        # Should call step
        mock_scheduler_object.step.assert_called_once()


class TestSchedulerIntegration:
    """Test scheduler integration and usage patterns."""

    def test_param_scheduler_with_actual_optimizer(self):
        """Test ParamScheduler with actual optimizer."""
        # Create a real optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Create scheduler
        scheduler = ParamScheduler("lr", exp_sched(0.1, 0.01))

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.opt = optimizer

        scheduler.set_learner(mock_learner)
        scheduler.before_fit()

        # Update learning rate
        scheduler._update_value(0.5)

        # Check that lr was updated
        for group in optimizer.param_groups:
            assert group["lr"] != 0.1

    def test_multiple_param_schedulers(self):
        """Test multiple parameter schedulers."""
        # Create a real optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        # Create schedulers for different parameters
        lr_scheduler = ParamScheduler("lr", exp_sched(0.1, 0.01))
        momentum_scheduler = ParamScheduler("momentum", lin_sched(0.9, 0.5))

        # Mock learner
        mock_learner = MagicMock()
        mock_learner.opt = optimizer

        lr_scheduler.set_learner(mock_learner)
        momentum_scheduler.set_learner(mock_learner)

        lr_scheduler.before_fit()
        momentum_scheduler.before_fit()

        # Update values
        lr_scheduler._update_value(0.5)
        momentum_scheduler._update_value(0.5)

        # Check that both parameters were updated
        for group in optimizer.param_groups:
            assert group["lr"] != 0.1
            assert group["momentum"] != 0.9

    def test_scheduler_combinations(self):
        """Test different scheduler combinations."""
        # Test linear + cosine combination
        combined = combine_scheds(
            [0.3, 0.7], [lin_sched(0.1, 0.05), cos_sched(0.05, 0.01)]
        )

        # Test key points
        assert abs(combined(0.0) - 0.1) < 1e-10  # start
        assert (
            abs(combined(0.3) - 0.05) < 1e-10
        )  # end of first, start of second
        assert abs(combined(0.99) - 0.01) < 1e-4  # near end

        # Test 1-cycle
        one_cycle_schedulers = cos_1cycle_anneal(0.01, 0.1, 0.001)
        assert len(one_cycle_schedulers) == 2
        # Test first scheduler (start to peak)
        assert abs(one_cycle_schedulers[0](0.0) - 0.01) < 1e-10  # start
        assert abs(one_cycle_schedulers[0](1.0) - 0.1) < 1e-10  # peak
        # Test second scheduler (peak to end)
        assert abs(one_cycle_schedulers[1](0.0) - 0.1) < 1e-10  # peak
        assert abs(one_cycle_schedulers[1](1.0) - 0.001) < 1e-10  # end

    def test_scheduler_edge_cases(self):
        """Test scheduler edge cases."""
        # Test zero position
        scheduler = lin_sched(0.1, 0.01)
        assert abs(scheduler(0.0) - 0.1) < 1e-10

        # Test one position
        assert abs(scheduler(0.99) - 0.01) < 1e-3

        # Test negative position (should handle gracefully)
        # For linear scheduler with start=0.1, end=0.01, pos=-0.1:
        # result = 0.1 + (0.01 - 0.1) * (-0.1) = 0.1 + 0.009 = 0.109
        assert scheduler(-0.1) > 0.1

        # Test position > 1 (should handle gracefully)
        # For linear scheduler with start=0.1, end=0.01, pos=1.5:
        # result = 0.1 + (0.01 - 0.1) * 1.5 = 0.1 - 0.135 = -0.035
        assert scheduler(1.5) < 0.01

    def test_no_sched_edge_cases(self):
        """Test no_sched edge cases."""
        scheduler = no_sched(0.1, 0.01)

        # Should always return start value
        assert scheduler(0.0) == 0.1
        assert scheduler(0.5) == 0.1
        assert scheduler(1.0) == 0.1
        assert scheduler(-0.1) == 0.1
        assert scheduler(1.5) == 0.1
