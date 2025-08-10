"""
Scheduling callbacks for hyperparameter management during training.

This module provides commonly used schedulers for hyperparameters such as learning rate.
It is very important to remember to apply hyperparameter updates (such as learning rate)
after the optimizer's update. This means that it should be either in `before_batch` OR
`after_batch` in our framework.

We have two choices to schedule hyperparameters:

1. Use any subclass of `Scheduler` such as `BatchScheduler` and pass any scheduler
   from PyTorch's schedulers (see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
2. Use `ParamScheduler` and pass it any callable that takes the position and returns
   the hyperparameter value such as `exp_sched`

Functions
---------
annealer : Callable
    Decorator to create annealer functions.
no_sched : Callable
    Constant scheduler that returns the start value regardless of position.
lin_sched : Callable
    Linear scheduler that interpolates linearly between start and end values.
cos_sched : Callable
    Cosine scheduler that interpolates using cosine annealing.
exp_sched : Callable
    Exponential scheduler that interpolates exponentially between start and end values.
cos_1cycle_anneal : Callable
    Combine two cosine schedulers for 1-cycle learning rate scheduling.
combine_scheds : Callable
    Combine multiple schedulers, each running for a given percentage of training.

Classes
-------
ParamScheduler : Callback
    Callback to schedule hyperparameter values during training.
Scheduler : Callback
    Base scheduler to change hyperparameters using PyTorch schedulers.
BatchScheduler : Scheduler
    Scheduler that updates hyperparameters after every batch.
EpochScheduler : Scheduler
    Scheduler that updates hyperparameters after every epoch.

Examples
--------
>>> # Using ParamScheduler with exponential scheduling
>>> scheduler = ParamScheduler('lr', exp_sched(1e-3, 1e-5))
>>> learner.add_callback(scheduler)

>>> # Using BatchScheduler with PyTorch's OneCycleLR
>>> from torch.optim.lr_scheduler import OneCycleLR
>>> from functools import partial
>>> scheduler = BatchScheduler(partial(OneCycleLR, max_lr=1e-2, total_steps=1000))
>>> learner.add_callback(scheduler)
"""

import math
from functools import partial, wraps
from typing import Callable

import torch

from ..utils.utils import listify
from .core import Callback


def annealer(func: Callable):
    """
    Decorator to create annealer functions.

    This decorator wraps a function to create a partial function that can be used
    as a scheduler. The wrapped function should accept start, end, and position
    parameters and return the scheduled value.

    Parameters
    ----------
    func : Callable
        The function to be wrapped. Should have signature func(start, end, pos).

    Returns
    -------
    Callable
        A partial function that can be used as a scheduler.

    Examples
    --------
    >>> @annealer
    >>> def my_scheduler(start, end, pos):
    ...     return start + (end - start) * pos
    >>> scheduler = my_scheduler(0.1, 0.01)
    >>> value = scheduler(0.5)  # pos = 0.5
    """
    wraps(func)

    def annealer_wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)

    return annealer_wrapper


@annealer
def no_sched(start, end, pos):
    """
    Constant scheduler that returns the start value regardless of position.

    Parameters
    ----------
    start : float
        The constant value to return.
    end : float
        Not used in this scheduler (kept for interface consistency).
    pos : float
        Current position in training (0 to 1). Not used in this scheduler.

    Returns
    -------
    float
        The start value (constant).

    Examples
    --------
    >>> scheduler = no_sched(0.01, 0.001)
    >>> scheduler(0.5)  # Returns 0.01 regardless of position
    0.01
    """
    return start


@annealer
def lin_sched(start, end, pos):
    """
    Linear scheduler that interpolates linearly between start and end values.

    Parameters
    ----------
    start : float
        The starting value.
    end : float
        The ending value.
    pos : float
        Current position in training (0 to 1).

    Returns
    -------
    float
        The linearly interpolated value.

    Examples
    --------
    >>> scheduler = lin_sched(0.01, 0.001)
    >>> scheduler(0.5)  # Returns 0.0055 (halfway between 0.01 and 0.001)
    0.0055
    """
    return start + (end - start) * pos


@annealer
def cos_sched(start, end, pos):
    """
    Cosine scheduler that interpolates using cosine annealing.

    Parameters
    ----------
    start : float
        The starting value.
    end : float
        The ending value.
    pos : float
        Current position in training (0 to 1).

    Returns
    -------
    float
        The cosine-interpolated value.

    Examples
    --------
    >>> scheduler = cos_sched(0.01, 0.001)
    >>> scheduler(0.0)  # Returns start value
    0.01
    >>> scheduler(1.0)  # Returns end value
    0.001
    """
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def exp_sched(start, end, pos):
    """
    Exponential scheduler that interpolates exponentially between start and end values.

    Parameters
    ----------
    start : float
        The starting value.
    end : float
        The ending value.
    pos : float
        Current position in training (0 to 1).

    Returns
    -------
    float
        The exponentially interpolated value.

    Examples
    --------
    >>> scheduler = exp_sched(0.01, 0.001)
    >>> scheduler(0.0)  # Returns start value
    0.01
    >>> scheduler(1.0)  # Returns end value
    0.001
    """
    return start * (end / start) ** pos


def cos_1cycle_anneal(start, high, end):
    """
    Combine two cosine schedulers for 1-cycle learning rate scheduling.

    Creates a list of two cosine schedulers where the first scheduler goes from
    `start` to `high` and the second scheduler goes from `high` to `end`.

    Parameters
    ----------
    start : float
        The starting value.
    high : float
        The peak value in the middle of training.
    end : float
        The ending value.

    Returns
    -------
    list
        A list containing two cosine schedulers for the 1-cycle policy.

    Examples
    --------
    >>> schedulers = cos_1cycle_anneal(0.001, 0.01, 0.0001)
    >>> len(schedulers)  # Returns 2 schedulers
    2
    """
    return [cos_sched(start, high), cos_sched(high, end)]


def combine_scheds(pcts, scheds):
    """
    Combine multiple schedulers, each running for a given percentage of training.

    Parameters
    ----------
    pcts : list[float]
        List of percentages (should sum to 1.0) indicating how much of training
        each scheduler should run for.
    scheds : list[Callable]
        List of scheduler functions corresponding to each percentage.

    Returns
    -------
    Callable
        A combined scheduler function that switches between schedulers based on
        the current training position.

    Raises
    ------
    AssertionError
        If the number of percentages doesn't match the number of schedulers,
        if percentages don't sum to 1.0, or if any percentage is negative.

    Examples
    --------
    >>> scheduler1 = lin_sched(0.01, 0.005)
    >>> scheduler2 = cos_sched(0.005, 0.001)
    >>> combined = combine_scheds([0.6, 0.4], [scheduler1, scheduler2])
    >>> # First 60% of training uses linear scheduler, last 40% uses cosine
    """
    assert len(pcts) == len(scheds), "Each scheduler should have its `pct`."
    assert sum(pcts) == 1.0, "Sum of the `pcts` should be equal to 1."
    pcts = torch.tensor([0] + listify(pcts))
    assert (pcts >= 0).all(), "All percentages should be non-negative."
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner


class ParamScheduler(Callback):
    """
    Callback to schedule hyperparameter values during training.

    This class is used to schedule the values of hyperparameters (such as learning
    rate) during the training process. It can handle different schedulers for
    different parameter groups in the optimizer.

    Attributes
    ----------
    order : int
        The order in which this callback should be executed (default: 60).
    pname : str
        The name of the hyperparameter to be scheduled.
    sched_funcs : list[Callable] | tuple[Callable]
        List or tuple of scheduler functions for each parameter group.

    Examples
    --------
    >>> # Schedule learning rate with exponential decay
    >>> scheduler = ParamScheduler('lr', exp_sched(0.01, 0.001))
    >>> learner.add_callback(scheduler)

    >>> # Different schedulers for different parameter groups
    >>> schedulers = [lin_sched(0.01, 0.001), cos_sched(0.005, 0.0005)]
    >>> scheduler = ParamScheduler('lr', schedulers)
    >>> learner.add_callback(scheduler)
    """

    order: int = 60

    def __init__(
        self, pname: str, sched_funcs: list[Callable] | tuple[Callable]
    ) -> None:
        """
        Initialize the parameter scheduler.

        Parameters
        ----------
        pname : str
            The name of the hyperparameter to be scheduled (e.g., 'lr' for learning rate).
        sched_funcs : list[Callable] | tuple[Callable]
            A list or tuple of schedulers for each parameter group. Each scheduler
            should accept a single argument (position) and return a value for the
            hyperparameter. If a single scheduler is provided, it will be applied
            to all parameter groups.
        """
        super().__init__()
        self.pname = pname
        self.sched_funcs = sched_funcs

    def before_fit(self) -> None:
        """
        Prepare the scheduler before training begins.

        If a single scheduler function is provided, it will be replicated for
        all parameter groups in the optimizer.
        """
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def _update_value(self, pos: float) -> None:
        """
        Update the hyperparameter value for all parameter groups.

        Parameters
        ----------
        pos : float
            Current position in training (0 to 1).
        """
        for pg, sched_func in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = sched_func(pos)

    def before_batch(self) -> None:
        """
        Update hyperparameter values before each batch during training.

        Only updates values during training mode, not during validation.
        """
        if self.training:
            self._update_value(self.pct_train)


class Scheduler(Callback):
    """
    Base scheduler to change hyperparameters using PyTorch schedulers.

    This class provides a base implementation for using PyTorch's built-in
    schedulers. PyTorch's schedulers take the optimizer as the first argument,
    so it's important to pass a scheduler that has all its arguments already
    passed except the optimizer.

    Attributes
    ----------
    scheduler : Callable
        The scheduler function to be used.
    scheduler_object : torch.optim.lr_scheduler._LRScheduler
        The instantiated scheduler object.

    Notes
    -----
    PyTorch's schedulers take optimizer as the first argument. Therefore, it is
    important to pass the scheduler that has all its arguments already passed
    except the optimizer. This will be done in `Scheduler`'s `before_fit` method.

    Examples
    --------
    >>> from torch.optim.lr_scheduler import OneCycleLR
    >>> from functools import partial
    >>> scheduler = Scheduler(partial(OneCycleLR, max_lr=1e-2, total_steps=1000))
    >>> learner.add_callback(scheduler)
    """

    def __init__(self, scheduler) -> None:
        """
        Initialize the scheduler.

        Parameters
        ----------
        scheduler : Callable
            The scheduler function to be used. Should accept an optimizer as
            its first argument and return a scheduler object.
        """
        super().__init__()
        self.scheduler = scheduler

    def before_fit(self) -> None:
        """
        Create the scheduler object before training begins.

        Instantiates the scheduler with the current optimizer.
        """
        self.scheduler_object = self.scheduler(self.opt)

    def step(self) -> None:
        """
        Step the scheduler to update hyperparameter values.

        This method calls the step method of the underlying PyTorch scheduler.
        """
        self.scheduler_object.step()


class BatchScheduler(Scheduler):
    """
    Scheduler that updates hyperparameters after every batch.

    This scheduler applies the hyperparameter updates after each batch during
    training, which is useful for schedulers that need frequent updates like
    OneCycleLR.

    Examples
    --------
    >>> from torch.optim.lr_scheduler import OneCycleLR
    >>> from functools import partial
    >>> scheduler = BatchScheduler(partial(OneCycleLR, max_lr=1e-2, total_steps=1000))
    >>> learner.add_callback(scheduler)
    """

    def after_batch(self) -> None:
        """
        Update hyperparameters after each batch during training.

        Only updates values during training mode, not during validation.
        """
        if self.training:
            self.step()


class EpochScheduler(Scheduler):
    """
    Scheduler that updates hyperparameters after every epoch.

    This scheduler applies the hyperparameter updates after each epoch during
    training, which is useful for schedulers that need less frequent updates
    like StepLR or ReduceLROnPlateau.

    Examples
    --------
    >>> from torch.optim.lr_scheduler import StepLR
    >>> from functools import partial
    >>> scheduler = EpochScheduler(partial(StepLR, step_size=30, gamma=0.1))
    >>> learner.add_callback(scheduler)
    """

    def after_epoch(self) -> None:
        """
        Update hyperparameters after each epoch during training.

        Only updates values during training mode, not during validation.
        """
        if self.training:
            self.step()
