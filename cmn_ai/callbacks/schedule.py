"""
This module provides commonly used schedulers with hyperparameters such as
learning rate.
"""
import math
from functools import partial, wraps
from typing import Callable

import torch

from ..utils.utils import listify
from .core import Callback


def annealer(func: Callable):
    wraps(func)

    def annealer_wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)

    return annealer_wrapper


@annealer
def no_sched(start, end, pos):
    """Constant schedular."""
    return start


@annealer
def lin_sched(start, end, pos):
    """Linear scheduler."""
    return start + (end - start) * pos


@annealer
def cos_sched(start, end, pos):
    """Cosine scheduler."""
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def exp_sched(start, end, pos):
    """Exponential scheduler."""
    return start * (end / start) ** pos


def cos_1cycle_anneal(start, high, end):
    """
    combine two cosine schedulers where first scheduler goes from `start` to
    `high` and second scheduler goes from `high` to `end`.
    """
    return [cos_sched(start, high), cos_sched(high, end)]


def combine_scheds(pcts, scheds):
    """
    Combine muliple schedulers, each run for a given percentage of the training
    process.
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
    order: int = 60

    def __init__(self, pname, sched_funcs) -> None:
        self.pname = pname
        self.sched_funcs = sched_funcs

    def before_fit(self) -> None:
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def _update_value(self, pos: float) -> None:
        for pg, sched_func in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = sched_func(pos)

    def before_batch(self) -> None:
        if self.training:
            self._update_value(self.pct_train)


class Scheduler(Callback):
    """Base scheduler to change hyperparameters using `scheduler."""

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def before_fit(self) -> None:
        self.scheduler_object = self.scheduler(self.opt)

    def step(self) -> None:
        self.scheduler_object.step()


class BatchScheduler(Scheduler):
    """Change hyperparameters after every batch using `scheduler`."""

    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)

    def after_batch(self) -> None:
        if self.training:
            self.step()


class EpochScheduler(Scheduler):
    """Change hyperparameters after every epoch using `scheduler`."""

    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)

    def after_epoch(self) -> None:
        if self.training:
            self.step()
