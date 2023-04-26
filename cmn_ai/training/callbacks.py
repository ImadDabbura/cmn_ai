import re
import tempfile
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar

from ..utils.data import default_device, to_cpu, to_device
from ..utils.utils import listify, setify


class CancelFitException(Exception):
    """Stop training and move to after_fit."""


class CancelEpochException(Exception):
    """Stop current epoch and move to after_epoch."""


class CancelTrainException(Exception):
    """Stop training current epoch and move to after_train."""


class CancelValidException(Exception):
    """Stop validation phase and move after_validate."""


class CancelBatchException(Exception):
    """Stop current batch and move to after_batch."""


class CancelStepException(Exception):
    """Skip stepping the optimizer."""


class CancelBackwardException(Exception):
    """Skip the backward pass and move to after_backward."""


class Callback:
    """Base class for all callbacks."""

    order = 0

    def set_learner(self, learner):
        self.learner = learner

    def __getattr__(self, k):
        return getattr(self.learner, k)

    @property
    def name(self):
        """
        Returns the name of the callback after removing the word `callback`
        and then convert it to snake (split words by underscores).
        """
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return Callback.camel2snake(name or "callback")

    def __call__(self, event_nm):
        method = getattr(self, event_nm, None)
        if method is not None:
            method()

    @staticmethod
    def camel2snake(name):
        """
        Convert name of callback by inserting underscores between small and capital
        letters. For example, `TestCallback` becomes `test_callback`.
        """
        pattern1 = re.compile("(.)([A-Z][a-z]+)")
        pattern2 = re.compile("([a-z0-9])([A-Z])")
        name = re.sub(pattern1, r"\1_\2", name)
        return re.sub(pattern2, r"\1_\2", name).lower()


class DeviceCallback(Callback):
    """Move batch and model to device."""

    def __init__(self, device=default_device):
        self.device = device

    def before_fit(self):
        self.model.to(self.device)

    def before_batch(self):
        self.learner.batch = to_device(self.batch, self.device)


class TrainEvalCallback(Callback):
    """
    Tracks the number of iterations and epoch done and set training and eval
    modes.
    """

    order = -10

    def before_fit(self):
        self.learner.n_iters = 0
        self.learner.pct_train = 0

    def after_batch(self):
        if self.training:
            self.learner.n_iters += 1
            self.learner.pct_train += 1 / (self.iters * self.n_epochs)

    def before_train(self):
        self.model.train()
        self.learner.training = True
        self.learner.pct_train = self.epoch / self.n_epochs

    def before_validate(self):
        self.model.eval()
        self.learner.training = False


class ProgressCallback(Callback):
    """Add progress bar as logger for tracking metrics."""

    _order = -20

    def before_fit(self):
        self.mbar = master_bar(range(self.n_epochs))
        self.mbar.on_iter_begin()
        # Overwrite default learner logger
        self.learner.logger = partial(self.mbar.write, table=True)

    def after_fit(self):
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.iter)

    def before_train(self):
        self.set_pb()

    def before_validate(self):
        self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)
        self.mbar.update(self.epoch)


class Recorder(Callback):
    """
    Keep track of losses and learning rates as training progress so we can
    plot them later.
    """

    _order = 50

    def before_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if self.training:
            for pg, lr in zip(self.opt.param_groups, self.lrs):
                lr.append(pg["lr"])
            self.losses.append(to_cpu(self.loss))

    def plot_lr(self, pgid=-1):
        """
        Plot learning rates in the parameter group id `pgid`, default to the last parameter group.
        """
        plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0):
        """
        Plot losses, optionally skip last `skip_last` losses.
        """
        n = len(self.losses) - skip_last
        plt.plot(self.losses[:n])

    def plot(self, skip_last=0, pgid=-1):
        """
        Plot both losses and learning rates.
        """
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(lrs[:n], losses[:n])
