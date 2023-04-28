import re
import tempfile
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar

from ..utils.data import default_device, to_cpu, to_device
from ..utils.utils import listify
from .core import Callback, CancelFitException, CancelValidException
from .schedule import exp_sched


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


class AvgStats:
    """
    Base class that compute average loss and `metrics` stats after every batch.
    """

    def __init__(self, metrics, training=True):
        self.metrics = listify(metrics)
        self.training = training

    def reset(self):
        self.tot_loss = torch.tensor(0.0)
        self.count = 0
        self.tot_metrics = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        """Returns a list of both loss and metrics."""
        return [self.tot_loss.item()] + self.tot_metrics

    @property
    def avg_stats(self):
        """Returns the average of loss/metrics."""
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.training else 'valid'}: {self.avg_stats}"

    def accumulate(self, learner):
        """Evaluate metrics and accumulate them to at the epoch level."""
        bs = len(learner.xb)
        self.count += bs
        self.tot_loss += learner.loss * bs
        for i, metric in enumerate(self.metrics):
            self.tot_metrics[i] += metric(learner.preds, learner.yb) * bs


class AvgStatsCallback(Callback):
    """
    Compute average loss/metrics after every batch and log the stats after
    every epoch.
    """

    _order = -10

    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def before_fit(self):
        metrics_names = ["loss"] + [
            metric.__name__ for metric in self.train_stats.metrics
        ]
        names = (
            ["epoch"]
            + [f"train_{name}" for name in metrics_names]
            + [f"valid_{name}" for name in metrics_names]
            + ["time"]
        )
        self.logger(names)

    def before_epoch(self):
        """Reset metrics/loss."""
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        """Evaluate metrics and accumulate them."""
        stats = self.train_stats if self.training else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.learner)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


class BatchTransformXCallback(Callback):
    """Transform X as a batch using `tfm` callable before every batch."""

    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def before_batch(self):
        self.learner.xb = self.tfm(self.learner.xb)


class ModelResetter(Callback):
    """
    Reset model's parameters. This is very useful in the context of NLP since
    we always reset hidden state. The assumption here is that `model` has a
    `reset` method that knows which parameters to reset and how.
    """

    def before_train(self):
        self.model.reset()

    def before_validate(self):
        self.model.reset()

    def after_fit(self):
        self.model.reset()


class ParamScheduler(Callback):
    _order = 60

    def __init__(self, pname, sched_funcs):
        self.pname = pname
        self.sched_funcs = sched_funcs

    def before_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    # def set_param(self, pos=None):
    #     assert_msg = (
    #         f"Number of schedulers should match number of parameter groups, "
    #         f"{print(len(self.opt.param_groups), len(self.sched_funcs))}"
    #     )
    #     assert len(self.opt.param_groups) == len(self.sched_funcs), assert_msg
    #     for pg, sched_func in zip(self.opt.param_groups, self.sched_funcs):
    #         pg[self.pname] = sched_func(self.pct_train if pos is None else pos)

    def _update_value(self, pos):
        for pg, sched_func in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = sched_func(pos)
        self

    def before_batch(self):
        if self.training:
            self._update_value(self.pct_train)


class LRFinder(ParamScheduler):
    "Training with exponentially growing learning rate"

    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        super().__init__("lr", exp_sched(start_lr, end_lr))
        self.num_it = num_it
        self.stop_div = stop_div

    def before_fit(self):
        super().before_fit()
        path = self.path / self.model_dir
        path.mkdir(parents=True, exist_ok=True)
        self.tmp_d = tempfile.TemporaryDirectory(dir=path)
        self.tmp_p = Path(self.tmp_d.name).stem
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            },
            path / self.tmp_p / "_tmp.pth",
        )
        self.best_loss = float("inf")

    def before_batch(self):
        self._update_value(self.n_iters / self.num_it)

    def after_batch(self):
        # super().after_batch()
        if self.loss < self.best_loss:
            self.best_loss = self.loss
        if self.loss > 4 * self.best_loss and self.stop_div:
            raise CancelFitException()
        if self.n_iters >= self.num_it:
            raise CancelFitException()

    def before_validate(self):
        raise CancelValidException()

    def after_fit(self):
        self.opt.zero_grad()
        tmp_f = self.path / self.model_dir / self.tmp_p / "_tmp.pth"
        if tmp_f.exists():
            checkpoint = torch.load(tmp_f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.tmp_d.cleanup()
