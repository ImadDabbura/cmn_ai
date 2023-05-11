import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar
from torch.optim.lr_scheduler import ExponentialLR

from ..plot import get_grid
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

    def __init__(self, *params):
        self.params = listify(params)

    def before_fit(self):
        self.params_records = {
            params: [[] for _ in self.opt.param_groups]
            for params in self.params
        }
        self.losses = []

    def after_batch(self):
        if self.training:
            for param, param_records in self.params_records.items():
                for pg, param_record in zip(
                    self.opt.param_groups, param_records
                ):
                    param_record.append(pg[param])
            self.losses.append(to_cpu(self.loss))

    def plot_params(
        self,
        params: str | Iterable[str] = "lr",
        pgid: int = -1,
        skip_last: int = 0,
        figsize: tuple = (8, 6),
    ):
        """
        Plot all `params` values across all iterations of training.
        """
        params = listify(params)
        _, axes = get_grid(len(params), figsize=figsize)
        for (
            ax,
            param,
        ) in zip(axes.flatten(), params):
            ax.plot(self.params_records[param][pgid], label=param)
            ax.legend()

    def plot_loss(self, skip_last: int = 0):
        """
        Plot losses, optionally skip last `skip_last` losses.
        """
        n = len(self.losses) - skip_last
        plt.plot(self.losses[:n])

    def plot(self, pgid: int = -1, skip_last: int = 0):
        """
        Plot loss vs lr (log-scale) across all iterations of training.
        """
        n = len(self.losses) - skip_last
        plt.xscale("log")
        plt.plot(
            self.params_records["lr"][pgid][:n],
            self.losses[:n],
        )


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

    def _update_value(self, pos):
        for pg, sched_func in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = sched_func(pos)

    def before_batch(self):
        if self.training:
            self._update_value(self.pct_train)


# TODO: Change loss to smooth_loss
class LRFinder(Callback):
    """
    Try different learning rates using exponential schedule to help pick
    the best learning rate following [Cyclical Learning Rates for Training
    Neural Networks](https://arxiv.org/pdf/1506.01186.pdf). When done, plot
    learning rate vs loss.

    Parameters
    ----------
    start_lr : float, default=1e-7
        Start learning rate.
    end_lr : float, default=10.0
        Last learning rate in the schedule.
    num_iter : int, default=100
        Number of iterations to run the training.
    stop_div : bool, default
        Whether to stop training if loss diverges (loss > 4 * best_loss).
    max_mult : int, default=4
        Divergence threshold. If loss >= max_mult * minimum loss, stop
        training.
    """

    def __init__(
        self,
        gamma: int = 1.3,
        num_iter: int = 100,
        stop_div: bool = True,
        max_mult: int = 4,
    ):
        self.gamma = gamma
        self.num_iter = num_iter
        self.stop_div = stop_div
        self.max_mult = max_mult

    def before_fit(self):
        self.scheduler = ExponentialLR(self.opt, self.gamma)
        path = self.path / self.model_dir
        path.mkdir(parents=True, exist_ok=True)
        self.tmp_d = tempfile.TemporaryDirectory(dir=path)
        self.tmp_p = Path(self.tmp_d.name).stem
        self.save_model(path / self.tmp_p / "_tmp.pth", with_opt=True)
        self.best_loss = float("inf")

    def after_batch(self):
        if self.loss < self.best_loss:
            self.best_loss = self.loss
        if self.loss > self.max_mult * self.best_loss and self.stop_div:
            raise CancelFitException()
        if self.n_iters >= self.num_iter:
            raise CancelFitException()
        self.scheduler.step()

    def before_validate(self):
        raise CancelValidException()

    def after_fit(self):
        self.opt.zero_grad()
        tmp_f = self.path / self.model_dir / self.tmp_p / "_tmp.pth"
        if tmp_f.exists():
            self.load_model(tmp_f, with_opt=True)
            self.tmp_d.cleanup()


class BatchTransform(Callback):
    """
    Transform X as a batch using `tfm` callable before every batch.
    Apply transformation `tfm` on the batch as a whole.

    Parameters
    ----------
    tfm : Callback
        Transformation to apply on the batch.
    on_train : bool, default=True
        Whether to apply the transformation during training.
    on_valid : bool, default=True
        Whether to apply the transformation during validation.
    """

    _order = 2

    def __init__(
        self, tfm: Callback, on_train: bool = True, on_valid: bool = True
    ):
        self.tfm = tfm
        self.on_train = on_train
        self.on_valid = on_valid

    def before_batch(self):
        if (self.on_train and self.training) or (
            self.on_valid and not self.training
        ):
            self.learner.batch = self.tfm(self.batch)


class SingleBatchCB(Callback):
    """
    Run 1 training/validation batch and stop by raising `CancelFitException`.
    Useful for debugging or want to check few parameters after 1 batch.
    """

    _order = 1

    def after_batch(self):
        raise CancelFitException()
