"""
Almost all training's related callbacks that will tweak/customize the
training/validation loop is in this module.

Raises
------
CancelFitException
    Stop training and move to after_fit.
CancelEpochException
    Stop current epoch and move to after_epoch.
CancelTrainException
    Stop training current epoch and move to after_train.
CancelValidException
    Stop validation phase and move after_validate.
CancelBatchException
    Stop current batch and move to after_batch.
CancelStepException
    Skip stepping the optimizer.
CancelBackwardException
    Skip the backward pass and move to after_backward.
"""
import tempfile
import time
from copy import copy
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import fastcore.all as fc
import matplotlib.pyplot as plt
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics import Mean

from ..losses import NoneReduce, reduce_loss
from ..plot import get_grid
from ..utils.data import default_device, to_cpu, to_device
from ..utils.utils import listify
from .core import Callback, CancelFitException, CancelValidException
from .schedule import exp_sched


class DeviceCallback(Callback):
    """Move batch and model to `device`."""

    def __init__(self, device: torch.device = default_device) -> None:
        self.device = device

    def before_fit(self) -> None:
        self.model.to(self.device)

    def before_batch(self) -> None:
        self.learner.batch = to_device(self.batch, self.device)


class TrainEvalCallback(Callback):
    """
    Tracks the number of iterations and epoch done and set training and eval
    modes.
    """

    order = -10

    def before_fit(self) -> None:
        self.learner.n_iters = 0
        self.learner.pct_train = 0

    def after_batch(self) -> None:
        if self.training:
            self.learner.n_iters += 1
            self.learner.pct_train += 1 / (self.iters * self.n_epochs)

    def before_train(self) -> None:
        self.model.train()
        self.learner.training = True
        self.learner.pct_train = self.epoch / self.n_epochs

    def before_validate(self) -> None:
        self.model.eval()
        self.learner.training = False


class ProgressCallback(Callback):
    """
    Track progress of training using progress bar as well as to plot losses
    (train and valid), which allows us to have live feedback of the model's
    performance while it is still training.

    Parameters
    ----------
    plot : bool, default=True
        Whether to plot train/valid losses during training.
    """

    order = -20

    def __init__(self, plot: bool = True) -> None:
        self.plot = plot
        self.train_losses = []
        self.valid_losses = []

    def before_fit(self) -> None:
        self.learner.epochs = self.mbar = master_bar(range(self.n_epochs))
        self.mbar.on_iter_begin()
        # Overwrite default learner logger
        self.learner.logger = partial(self.mbar.write, table=True)

    def after_fit(self) -> None:
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.iter)
        self.pb.comment = f"{self.loss:.3f}"
        if self.plot and hasattr(self.learner, "metrics") and self.training:
            self.train_losses.append(self.loss.item())
            if self.valid_losses:
                self.mbar.update_graph(
                    [
                        [fc.L.range(self.train_losses), self.train_losses],
                        [
                            fc.L.range(self.epoch).map(
                                lambda x: (x + 1) * len(self.dls.train)
                            ),
                            self.valid_losses,
                        ],
                    ]
                )
            else:
                self.mbar.update_graph(
                    [
                        [fc.L.range(self.train_losses), self.train_losses],
                    ]
                )

    def after_epoch(self) -> None:
        if not self.training:
            if self.plot and hasattr(self.learner, "metrics"):
                self.valid_losses.append(
                    self.learner.metrics.all_metrics["loss"].compute().item()
                )
                self.mbar.update_graph(
                    [
                        [fc.L.range(self.train_losses), self.train_losses],
                        [
                            fc.L.range(self.epoch + 1).map(
                                lambda x: (x + 1) * len(self.dls.train)
                            ),
                            self.valid_losses,
                        ],
                    ]
                )

    def before_train(self) -> None:
        self.set_pb()

    def before_validate(self) -> None:
        self.set_pb()

    def set_pb(self) -> None:
        self.pb = progress_bar(self.dl, leave=False, parent=self.mbar)
        self.mbar.update(self.epoch)


class Recorder(Callback):
    """
    Keep track of losses and learning rates as training progress so we can
    plot them later.
    """

    order = 50

    def __init__(self, *params: tuple[str]) -> None:
        self.params = listify(params)

    def before_fit(self) -> None:
        self.params_records = {
            params: [[] for _ in self.opt.param_groups]
            for params in self.params
        }
        self.losses = []

    def after_batch(self) -> None:
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
    ) -> None:
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

    def plot_loss(self, skip_last: int = 0) -> None:
        """
        Plot losses, optionally skip last `skip_last` losses.
        """
        n = len(self.losses) - skip_last
        plt.plot(self.losses[:n])

    def plot(self, pgid: int = -1, skip_last: int = 0) -> None:
        """
        Plot loss vs lr (log-scale) across all iterations of training.
        """
        n = len(self.losses) - skip_last
        plt.xscale("log")
        plt.plot(
            self.params_records["lr"][pgid][:n],
            self.losses[:n],
        )


class ModelResetter(Callback):
    """
    Reset model's parameters. This is very useful in the context of NLP since
    we always reset hidden state. The assumption here is that `model` has a
    `reset` method that knows which parameters to reset and how.
    """

    def before_train(self) -> None:
        self.model.reset()

    def before_validate(self) -> None:
        self.model.reset()

    def after_fit(self) -> None:
        self.model.reset()


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
    ) -> None:
        self.gamma = gamma
        self.num_iter = num_iter
        self.stop_div = stop_div
        self.max_mult = max_mult

    def before_fit(self) -> None:
        self.scheduler = ExponentialLR(self.opt, self.gamma)
        path = self.path / self.model_dir
        path.mkdir(parents=True, exist_ok=True)
        self.tmp_d = tempfile.TemporaryDirectory(dir=path)
        self.tmp_p = Path(self.tmp_d.name).stem
        self.save_model(path / self.tmp_p / "_tmp.pth", with_opt=True)
        self.best_loss = float("inf")

    def after_batch(self) -> None:
        if self.loss < self.best_loss:
            self.best_loss = self.loss
        if self.loss > self.max_mult * self.best_loss and self.stop_div:
            raise CancelFitException()
        if self.n_iters >= self.num_iter:
            raise CancelFitException()
        self.scheduler.step()

    def before_validate(self) -> None:
        raise CancelValidException()

    def after_fit(self) -> None:
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

    order = 2

    def __init__(
        self, tfm: Callback, on_train: bool = True, on_valid: bool = True
    ) -> None:
        self.tfm = tfm
        self.on_train = on_train
        self.on_valid = on_valid

    def before_batch(self) -> None:
        if (self.on_train and self.training) or (
            self.on_valid and not self.training
        ):
            self.learner.batch = self.tfm(self.batch)


class SingleBatchCB(Callback):
    """
    Run 1 training/validation batch and stop by raising `CancelFitException`.
    Useful for debugging or want to check few parameters after 1 batch.
    """

    order = 1

    def after_batch(self) -> None:
        raise CancelFitException()


class MetricsCallback(Callback):
    """
    Compute/update given metrics and log it using `learner` defined logger
    after every `train`/`validate` epoch. Metrics have to implement `reset`
    and `compute` methods. Highly recommended to use metrics from
    `torcheval` package or inherit from its Metrics baseclass for custom
    metrics.
    """

    def __init__(self, *metrics, **named_metrics) -> None:
        self.metrics = named_metrics
        for metric in metrics:
            self.metrics[type(metric).__name__] = metric
        self.all_metrics = {"loss": Mean()}
        self.all_metrics.update(copy(self.metrics))

    def _compute(self):
        for metric in self.all_metrics.values():
            self.stats.append(f"{metric.compute():.3f}")
        self.stats.append("train" if self.training else "eval")
        self.stats.append(format_time(time.time() - self.start_time))

    def _reset(self):
        [metric.reset() for metric in self.all_metrics.values()]
        self.stats = [str(self.epoch + 1)]
        self.start_time = time.time()

    def before_fit(self) -> str:
        names = (
            ["epoch"]
            + ["loss"]
            + [name for name in self.metrics]
            + ["train"]
            + ["time"]
        )
        self.logger(names)

    def before_train(self) -> None:
        self._reset()

    def before_validate(self) -> None:
        self._reset()

    def after_train(self) -> str:
        self._compute()
        self.logger(self.stats)

    def after_validate(self) -> str:
        self._compute()
        self.logger(self.stats)

    def after_batch(self) -> None:
        for metric in self.metrics.values():
            metric.update(to_cpu(self.preds), to_cpu(*self.yb))
        self.all_metrics["loss"].update(
            to_cpu(self.learner.loss), weight=len(self.learner.xb[0])
        )


class Mixup(Callback):
    order = 90

    def __init__(self, alpha: float = 0.4) -> None:
        """
        Train the model with a mix of samples from each batch in the training
        data. Instead of feeding the model with raw data, we use linear
        combination of the input using `alpha` from beta distribution. This
        means that the labels would also be the linear combination of the
        labels and not the original labels. The implementation is largely
        based on this [paper](https://arxiv.org/abs/1710.09412).

        Parameters
        ----------
        alpha : float, default=0.4
            Concetration for Beta distribution.
        """
        self.distrib = torch.distributions.beta.Beta(
            torch.tensor([alpha]), torch.tensor([alpha])
        )

    def before_fit(self) -> None:
        self.learner.loss_func, self.old_loss_func = (
            self.loss_func,
            self.learner.loss_func,
        )

    def after_fit(self) -> None:
        self.learner.loss_func = self.old_loss_func

    def before_batch(self) -> None:
        λ = self.distrib.sample((len(self.learner.xb),)).to(
            self.learner.xb[0].device
        )
        λ = torch.stack([λ, 1 - λ], dim=1)
        self.λ = λ.max(1)[0].view(-1, 1, 1, 1)
        shuffle = torch.randperm(len(self.xb[0]))
        self.learner.xb = [
            x * self.λ + x[shuffle] * (1 - self.λ) for x in self.xb
        ]
        self.yb1 = [y[shuffle] for y in self.yb]

    def loss_func(self, pred, yb) -> torch.Tensor:
        if not self.training:
            return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, *self.yb1)
        loss = loss1 * self.λ + loss2 * (1 - self.λ)
        return reduce_loss(
            loss, getattr(self.old_loss_func, "reduction", "mean")
        )

