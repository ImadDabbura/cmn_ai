"""
Training-related callbacks for customizing training and validation loops.

This module provides a comprehensive collection of callbacks that enhance
and customize the training process. These callbacks handle various aspects
of training including device management, progress tracking, metrics computation,
data augmentation, and learning rate optimization.

The callbacks are designed to be easily composable and can be used together
to create sophisticated training pipelines. Each callback has a specific
execution order to ensure proper sequencing of operations.

Classes
-------
DeviceCallback
    Move model and data to specified device (CPU/GPU).
TrainEvalCallback
    Track training progress and manage training/evaluation modes.
ProgressCallback
    Display progress bars and live loss plots during training.
Recorder
    Record training metrics and optimizer parameters for analysis.
ModelResetter
    Reset model parameters at various training stages.
LRFinder
    Find optimal learning rate using exponential scheduling.
BatchTransform
    Apply transformations to entire batches.
BatchTransformX
    Apply transformations to input features only.
SingleBatchCallback
    Run only one batch for debugging purposes.
SingleBatchForwardCallback
    Run one batch and stop after forward pass.
MetricsCallback
    Compute and log various metrics during training.
Mixup
    Implement mixup data augmentation technique.

Examples
--------
Basic training with device management and progress tracking:

>>> from cmn_ai.callbacks.training import DeviceCallback, ProgressCallback
>>> from cmn_ai.callbacks.training import TrainEvalCallback, Recorder
>>>
>>> # Create callbacks
>>> callbacks = [
...     DeviceCallback(device='cuda'),  # Move to GPU
...     TrainEvalCallback(),            # Track progress
...     ProgressCallback(plot=True),    # Show progress bars
...     Recorder('lr', 'momentum')      # Record learning rate and momentum
... ]
>>>
>>> # Add to learner
>>> learner.add_cbs(callbacks)

Learning rate finding:

>>> from cmn_ai.callbacks.training import LRFinder
>>>
>>> # Find optimal learning rate
>>> lr_finder = LRFinder(gamma=1.3, num_iter=100, stop_div=True)
>>> learner.add_cb(lr_finder)
>>> learner.fit(1)  # Run for 1 epoch
>>> lr_finder.recorder.plot()  # Plot lr vs loss

Data augmentation with mixup:

>>> from cmn_ai.callbacks.training import Mixup
>>>
>>> # Add mixup augmentation
>>> mixup = Mixup(alpha=0.4)
>>> learner.add_cb(mixup)
>>> learner.fit(10)

Metrics tracking:

>>> from torcheval.metrics import MulticlassAccuracy
>>> from cmn_ai.callbacks.training import MetricsCallback
>>>
>>> # Track accuracy during training
>>> accuracy = MulticlassAccuracy()
>>> metrics_cb = MetricsCallback(accuracy=accuracy)
>>> learner.add_cb(metrics_cb)
>>> learner.fit(5)

Debugging with single batch:

>>> from cmn_ai.callbacks.training import SingleBatchCallback
>>>
>>> # Run only one batch for debugging
>>> debug_cb = SingleBatchCallback()
>>> learner.add_cb(debug_cb)
>>> learner.fit(1)  # Will stop after first batch

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

Notes
-----
- Callbacks are executed in order based on their `order` attribute
- Lower order numbers execute earlier
- Some callbacks modify the training loop behavior significantly
- Always test callbacks individually before combining them
- The `Recorder` callback is essential for post-training analysis
- `LRFinder` should be used before full training to find optimal learning rate
"""

import tempfile
import time
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import fastcore.all as fc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics import Mean

from ..losses import NoneReduce, reduce_loss
from ..plot import get_grid
from ..utils.data import DEFAULT_DEVICE, to_cpu, to_device
from ..utils.utils import listify
from .core import Callback, CancelFitException, CancelValidateException


class DeviceCallback(Callback):
    """
    Move batch and model to specified device.

    This callback ensures that both the model and input data are moved to
    the specified device (CPU/GPU) before training begins.

    Attributes
    ----------
    device : str | torch.device
        Device to copy batch and model to.
    """

    def __init__(self, device: str | torch.device = DEFAULT_DEVICE) -> None:
        """
        Initialize DeviceCallback.

        Parameters
        ----------
        device : str | torch.device, default=DEFAULT_DEVICE
            Device to copy batch and model to.
        """
        self.device = device

    def before_fit(self) -> None:
        """
        Move model to specified device before training starts.
        """
        self.model.to(self.device)

    def before_batch(self) -> None:
        """
        Move batch data to specified device before each batch.
        """
        self.learner.batch = to_device(self.batch, self.device)
        self.learner.xb = self.batch[: self.n_inp]
        self.learner.yb = self.batch[self.n_inp :]


class TrainEvalCallback(Callback):
    """
    Track training progress and manage training/evaluation modes.

    This callback tracks the number of iterations, percentage of training
    completed, and sets the appropriate training or evaluation mode for
    the model.

    Attributes
    ----------
    order : int
        Callback execution order (-10).
    """

    order = -10

    def before_fit(self) -> None:
        """
        Initialize training counters before training starts.
        """
        self.learner.n_iters = 0
        self.learner.pct_train = 0

    def after_batch(self) -> None:
        """
        Update iteration counter and training percentage after each batch.
        """
        if self.training:
            self.learner.n_iters += 1
            self.learner.pct_train += 1 / (self.iters * self.n_epochs)

    def before_train(self) -> None:
        """
        Set model to training mode and update training percentage.
        """
        self.model.train()
        self.learner.training = True
        self.learner.pct_train = self.epoch / self.n_epochs

    def before_validate(self) -> None:
        """
        Set model to evaluation mode before validation.
        """
        self.model.eval()
        self.learner.training = False


class ProgressCallback(Callback):
    """
    Track training progress with progress bars and live loss plotting.

    This callback provides visual feedback during training by displaying
    progress bars and optionally plotting training and validation losses
    in real-time.

    Attributes
    ----------
    order : int
        Callback execution order (-20).
    plot : bool
        Whether to plot train/valid losses during training.
    train_losses : List[float]
        List of training losses for plotting.
    valid_losses : List[float]
        List of validation losses for plotting.
    mbar : master_bar
        Master progress bar for epochs.
    pb : progress_bar
        Progress bar for batches.
    """

    order = -20

    def __init__(self, plot: bool = True) -> None:
        """
        Initialize ProgressCallback.

        Parameters
        ----------
        plot : bool, default=True
            Whether to plot train/valid losses during training.
        """
        super().__init__()
        self.plot = plot

    def before_fit(self) -> None:
        """
        Initialize progress tracking and create progress bars.
        """
        self.train_losses = []
        self.valid_losses = []
        self.learner.epochs = self.mbar = master_bar(range(self.n_epochs))
        self.mbar.on_iter_begin()
        # Overwrite default learner logger
        self.learner.logger = partial(self.mbar.write, table=True)

    def after_fit(self) -> None:
        """
        Clean up progress bar after training ends.
        """
        self.mbar.on_iter_end()

    def after_batch(self) -> None:
        """
        Update progress bar and optionally plot losses after each batch.
        """
        self.pb.update(self.iter)
        self.pb.comment = f"{self.loss:.3f}"  # noqa: E231
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
        """
        Update validation loss plot after each epoch.
        """
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
        """
        Set up progress bar before training phase.
        """
        self.set_pb()

    def before_validate(self) -> None:
        """
        Set up progress bar before validation phase.
        """
        self.set_pb()

    def set_pb(self) -> None:
        """
        Create and configure progress bar for current phase.
        """
        self.pb = progress_bar(self.dl, leave=False, parent=self.mbar)
        self.mbar.update(self.epoch)


class Recorder(Callback):
    """
    Record training metrics and optimizer parameters for later analysis.

    This callback keeps track of losses and optimizer parameters (like
    learning rates) throughout training, enabling post-training analysis
    and visualization.

    Attributes
    ----------
    order : int
        Callback execution order (50).
    params : List[str]
        List of parameter names to track.
    params_records : Dict[str, List[List[float]]]
        Recorded parameter values for each parameter group.
    losses : List[float]
        Recorded training losses.
    """

    order = 50

    def __init__(self, *params: tuple[str, ...]) -> None:
        """
        Initialize Recorder.

        Parameters
        ----------
        *params : tuple[str, ...]
            Parameter names to track (e.g., 'lr', 'momentum').
        """
        super().__init__()
        self.params = listify(params)
        sns.set()

    def before_fit(self) -> None:
        """
        Initialize recording structures before training starts.
        """
        self.params_records = {
            params: [[] for _ in self.opt.param_groups]
            for params in self.params
        }
        self.losses = []

    def after_batch(self) -> None:
        """
        Record parameters and loss after each training batch.
        """
        if self.training:
            for param, param_records in self.params_records.items():
                for pg, param_record in zip(
                    self.opt.param_groups, param_records
                ):
                    param_record.append(pg[param])
            self.losses.append(to_cpu(self.loss.item()))

    def plot_params(
        self,
        params: str | Iterable[str] = "lr",
        pgid: int = -1,
        figsize: tuple[int, int] = (8, 6),
    ) -> None:
        """
        Plot parameter values across training iterations.

        Parameters
        ----------
        params : str | Iterable[str], default="lr"
            Parameter name(s) to plot.
        pgid : int, default=-1
            Parameter group index to plot.
        figsize : tuple[int, int], default=(8, 6)
            Figure size for the plot.
        """
        params = listify(params)
        _, axes = get_grid(len(params), figsize=figsize)
        for (
            ax,
            param,
        ) in zip(axes.flatten(), params):
            ax.plot(self.params_records[param][pgid], label=param)
            ax.legend()
        plt.xlabel("Iteration")

    def plot_loss(self, skip_last: int = 0) -> None:
        """
        Plot training losses.

        Parameters
        ----------
        skip_last : int, default=0
            Number of last losses to skip in plotting.
        """
        n = len(self.losses) - skip_last
        plt.plot(self.losses[:n])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

    def plot(self, pgid: int = -1, skip_last: int = 0) -> None:
        """
        Plot loss vs learning rate (log-scale).

        Parameters
        ----------
        pgid : int, default=-1
            Parameter group index to plot.
        skip_last : int, default=0
            Number of last losses to skip in plotting.
        """
        n = len(self.losses) - skip_last
        plt.xscale("log")
        plt.plot(
            self.params_records["lr"][pgid][:n],
            self.losses[:n],
        )
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")


class ModelResetter(Callback):
    """
    Reset model parameters at various training stages.

    This callback is particularly useful for NLP models that need to reset
    hidden states. It assumes the model has a `reset` method that knows
    which parameters to reset and how.
    """

    def before_train(self) -> None:
        """
        Reset model before training phase.
        """
        self.model.reset()

    def before_validate(self) -> None:
        """
        Reset model before validation phase.
        """
        self.model.reset()

    def after_fit(self) -> None:
        """
        Reset model after training ends.
        """
        self.model.reset()


class LRFinder(Callback):
    """
    Find optimal learning rate using exponential schedule.

    This callback implements the learning rate finder technique from
    "Cyclical Learning Rates for Training Neural Networks". It tries
    different learning rates using an exponential schedule to help
    determine the best learning rate for training.

    Attributes
    ----------
    gamma : int
        Multiplicative factor for learning rate increase.
    num_iter : int
        Number of iterations to run the training.
    stop_div : bool
        Whether to stop training if loss diverges.
    max_mult : int
        Divergence threshold multiplier.
    scheduler : ExponentialLR
        Learning rate scheduler.
    best_loss : float
        Best loss encountered during training.
    """

    def __init__(
        self,
        gamma: int = 1.3,
        num_iter: int = 100,
        stop_div: bool = True,
        max_mult: int = 4,
    ) -> None:
        """
        Initialize LRFinder.

        Parameters
        ----------
        gamma : int, default=1.3
            Multiplicative factor for learning rate increase.
        num_iter : int, default=100
            Number of iterations to run the training.
        stop_div : bool, default=True
            Whether to stop training if loss diverges (loss > 4 * best_loss).
        max_mult : int, default=4
            Divergence threshold. If loss >= max_mult * minimum loss, stop
            training.
        """
        super().__init__()
        self.gamma = gamma
        self.num_iter = num_iter
        self.stop_div = stop_div
        self.max_mult = max_mult

    def before_fit(self) -> None:
        """
        Set up learning rate scheduler and save initial model state.
        """
        self.scheduler = ExponentialLR(self.opt, self.gamma)
        self.model_dir_path.mkdir(parents=True, exist_ok=True)
        self.tmp_d = tempfile.TemporaryDirectory(dir=self.model_dir_path)
        self.tmp_p = Path(self.tmp_d.name).stem
        self.save_model(
            self.model_dir_path / self.tmp_p / "_tmp.pth", with_opt=True
        )
        self.best_loss = float("inf")

    def after_batch(self) -> None:
        """
        Update best loss and check for divergence or completion.
        """
        if self.loss < self.best_loss:
            self.best_loss = self.loss
        if self.loss > self.max_mult * self.best_loss and self.stop_div:
            raise CancelFitException()
        if self.n_iters >= self.num_iter:
            raise CancelFitException()
        self.scheduler.step()

    def before_validate(self) -> None:
        """
        Skip validation during learning rate finding.
        """
        raise CancelValidateException()

    def after_fit(self) -> None:
        """
        Restore model state and plot learning rate vs loss.
        """
        self.opt.zero_grad()
        tmp_f = self.model_dir_path / self.tmp_p / "_tmp.pth"
        if tmp_f.exists():
            self.load_model(tmp_f, with_opt=True)
            self.tmp_d.cleanup()
        self.recorder.plot()


class BatchTransform(Callback):
    """
    Apply transformations to entire batches before processing.

    This callback applies a transformation function to the entire batch
    before it's processed by the model. The transformation runs on the
    device where the data is located.

    Attributes
    ----------
    order : int
        Callback execution order (DeviceCallback.order + 1).
    tfm : Callback
        Transformation function to apply.
    on_train : bool
        Whether to apply transformation during training.
    on_valid : bool
        Whether to apply transformation during validation.
    """

    # So transforms run on the device
    order = DeviceCallback.order + 1

    def __init__(
        self, tfm: Callback, on_train: bool = True, on_valid: bool = True
    ) -> None:
        """
        Initialize BatchTransform.

        Parameters
        ----------
        tfm : Callback
            Transformation function to apply on the batch.
        on_train : bool, default=True
            Whether to apply the transformation during training.
        on_valid : bool, default=True
            Whether to apply the transformation during validation.
        """
        super().__init__()
        self.tfm = tfm
        self.on_train = on_train
        self.on_valid = on_valid

    def before_batch(self) -> None:
        """
        Apply transformation to batch if conditions are met.
        """
        if (self.on_train and self.training) or (
            self.on_valid and not self.training
        ):
            self.learner.batch = self.tfm(self.batch)
            self.learner.xb = self.batch[: self.n_inp]
            self.learner.yb = self.batch[self.n_inp :]


class BatchTransformX(Callback):
    """
    Apply transformations to input features (X) only.

    This callback applies a transformation function specifically to the
    input features (X) of the batch, leaving the targets (Y) unchanged.

    Attributes
    ----------
    order : int
        Callback execution order (DeviceCallback.order + 1).
    tfm : Callback
        Transformation function to apply.
    on_train : bool
        Whether to apply transformation during training.
    on_valid : bool
        Whether to apply transformation during validation.
    """

    order = DeviceCallback.order + 1

    def __init__(
        self, tfm: Callback, on_train: bool = True, on_valid: bool = True
    ) -> None:
        """
        Initialize BatchTransformX.

        Parameters
        ----------
        tfm : Callback
            Transformation function to apply on the input features.
        on_train : bool, default=True
            Whether to apply the transformation during training.
        on_valid : bool, default=True
            Whether to apply the transformation during validation.
        """
        super().__init__()
        self.tfm = tfm
        self.on_train = on_train
        self.on_valid = on_valid

    def before_batch(self) -> None:
        """
        Apply transformation to input features if conditions are met.
        """
        if (self.on_train and self.training) or (
            self.on_valid and not self.training
        ):
            self.learner.xb = self.tfm(self.xb)


class SingleBatchCallback(Callback):
    """
    Run only one training/validation batch and stop.

    This callback is useful for debugging or when you want to check
    parameters after processing just one batch. It raises CancelFitException
    after the first batch to stop training.

    Attributes
    ----------
    order : int
        Callback execution order (1).
    """

    order = 1

    def after_batch(self) -> None:
        """
        Stop training after the first batch.
        """
        raise CancelFitException()


class SingleBatchForwardCallback(Callback):
    """
    Run one batch and stop after forward pass.

    This callback runs one training/validation batch and stops after
    computing the loss (after forward pass) by raising CancelFitException.
    Useful for debugging or checking parameters after one forward pass.

    Attributes
    ----------
    order : int
        Callback execution order (1).
    """

    order = 1

    def after_loss(self) -> None:
        """
        Stop training after computing loss for the first batch.
        """
        raise CancelFitException()


class MetricsCallback(Callback):
    """
    Compute and log metrics during training and validation.

    This callback computes various metrics after each training/validation
    epoch and logs them using the learner's logger. Metrics must implement
    `reset` and `compute` methods. It's recommended to use metrics from
    the `torcheval` package or inherit from its Metrics base class.

    Attributes
    ----------
    metrics : dict[str, Any]
        Dictionary of named metrics to compute.
    all_metrics : dict[str, Any]
        All metrics including loss metric.
    stats : list[str]
        Current statistics to log.
    start_time : float
        Start time for timing calculations.
    """

    def __init__(self, *metrics: Any, **named_metrics: Any) -> None:
        """
        Initialize MetricsCallback.

        Parameters
        ----------
        *metrics : Any
            Positional metrics to add.
        **named_metrics : Any
            Named metrics to add.
        """
        super().__init__()
        self.metrics = named_metrics
        for metric in metrics:
            self.metrics[type(metric).__name__] = metric
        self.all_metrics = {"loss": Mean()}
        self.all_metrics.update(copy(self.metrics))

    def _compute(self) -> None:
        """
        Compute all metrics and prepare statistics for logging.
        """
        for metric in self.all_metrics.values():
            self.stats.append(f"{metric.compute():.3f}")  # noqa: E231
        self.stats.append("train" if self.training else "eval")
        self.stats.append(format_time(time.time() - self.start_time))

    def _reset(self) -> None:
        """
        Reset all metrics and initialize statistics.
        """
        [metric.reset() for metric in self.all_metrics.values()]
        self.stats = [str(self.epoch + 1)]
        self.start_time = time.time()

    def before_fit(self) -> str:
        """
        Log metric names as header before training starts.

        Returns
        -------
        str
            Header string with metric names.
        """
        names = (
            ["epoch"]
            + ["loss"]
            + [name for name in self.metrics]
            + ["train"]
            + ["time"]
        )
        self.logger(names)

    def before_train(self) -> None:
        """
        Reset metrics before training phase.
        """
        self._reset()

    def before_validate(self) -> None:
        """
        Reset metrics before validation phase.
        """
        self._reset()

    def after_train(self) -> str:
        """
        Compute and log metrics after training epoch.

        Returns
        -------
        str
            Logged statistics string.
        """
        self._compute()
        self.logger(self.stats)

    def after_validate(self) -> str:
        """
        Compute and log metrics after validation epoch.

        Returns
        -------
        str
            Logged statistics string.
        """
        self._compute()
        self.logger(self.stats)

    def after_batch(self) -> None:
        """
        Update metrics with batch predictions and targets.
        """
        for metric in self.metrics.values():
            metric.update(to_cpu(self.preds), to_cpu(*self.yb))
        self.all_metrics["loss"].update(
            to_cpu(self.learner.loss), weight=len(self.learner.xb[0])
        )


class Mixup(Callback):
    """
    Implement mixup data augmentation technique.

    This callback implements the mixup technique where instead of feeding
    raw data to the model, it uses linear combinations of inputs using
    alpha from a beta distribution. The labels are also linear combinations
    of the original labels. Based on the paper "mixup: BEYOND EMPIRICAL
    RISK MINIMIZATION".

    Attributes
    ----------
    order : int
        Callback execution order (90).
    alpha : float
        Concentration parameter for Beta distribution.
    distrib : torch.distributions.beta.Beta
        Beta distribution for sampling mixup weights.
    old_loss_func : Callable
        Original loss function before mixup.
    λ : torch.Tensor
        Mixup weight for current batch.
    yb1 : List[torch.Tensor]
        Shuffled targets for mixup.
    """

    order = 90

    def __init__(self, alpha: float = 0.4) -> None:
        """
        Initialize Mixup.

        Parameters
        ----------
        alpha : float, default=0.4
            Concentration parameter for Beta distribution.
        """
        super().__init__()
        self.alpha = alpha
        self.distrib = torch.distributions.beta.Beta(
            torch.tensor([alpha]), torch.tensor([alpha])
        )

    def before_fit(self) -> None:
        """
        Store original loss function before training starts.
        """
        self.learner.loss_func, self.old_loss_func = (
            self.loss_func,
            self.learner.loss_func,
        )

    def after_fit(self) -> None:
        """
        Restore original loss function after training ends.
        """
        self.learner.loss_func = self.old_loss_func

    def before_batch(self) -> None:
        """
        Apply mixup transformation to batch inputs and targets.

        The mixup process involves:
        1. Drawing samples from a beta distribution for each image
        2. Taking the maximum of λ and 1-λ to avoid identical combinations
        3. Shuffling the batch for combination
        4. Creating linear combinations of inputs and targets
        """
        λ = self.distrib.sample((len(self.xb),)).to(self.xb[0].device)
        λ = torch.stack([λ, 1 - λ], dim=1)
        self.λ = λ.max(1)[0].view(-1, 1, 1, 1)
        shuffle = torch.randperm(len(self.xb[0]))
        self.learner.xb = [
            x * self.λ + x[shuffle] * (1 - self.λ) for x in self.xb
        ]
        self.yb1 = [y[shuffle] for y in self.yb]

    def loss_func(self, pred: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        """
        Compute mixup loss combining original and shuffled targets.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions.
        yb : torch.Tensor
            Original targets.

        Returns
        -------
        torch.Tensor
            Mixup loss value.
        """
        if not self.training:
            return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, *self.yb1)
        loss = loss1 * self.λ + loss2 * (1 - self.λ)
        return reduce_loss(
            loss, getattr(self.old_loss_func, "reduction", "mean")
        )
