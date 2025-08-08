"""
Learner module for training PyTorch models with callback system.

This module provides a `Learner` class that handles the training loop of PyTorch models
using a customizable callback system. The training loop consists of a minimal set of
instructions that can be extended and customized through callbacks.

The basic training loop iterates through data and:

- Computes the output of the model from the input
- Calculates a loss between this output and the desired target
- Computes the gradients of this loss with respect to model parameters
- Updates the parameters accordingly
- Zeros all the gradients

Any customization of this training loop is defined in a `Callback` object.
A callback can implement actions on the following events:

- `before_fit`: Called before doing anything, ideal for initial setup
- `before_epoch`: Called at the beginning of each epoch, useful for any
  behavior you need to reset at each epoch
- `before_train`: Called at the beginning of the training part of an epoch
- `before_batch`: Called at the beginning of each batch, just after drawing
  said batch. It can be used to do any setup necessary for the batch
  (like hyper-parameter scheduling) or to change the input/target before
  it goes in the model (change of the input with techniques like mixup)
- `after_pred`: Called after computing the output of the model on the batch.
  It can be used to change that output before it's fed to the loss function
- `after_loss`: Called after the loss has been computed, but before the
  backward pass. It can be used to add any penalty to the loss
  (AR or TAR in RNN training for instance)
- `after_cancel_backward`: Reached immediately after `CancelBackwardException`
- `after_backward`: Called after the backward pass, but before updating
  the parameters. It can be used to do any change to the gradients
  before any updates (gradient clipping for instance)
- `after_cancel_step`: Reached immediately after `CancelStepException`
- `after_step`: Called after the step and before gradients are zeroed
- `after_cancel_batch`: Reached immediately after `CancelBatchException`
  before proceeding to `after_batch`
- `after_batch`: Called at the end of a batch, for any clean-up before the next one
- `after_cancel_train`: Reached immediately after `CancelTrainException`
  before proceeding to `after_train`
- `after_train`: Called at the end of the training phase of an epoch
- `before_validate`: Called at the beginning of the validation phase of an epoch,
  useful for any setup needed specifically for validation
- `after_cancel_validate`: Reached immediately after `CancelValidateException`
  before proceeding to `after_validate`
- `after_validate`: Called at the end of the validation phase of an epoch
- `after_cancel_epoch`: Reached immediately after `CancelEpochException`
  before proceeding to `after_epoch`
- `after_epoch`: Called at the end of an epoch, for any clean-up before the next one
- `after_cancel_fit`: Reached immediately after `CancelFitException`
  before proceeding to `after_fit`
- `after_fit`: Called at the end of training, for any final clean-up

Classes
-------
Learner
    Main class for training PyTorch models with callback system.

Functions
---------
params_getter
    Get all parameters of a model recursively.

Examples
--------
Basic usage:

>>> from cmn_ai.learner import Learner
>>> from cmn_ai.utils.data import DataLoaders
>>> import torch.nn as nn
>>>
>>> # Create a simple model and data loaders
>>> model = nn.Linear(10, 1)
>>> dls = DataLoaders(train_dl, valid_dl)
>>>
>>> # Create learner
>>> learner = Learner(model, dls, loss_func=nn.MSELoss())
>>>
>>> # Train the model
>>> learner.fit(n_epochs=10)

Learning rate finding:

>>> learner.lr_find(start_lr=1e-6, num_iter=200)

Model checkpointing:

>>> learner.save_model("checkpoint.pt", with_opt=True, with_epoch=True)
>>> learner.load_model("checkpoint.pt", with_opt=True)

Notes
-----
The `TrainEvalCallback` is automatically added to all learners and doesn't need
to be provided manually. This callback handles the basic training and validation
loop management.

See Also
--------
cmn_ai.callbacks.core.Callback : Base callback class
cmn_ai.callbacks.training.TrainEvalCallback : Default training callback
cmn_ai.utils.data.DataLoaders : Data loader container
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch import tensor
from torch.utils.data import DataLoader
from torchinfo import summary

from .callbacks.core import (
    Callback,
    CancelBackwardException,
    CancelBatchException,
    CancelEpochException,
    CancelFitException,
    CancelStepException,
    CancelTrainException,
    CancelValidateException,
)
from .callbacks.training import (
    LRFinder,
    ProgressCallback,
    Recorder,
    TrainEvalCallback,
)
from .utils.data import DataLoaders
from .utils.utils import listify


def params_getter(model: nn.Module) -> Iterable[nn.Parameter]:
    """
    Get all parameters of a model recursively.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to extract parameters from.

    Yields
    ------
    nn.Parameter
        Each parameter of the model.

    Examples
    --------
    >>> model = nn.Linear(10, 1)
    >>> params = list(params_getter(model))
    >>> len(params)
    2
    """
    return model.parameters()


class Learner:
    """
    A learner class that handles training loop of PyTorch models.

    This class provides a customizable training loop using a callback system.
    It handles model training, validation, saving/loading, and learning rate
    finding. The training process can be customized through various callback
    events.

    Attributes
    ----------
    model : nn.Module
        The PyTorch model to train.
    dls : DataLoaders
        Training and validation data loaders.
    n_inp : int
        Number of inputs to the model.
    loss_func : Callable[[tensor, tensor], tensor]
        Loss function that takes predictions and targets.
    opt_func : opt.Optimizer
        Optimizer class (not instance).
    lr : float
        Learning rate for training.
    splitter : Callable[[nn.Module], Iterable[nn.Parameter]]
        Function to split model's parameters into groups.
    path : Path
        Base path for saving artifacts.
    model_dir_path : Path
        Directory path for saving models.
    callbacks : list[Callback]
        List of all callbacks used by the learner.
    logger : Any
        Logger for metrics. Default is `print` but typically modified
        by callbacks such as `ProgressCallback`.

    Examples
    --------
    >>> from cmn_ai.learner import Learner
    >>> from cmn_ai.utils.data import DataLoaders
    >>> import torch.nn as nn
    >>>
    >>> model = nn.Linear(10, 1)
    >>> dls = DataLoaders(train_dl, valid_dl)
    >>> learner = Learner(model, dls, loss_func=nn.MSELoss())
    >>> learner.fit(n_epochs=5)
    """

    def __init__(
        self,
        model: nn.Module,
        dls: DataLoaders,
        n_inp: int = 1,
        loss_func: Callable[[tensor, tensor], tensor] = F.mse_loss,
        opt_func: opt.Optimizer = opt.SGD,
        lr: float = 1e-2,
        splitter: Callable[
            [nn.Module], Iterable[nn.Parameter]
        ] = params_getter,
        path: str | Path = ".",
        model_dir: str = "models",
        callbacks: Iterable[Callback] | None = None,
    ) -> None:
        """
        Initialize the Learner.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to train.
        dls : DataLoaders
            Training and validation data loaders.
        n_inp : int, default=1
            Number of inputs to the model.
        loss_func : Callable[[tensor, tensor], tensor], default=F.mse_loss
            Loss function that takes predictions and targets.
        opt_func : opt.Optimizer, default=opt.SGD
            Optimizer class (not instance).
        lr : float, default=1e-2
            Learning rate for training.
        splitter : Callable[[nn.Module], Iterable[nn.Parameter]], default=params_getter
            Function to split model's parameters into groups.
        path : str | Path, default="."
            Base path for saving artifacts.
        model_dir : str, default="models"
            Model directory name relative to `path`.
        callbacks : Iterable[Callback] | None, default=None
            Initial callbacks to add to the learner.

        Notes
        -----
        The `TrainEvalCallback` is automatically added to the callbacks list
        and doesn't need to be provided manually.
        """
        self.model = model
        self.dls = dls
        self.n_inp = n_inp
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.opt: opt.Optimizer | None = None
        self.lr = lr
        self.splitter = splitter
        self.path = Path(path)
        self.model_dir_path = self.path / Path(model_dir)
        self.logger: Any = print
        self.callbacks: list[Callback] = []
        callbacks = listify(callbacks) + [
            TrainEvalCallback(),
        ]
        self._add_callbacks(callbacks)

    def _with_events(
        self, func: Callable[[], None], event_nm: str, exc: type[Exception]
    ) -> None:
        """Execute a function with callback events before and after."""
        try:
            self._callback(f"before_{event_nm}")
            func()
        except exc:
            self._callback(f"after_cancel_{event_nm}")
        finally:
            self._callback(f"after_{event_nm}")

    def _backward(self) -> None:
        """Perform backward pass."""
        self.loss.backward()

    def _step(self) -> None:
        """Perform optimizer step."""
        self.opt.step()

    def _one_batch(self) -> None:
        """Process one batch through the model."""
        self.preds = self.model(*self.xb)
        self._callback("after_predict")
        self.loss = self.loss_func(self.preds, *self.yb)
        self._callback("after_loss")
        if self.training:
            self._with_events(
                self._backward, "backward", CancelBackwardException
            )
            self._with_events(self._step, "step", CancelStepException)
            self.opt.zero_grad()

    def _all_batches(self) -> None:
        """Process all batches in the current data loader."""
        self.iters = len(self.dl)
        for self.iter, self.batch in enumerate(self.dl):
            self.xb = self.batch[: self.n_inp]
            self.yb = self.batch[self.n_inp :]
            self._with_events(self._one_batch, "batch", CancelBatchException)

    def _one_epoch_train(self) -> None:
        """Train for one epoch."""
        self.dl = self.dls.train
        self._with_events(self._all_batches, "train", CancelTrainException)

    def _one_epoch_validate(self) -> None:
        """Validate for one epoch."""
        self.dl = self.dls.valid
        with torch.no_grad():
            self._with_events(
                self._all_batches, "validate", CancelValidateException
            )

    def _one_epoch(self) -> None:
        """Process one epoch (train and/or validate)."""
        if self.run_train:
            self._one_epoch_train()
        if self.run_valid:
            self._one_epoch_validate()

    def _fit(self) -> None:
        """Main training loop."""
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self._with_events(self._one_epoch, "epoch", CancelEpochException)

    def fit(
        self,
        n_epochs: int = 1,
        run_train: bool = True,
        run_valid: bool = True,
        callbacks: Iterable[Callback] | None = None,
        lr: float | None = None,
        reset_opt: bool = False,
    ) -> None:
        """
        Fit the model for a specified number of epochs.

        Parameters
        ----------
        n_epochs : int, default=1
            Number of epochs to train the model.
        run_train : bool, default=True
            Whether to run training passes.
        run_valid : bool, default=True
            Whether to run validation passes.
        callbacks : Iterable[Callback] | None, default=None
            Additional callbacks to add temporarily for this fit call.
            These callbacks will be removed after training completes.
        lr : float | None, default=None
            Learning rate to use. If None, uses the learner's default lr.
        reset_opt : bool, default=False
            Whether to reset the optimizer.

        Examples
        --------
        >>> learner.fit(n_epochs=10, lr=0.001)
        >>> learner.fit(n_epochs=5, run_valid=False)
        """
        self.run_train = run_train
        self.run_valid = run_valid
        callbacks = self._add_callbacks(callbacks)
        self.n_epochs = n_epochs
        if lr is None:
            lr = self.lr
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr)
        try:
            self._with_events(self._fit, "fit", CancelFitException)
        finally:
            self._remove_callbacks(callbacks)

    def lr_find(
        self,
        start_lr: float = 1e-7,
        gamma: float = 1.3,
        num_iter: int = 100,
        stop_div: bool = True,
        max_mult: int = 4,
    ) -> None:
        """
        Find optimal learning rate using exponential schedule.

        This method implements the learning rate finder described in
        [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf).
        It tries different learning rates using an exponential schedule and
        plots learning rate vs loss to help identify the optimal learning rate.

        Parameters
        ----------
        start_lr : float, default=1e-7
            Starting learning rate for the search.
        gamma : float, default=1.3
            Multiplicative factor for learning rate increase.
        num_iter : int, default=100
            Number of iterations to run the learning rate search.
        stop_div : bool, default=True
            Whether to stop training if the loss diverges.
        max_mult : int, default=4
            Divergence threshold. If loss >= max_mult * minimum loss,
            training stops.

        Examples
        --------
        >>> learner.lr_find(start_lr=1e-6, num_iter=200)
        """
        n_epochs = num_iter // len(self.dls.train) + 1
        callbacks = [
            LRFinder(gamma, num_iter, stop_div, max_mult),
            Recorder("lr"),
        ]
        self.fit(n_epochs, run_valid=False, callbacks=callbacks, lr=start_lr)

    def save_model(
        self,
        path: str | Path | None = None,
        with_opt: bool = False,
        with_epoch: bool = False,
        with_loss: bool = False,
        pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        """
        Save the model and optionally optimizer state, epoch, and loss.

        This method is useful for checkpointing during training. It saves
        the model state dict and optionally includes optimizer state,
        current epoch, and current loss.

        Parameters
        ----------
        path : str | Path | None, default=None
            File path to save the model. If None, uses
            `learner.model_dir_path`/model.
        with_opt : bool, default=False
            Whether to save the optimizer state.
        with_epoch : bool, default=False
            Whether to save the current epoch number.
        with_loss : bool, default=False
            Whether to save the current loss value.
        pickle_protocol : int, default=pickle.HIGHEST_PROTOCOL
            Protocol used by pickler when saving the checkpoint.

        Examples
        --------
        >>> learner.save_model("checkpoint.pt", with_opt=True, with_epoch=True)
        >>> learner.save_model()  # Uses default path
        """
        if path is None:
            path = self.model_dir_path / "model"
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
        }
        if with_opt:
            checkpoint["optimizer_state_dict"] = self.opt.state_dict()
        if with_epoch:
            checkpoint["epoch"] = self.epoch
        if with_loss:
            checkpoint["loss"] = self.loss
        torch.save(checkpoint, path, pickle_protocol=pickle_protocol)

    def load_model(
        self,
        path: str | Path | None = None,
        with_opt: bool = False,
        with_epoch: bool = False,
        with_loss: bool = False,
    ) -> None:
        """
        Load the model and optionally optimizer state, epoch, and loss.

        Parameters
        ----------
        path : str | Path | None, default=None
            Model's file path. If None, uses `learner.model_dir_path`/model.
        with_opt : bool, default=False
            Whether to load the optimizer state.
        with_epoch : bool, default=False
            Whether to load the current epoch number.
        with_loss : bool, default=False
            Whether to load the current loss value.

        Examples
        --------
        >>> learner.load_model("checkpoint.pt", with_opt=True)
        >>> learner.load_model()  # Uses default path
        """
        if path is None:
            path = self.model_dir_path / "model"
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if with_opt:
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if with_epoch:
            self.epoch = checkpoint["epoch"]
        if with_loss:
            self.loss = checkpoint["loss"]

    @property
    def training(self) -> bool:
        """
        Get the training mode of the model.

        Returns
        -------
        bool
            True if the model is in training mode, False otherwise.
        """
        return self.model.training

    @training.setter
    def training(self, v: bool) -> None:
        """
        Set the training mode of the model.

        Parameters
        ----------
        v : bool
            True to set training mode, False for evaluation mode.
        """
        self.model.training = v

    def _add_callbacks(self, cbs: Iterable[Callback] | None) -> list[Callback]:
        """Add callbacks to the learner."""
        added_callbacks: list[Callback] = []
        for cb in listify(cbs):
            if not hasattr(self, cb.name):
                cb.set_learner(self)
                setattr(self, cb.name, cb)
                self.callbacks.append(cb)
                added_callbacks.append(cb)
        return added_callbacks

    def _remove_callbacks(self, cbs: Iterable[Callback] | None) -> None:
        """Remove callbacks from the learner."""
        for cb in listify(cbs):
            delattr(self, cb.name)
            self.callbacks.remove(cb)

    def _callback(self, event_nm: str) -> None:
        """Execute all callbacks for a given event."""
        for cb in sorted(self.callbacks, key=lambda x: x.order):
            cb(event_nm)

    def summary(self, verbose: int = 2, **kwargs: Any) -> Any:
        """
        Generate and display model summary using torchinfo.

        Parameters
        ----------
        verbose : int, default=2
            Verbosity level for the summary output.
        **kwargs : Any
            Additional arguments passed to torchinfo.summary.

        Returns
        -------
        Any
            The summary object returned by torchinfo.

        Examples
        --------
        >>> learner.summary(verbose=1)
        >>> learner.summary(col_names=["input_size", "output_size"])
        """
        return summary(
            self.model,
            input_data=next(iter(self.dls.train))[0],
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
                "params_percent",
            ],
            verbose=verbose,
            **kwargs,
        )

    def show_batch(
        self,
        sample_sz: int = 1,
        callbacks: Iterable[Callback] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Show a sample batch of input data.

        This method displays what the model would see when making predictions,
        including all transformations and augmentations applied to the input.

        Parameters
        ----------
        sample_sz : int, default=1
            Number of input samples to show.
        callbacks : Iterable[Callback] | None, default=None
            Additional callbacks to add temporarily for this operation.
            These callbacks will be removed after the operation completes.
        **kwargs : Any
            Additional arguments passed to the show_batch implementation.

        Raises
        ------
        NotImplementedError
            Different types of `Learner`'s must implement their own
            version depending on the type of input data. For example,
            `VisionLearner`'s would show images.

        Examples
        --------
        >>> learner.show_batch(sample_sz=3)
        """
        raise NotImplementedError(
            "show_batch must be implemented by specific learner types"
        )
