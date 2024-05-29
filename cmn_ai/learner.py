"""
`Learner` is a basic class that provides useful functionalities:

- Tweaking/customization of the training loop using a system of
callbacks through Exceptions
- Loading/saving model
- Fit the model
- Get model summary
- Learning rate finder

The training loop consists of a minimal set of instructions; looping
through the data we:

- Compute the output of the model from the input
- Calculate a loss between this output and the desired target
- Compute the gradients of this loss with respect to model parameters
- Update the parameters accordingly
- Zero all the gradients

Any tweak of this training loop is defined in a [Callback]
[cmn_ai.callbacks.core.Callback]. A callback can implement actions on
the following events:

- `before_fit`: called before doing anything, ideal for initial setup
- `before_epoch`: called at the beginning of each epoch, useful for any
behavior you need to reset at each epoch
- `before_train`: called at the beginning of the training part of an
epoch
- `before_batch`: called at the beginning of each batch, just after
drawing said batch. It can be used to do any setup necessary for the
batch (like hyper-parameter scheduling) or to change the input/target
before it goes in the model (change of the input with techniques like
mixup for instance)
- `after_pred`: called after computing the output of the model on the
batch. It can be used to change that output before it's fed to the loss
function
- `after_loss`: called after the loss has been computed, but before the
backward pass. It can be used to add any penalty to the loss (AR or TAR
in RNN training for instance)
- `after_cancel_backward`: reached immediately after
  [CancelBackwardException][cmn_ai.callbacks.core.CancelBackwardException]
- `after_backward`: called after the backward pass, but before updating
the parameters. It can be used to do any change to the
gradients before any updates (gradient clipping for instance)
- `after_cancel_step`: reached immediately after
  [CancelStepException][cmn_ai.callbacks.core.CancelStepException]
- `after_step`: called after the step and before gradients are zeroed
- `after_cancel_batch`: reached immediately after
  [CancelBatchException][cmn_ai.callbacks.core.CancelBatchException]
before proceeding to `after_batch`
- `after_batch`: called at the end of a batch, for any clean-up before
the next one
- `after_cancel_train`: reached immediately after
  [CancelTrainException][cmn_ai.callbacks.core.CancelTrainException]
before proceeding to `after_train`
- `after_train`: called at the end of the training phase of an epoch
- `before_validate`: called at the beginning of the validation phase of
an epoch, useful for any setup needed specifically for validation
- `after_cancel_validate`: reached immediately after
[CancelValidateException][cmn_ai.callbacks.core.CancelValidateException]
before proceeding to `after_validate`
- `after_validate`: called at the end of the validation phase of an
epoch
- `after_cancel_epoch`: reached immediately after
  [CancelEpochException][cmn_ai.callbacks.core.CancelEpochException]
before proceeding to `after_epoch`
- `after_epoch`: called at the end of an epoch, for any clean-up before
the next one
- `after_cancel_fit`: reached immediately after
  [CancelFitException][cmn_ai.callbacks.core.CancelFitException]
before proceeding to `after_fit`
- `after_fit`: called at the end of training, for any final clean-up
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable
from pathlib import Path

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
    Get all parameters of `model` recursively.

    Parameters
    ----------
    model : nn.Module
        Model.

    Yields
    -------
    Parameter
        Module parameter.
    """
    return model.parameters()


class Learner:
    """
    Learner is a basic class that handles training loop of pytorch model
    and utilize a systems of callbacks that makes training loop very
    customizable and extensible. You just need to provide a list of
    callbacks and callback functions.

    Attributes
    ----------
    model : nn.Module
        Pytorch's model.
    dls : DataLoaders
        Train and valid data loaders.
    n_inp : int
        Number of inputs to the model.
    loss_func : Callable[[tensor, tensor], tensor]
        Loss function.
    opt_func : opt.Optimizer
        Optimizer function/class.
    lr : float
        Learning rate.
    splitter : Callable[[nn.Module], Iterable[nn.Parameter]]
        Function to split model's parameters into groups.
    path : Path
        Path to save all artifacts.
    model_dir_path : Path
        Model directory path.
    callbacks : Iterable[Callable] | None, default=None
        Iterable of callbacks of type `Callback`.
    logger : Any
        Logger to log metrics. Default is `print` but is typically
        modified by callbacks such as `ProgressCallback`.
    callbacks: list[Callback]
        List of all the used callbacks by `learner.`
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
        default_callbacks: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            Pytorch's model.
        dls : DataLoaders
            Train and valid data loaders.
        n_inp : int, default=1
            Number of inputs to the model.
        loss_func : Callable[[tensor, tensor], tensor],\
                default=F.mse_loss
            Loss function.
        opt_func : opt.Optimizer, default=opt.SGD
            Optimizer function/class.
        lr : float, default=`1e-2`
            Learning rate.
        splitter : Callable[[nn.Module], Iterable[nn.Parameter]],\
                default=`params_getter`
            Function to split model's parameters into groups, default
            all parameters belong to 1 group.
        path : str, default="."
            Path to save all artifacts.
        model_dir : str, default="models"
            Model directory name relative to `path`.
        callbacks : Iterable[Callable] | None, default=None
            Iterable of callbacks of type `Callback`.
        default_callbacks : bool, default=True
            Whether to add `TrainEvalCallback`, `ProgressCallback`, and
            `Recorder` to the list of callbacks.
        """
        self.model = model
        self.dls = dls
        self.n_inp = n_inp
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.opt = None
        self.lr = lr
        self.splitter = splitter
        self.path = Path(path)
        self.model_dir_path = self.path / Path(model_dir)
        self.logger = print
        self.callbacks = []
        if default_callbacks:
            callbacks = listify(callbacks) + [
                TrainEvalCallback(),
                ProgressCallback(),
                Recorder("lr"),
            ]
        self._add_callbacks(callbacks)

    def _with_events(self, func, event_nm, exc):
        try:
            self._callback(f"before_{event_nm}")
            func()
        except exc:
            self._callback(f"after_cancel_{event_nm}")
        finally:
            self._callback(f"after_{event_nm}")

    def _backward(self):
        self.loss.backward()

    def _step(self):
        self.opt.step()

    def _one_batch(self):
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

    def _all_batches(self):
        self.iters = len(self.dl)
        for self.iter, self.batch in enumerate(self.dl):
            self.xb = self.batch[: self.n_inp]
            self.yb = self.batch[self.n_inp :]
            self._with_events(self._one_batch, "batch", CancelBatchException)

    def _one_epoch_train(self):
        self.model.train()
        self.dl = self.dls.train
        self._with_events(self._all_batches, "train", CancelTrainException)

    def _one_epoch_validate(self):
        self.model.eval()
        self.dl = self.dls.valid
        with torch.no_grad():
            self._with_events(
                self._all_batches, "validate", CancelValidateException
            )

    def _one_epoch(self):
        if self.run_train:
            self._one_epoch_train()
        if self.run_valid:
            self._one_epoch_validate()

    def _fit(self):
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
        Fit the model for `n_epochs`.

        Parameters
        ----------
        n_epochs : int, default=1
            Number epochs to train the model.
        run_train : bool, default=True
            Whether to run training passes.
        run_valid : bool, default=True
            Whether to run validation passes.
        callbacks : Iterable[Callback] | None, default=None
            Callbacks to add to the existing callbacks. The added
            callbacks will be removed  before `fit` returns.
        lr : float | None, default=None
            Learning rate. If None, `lr` passed to `Learner` will be
            used.
        reset_opt : bool, default=False
            Whether to reset the optimizer.
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
    ):
        """
        Try different learning rates using exponential schedule to help pick
        the best learning rate following [Cyclical Learning Rates for Training
        Neural Networks](https://arxiv.org/pdf/1506.01186.pdf). When done, plot
        learning rate vs loss.

        Parameters
        ----------
        start_lr : float, default=1e-7
            Start learning rate.
        gamma : float, default=1.3
            Multiplicative factor of learning rate decay.
        num_iter : int, default=100
            Number of iterations to run the training.
        stop_div : bool, default
            Whether to stop training if the loss diverges.
        max_mult : int, default=4
            Divergence threshold. If loss >= max_mult * minimum loss, stop
            training.
        """
        n_epochs = num_iter // len(self.dls.train) + 1
        callbacks = [
            LRFinder(gamma, num_iter, stop_div, max_mult),
            Recorder("lr"),
        ]
        self.fit(n_epochs, run_valid=False, callbacks=callbacks, lr=start_lr)
        self.recorder.plot()

    def save_model(
        self,
        path: str | Path | None = None,
        with_opt: bool = False,
        with_epoch: bool = False,
        with_loss: bool = False,
        pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        """
        Save the model and optionally the optimizer, epoch, and the loss.
        Useful for checkpointing.

        Parameters
        ----------
        path : str | Path | None, default=None
            File path to save the model. If None, use
            `learner.model_dir_path`/model.
        with_opt : bool, default=False
            Whether to save the optimizer state.
        with_epoch : bool, default=False
            Whether to save the current epoch number.
        with_loss : bool, default=False
            Whether to save the current loss.
        pickle_protocol : int, default=pickle.HIGHEST_PROTOCOL
            Protocol used by pickler when saving the checkpoint.
        """
        if path is None:
            path = self.model_dir_path / "model"
        checkpoint = {
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
        Load the model and optionally the optimizer, epoch, and the loss.

        Parameters
        ----------
        path : str | Path | None, default=None
            Model's file path. If None, use `learner.model_dir_path`/model.
        with_opt : bool, default=False
            Whether to load the optimizer state.
        with_epoch : bool, default=False
            Whether to load the current epoch number.
        with_loss : bool, default=False
            Whether to load the current loss.
        """
        if path is None:
            path = self.model_dir_path / "model"
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if with_opt:
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if with_epoch:
            self.epoch = checkpoint["epoch"]
        if with_loss:
            self.loss = checkpoint["loss"]

    @property
    def training(self) -> bool:
        return self.model.training

    @training.setter
    def training(self, v) -> None:
        self.model.training = v

    def _add_callbacks(self, cbs):
        added_callbacks = []
        for cb in listify(cbs):
            if not hasattr(self, cb.name):
                cb.set_learner(self)
                setattr(self, cb.name, cb)
                self.callbacks.append(cb)
                added_callbacks.append(cb)
        return added_callbacks

    def _remove_callbacks(self, cbs):
        for cb in listify(cbs):
            delattr(self, cb.name)
            self.callbacks.remove(cb)

    def _callback(self, event_nm):
        for cb in sorted(self.callbacks, key=lambda x: x.order):
            cb(event_nm)

    def summary(self, verbose: int = 2, **kwargs):
        """Use `torchinfo` package to print out the model summary."""
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
        callbacks: Iterable[Callable] | None = None,
        **kwargs,
    ):
        """
        Show `sample_sz` batch of input. The input would be what the
        model would see when making predictions. Therefore, all
        transformations and other augmentation will be applied to the
        input.


        Parameters
        ----------
        sample_sz : int, default=1
            Number of input samples to show.
        callbacks : Iterable[Callback] | None, default=None
            Callbacks to add to the existing callbacks. The added
            callbacks will be removed  before `show_batch` returns.

        Raises
        ------
        NotImplementedError
            Different types of `Learner`'s must implement their own
            version depending on the type of input data. For example,
            `VisionLearner`'s would show images.
        """
        raise NotImplementedError()
