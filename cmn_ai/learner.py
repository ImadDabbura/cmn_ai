import pickle
from collections.abc import Callable, Iterable
from pathlib import Path

import fastcore.all as fc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch import tensor
from torch.utils.data import DataLoader

from .callbacks.core import (
    Callback,
    CancelBackwardException,
    CancelBatchException,
    CancelEpochException,
    CancelFitException,
    CancelStepException,
    CancelTrainException,
    CancelValidException,
)
from .callbacks.training import (
    LRFinder,
    ProgressCallback,
    Recorder,
    TrainEvalCallback,
)
from .utils.utils import listify


def params_getter(model):
    return model.params_getter()


class Learner:
    """
    Learner is a basic class that handles training loop of pytorch model
    and utilize a systems of callbacks that makes training loop very
    customizable and extensible. You just need to provide a list of
    callbacks and callback functions.

    Parameters
    ----------
    model : nn.Module
        pytorch's model.
    dls : Iterable of Dataloaders.
        train and valid data loaders.
    n_inp : int, default=1
        Number of inputs to the model.
    loss_func : Callable, default=`MSE`
        Loss function.
    opt_func : Optimizer, default=`SGD`
        Optimizer function/class.
    lr : float, default=`1e-2`
        Learning rate.
    splitter : Callable, default=`params_getter`
        Function to split model's parameters into groups, default all
        parameters belong to 1 group.
    callbacks : Iterable, default=None
        Iterable of callbacks of type `Callback`.
    default_callbacks : bool, default=True
        Whether to add `TrainEvalCallback`, `ProgressCallback`, and `Recorder`
        to the list of callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        dls: Iterable[DataLoader],
        n_inp: int = 1,
        loss_func: Callable[[tensor, tensor], tensor] = F.mse_loss,
        opt_func: opt.Optimizer = opt.SGD,
        lr: float = 1e-2,
        splitter: Callable = params_getter,
        path: str = ".",
        model_dir: str = "models",
        callbacks: Iterable[Callback] | None = None,
        default_callbacks: bool = True,
    ):
        self.model = model
        self.dls = dls
        self.n_inp = n_inp
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.opt = None
        self.lr = lr
        self.splitter = splitter
        self.path = Path(path)
        self.model_dir = Path(model_dir)
        self.logger = print
        self.callbacks = fc.L()
        if default_callbacks:
            self.add_callbacks(
                [
                    TrainEvalCallback(),
                    ProgressCallback(),
                    Recorder("lr"),
                ]
            )
        self.add_callbacks(callbacks)

    def _with_events(self, func, event_nm, exc):
        try:
            self.callback(f"before_{event_nm}")
            func()
        except exc:
            self.callback(f"after_cancel_{event_nm}")
        finally:
            self.callback(f"after_{event_nm}")

    def _backward(self):
        self.loss.backward()

    def _step(self):
        self.opt.step()

    def _one_batch(self):
        self.preds = self.model(*self.batch[: self.n_inp])
        self.callback("after_predict")
        self.loss = self.loss_func(self.preds, *self.batch[self.n_inp :])
        self.callback("after_loss")
        if self.training:
            self._with_events(
                self._backward, "backward", CancelBackwardException
            )
            self._with_events(self._step, "step", CancelStepException)
            self.opt.zero_grad()

    def _all_batches(self):
        self.iters = len(self.dl)
        for self.iter, self.batch in enumerate(self.dl):
            self.xb, self.yb = self.batch
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
                self._all_batches, "validate", CancelValidException
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
        train: bool = True,
        valid: bool = True,
        callbacks: Iterable | None = None,
        lr: float | None = None,
        reset_opt: bool = False,
    ):
        """
        Fit the model for `n_epochs`.

        Parameters
        ----------
        n_epochs : int, default=1
            Number epochs to train the model.
        train : bool, default=True
            Whether to run training passes.
        valid : bool, default=True
            Whether to run validation passes.
        callbacks : Iterable | None, default=None
            Callbacks to add the existing callbacks. The added callbacks will
            be removed when before `fit` returns.
        lr : float | None, default=None
            Learning rate. If None, use `lr` passed when created `Learner`.
        reset_opt : bool, default=False
            Whether to reset the optimizer.
        """
        self.run_train = train
        self.run_valid = valid
        self.add_callbacks(callbacks)
        self.n_epochs = n_epochs
        if lr is None:
            lr = self.lr
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.model.parameters(), lr)
        try:
            self._with_events(self._fit, "fit", CancelFitException)
        finally:
            self.remove_callbacks(callbacks)

    def lr_find(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        stop_div: bool = True,
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
        end_lr : float, default=10.0
            Last learning rate in the schedule.
        num_iter : int, default=100
            Number of iterations to run the training.
        stop_div : bool, default
            Whether to stop training if the loss diverges.
        """
        n_epochs = num_iter // len(self.dls.train) + 1
        callbacks = [LRFinder(start_lr, end_lr, num_iter, stop_div)]
        if not self.callbacks or "Recorder" not in [
            cb.__class__.__name__ for cb in self.callbacks
        ]:
            callbacks.append(Recorder("lr"))
        self.fit(n_epochs, valid=False, callbacks=callbacks)
        self.recorder.plot_param(param="lr")

    def save_model(
        self,
        path: str | Path,
        with_opt: bool = False,
        with_epoch: bool = False,
        with_loss: bool = False,
        pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
    ):
        """
        Save the model and optionally the optimizer, epoch, and the loss.
        Useful for checkpointing.

        Parameters
        ----------
        path : str | Path
            File path to save the model.
        with_opt : bool, default=False
            Whether to save the optimizer state.
        with_epoch : bool, default=False
            Whether to save the current epoch number.
        with_loss : bool, default=False
            Whether to save the current loss.
        pickle_protocol : int, default=pickle.HIGHEST_PROTOCOL
            Protocol used by pickler when saving the checkpoint.
        """
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
        path: str | Path,
        with_opt: bool = False,
        with_epoch: bool = False,
        with_loss: bool = False,
    ):
        """
        Load the model and optionally the optimizer, epoch, and the loss.

        Parameters
        ----------
        path : str | Path
            Model's file path.
        with_opt : bool, default=False
            Whether to load the optimizer state.
        with_epoch : bool, default=False
            Whether to load the current epoch number.
        with_loss : bool, default=False
            Whether to load the current loss.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if with_opt:
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if with_epoch:
            self.epoch = checkpoint["epoch"]
        if with_loss:
            self.loss = checkpoint["loss"]

    @property
    def training(self):
        return self.model.training

    @training.setter
    def training(self, v):
        self.model.training = v

    def add_callbacks(self, cbs):
        for cb in listify(cbs):
            cb.set_learner(self)
            setattr(self, cb.name, cb)
            self.callbacks.append(cb)

    def remove_callbacks(self, cbs):
        for cb in listify(cbs):
            delattr(self, cb.name)
            self.callbacks.remove(cb)

    def callback(self, event_nm):
        for cb in sorted(self.callbacks, key=lambda x: x.order):
            cb(event_nm)
