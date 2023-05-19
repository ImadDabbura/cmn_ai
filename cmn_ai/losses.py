from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoneReduce:
    def __init__(self, loss_func: Callable):
        """
        Force non-reduction on the loss tensor so it can used later in methods
        such as `Mixup` or `LabelSmoothing`.

        Parameters
        ----------
        loss_func : Callable
            Loss function.
        """
        self.loss_func = loss_func
        self.old_reduction = None

    def __enter__(self):
        if hasattr(self.loss_func, "reduction"):
            self.old_reduction = getattr(self.loss_func, "reduction")
            setattr(self.loss_func, "reduction", "none")
            return self.loss_func
        else:
            return partial(self.loss_func, reduction="none")

    def __exit__(self, exc_type, exc_val, traceback):
        if self.old_reduction is not None:
            setattr(self.loss_func, "reduction", self.old_reduction)


def reduce_loss(
    loss: torch.Tensor, reduction: str | None = None
) -> torch.Tensor:
    """
    Reduce the `loss` tensor using `reduction` method. If `reduction` is None,
    returns the passed loss tensor.

    Parameters
    ----------
    loss : torch.Tensor
        Loss tensor.
    reduction : str | None, default=None
        Reduction applied to the loss tensor.

    Returns
    -------
    torch.Tensor
        Reduced loss tensor.
    """
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )
