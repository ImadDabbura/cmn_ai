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


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing is designed to make the model a little bit less certain of
    it's decision by changing a little bit its target: instead of wanting to
    predict 1 for the correct class and 0 for all the others, we ask it to
    predict 1-eps for the correct class and eps for all the others, with eps a
    (small) positive number.

    Parameters
    ----------
    eps : float, default=0.1
        Weight for the interpolation formula.
    reduction : str, default=mean
        Reduction applied to the loss tensor.
    """

    def __init__(self, eps: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.eps * (loss / c) + (1 - self.eps) * nll
