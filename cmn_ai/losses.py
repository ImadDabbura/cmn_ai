"""
Loss functions and utilities for neural network training.

This module provides various loss functions and utilities for training neural networks.
It includes specialized loss functions like label smoothing cross-entropy and
utilities for loss reduction and non-reduction contexts.

Functions
---------
reduce_loss : Callable
    Reduce loss tensor using specified reduction method.

Classes
-------
NoneReduce : object
    Context manager to force non-reduction on loss functions.
LabelSmoothingCrossEntropy : nn.Module
    Cross-entropy loss with label smoothing for regularization.

Examples
--------
>>> # Use label smoothing cross-entropy
>>> loss_func = LabelSmoothingCrossEntropy(eps=0.1)
>>> loss = loss_func(outputs, targets)

>>> # Temporarily disable loss reduction
>>> with NoneReduce(nn.CrossEntropyLoss()) as loss_func:
...     loss = loss_func(outputs, targets)  # No reduction applied

>>> # Reduce loss manually
>>> reduced_loss = reduce_loss(loss, reduction="mean")
"""

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoneReduce:
    """
    Context manager to force non-reduction on loss functions.

    This class provides a context manager that temporarily modifies a loss
    function to return unreduced loss tensors. This is useful when you need
    to access individual loss values, such as in mixup training or when
    implementing custom loss reduction strategies.

    Attributes
    ----------
    loss_func : Callable
        The loss function to be modified.
    old_reduction : str | None
        The original reduction setting of the loss function.

    Examples
    --------
    >>> # Temporarily disable reduction for cross-entropy loss
    >>> with NoneReduce(nn.CrossEntropyLoss()) as loss_func:
    ...     loss = loss_func(outputs, targets)  # Shape: (batch_size,)
    ...     # Process individual loss values
    ...     weighted_loss = loss * weights

    >>> # Use with custom loss functions
    >>> with NoneReduce(my_custom_loss) as loss_func:
    ...     unreduced_loss = loss_func(predictions, targets)
    """

    def __init__(self, loss_func: Callable):
        """
        Initialize the context manager.

        Parameters
        ----------
        loss_func : Callable
            The loss function to be modified. Should either have a `reduction`
            attribute (like PyTorch loss functions) or accept a `reduction`
            parameter.
        """
        self.loss_func = loss_func
        self.old_reduction = None

    def __enter__(self):
        """
        Enter the context and modify the loss function.

        Returns
        -------
        Callable
            The modified loss function that returns unreduced tensors.

        Notes
        -----
        If the loss function has a `reduction` attribute, it will be temporarily
        set to "none". Otherwise, a partial function is returned with
        `reduction="none"` as a keyword argument.
        """
        if hasattr(self.loss_func, "reduction"):
            self.old_reduction = getattr(self.loss_func, "reduction")
            setattr(self.loss_func, "reduction", "none")
            return self.loss_func
        else:
            return partial(self.loss_func, reduction="none")

    def __exit__(self, exc_type, exc_val, traceback):
        """
        Exit the context and restore the original loss function.

        Parameters
        ----------
        exc_type : type | None
            Exception type if an exception occurred.
        exc_val : Exception | None
            Exception value if an exception occurred.
        traceback : traceback | None
            Traceback if an exception occurred.

        Notes
        -----
        This method restores the original reduction setting of the loss function
        if it was modified. If no modification was made (e.g., the loss function
        didn't have a `reduction` attribute), nothing is done.
        """
        if self.old_reduction is not None:
            setattr(self.loss_func, "reduction", self.old_reduction)


def reduce_loss(
    loss: torch.Tensor, reduction: str | None = None
) -> torch.Tensor:
    """
    Reduce loss tensor using specified reduction method.

    This function applies the specified reduction method to a loss tensor.
    It supports the standard PyTorch reduction methods: "mean", "sum", or None
    (no reduction).

    Parameters
    ----------
    loss : torch.Tensor
        The loss tensor to be reduced.
    reduction : str | None, default=None
        The reduction method to apply:
        - "mean": Compute the mean of all elements
        - "sum": Compute the sum of all elements
        - None: Return the tensor as-is (no reduction)

    Returns
    -------
    torch.Tensor
        The reduced loss tensor. If reduction is None, returns the original tensor.

    Examples
    --------
    >>> # Reduce to mean
    >>> loss = torch.tensor([1.0, 2.0, 3.0])
    >>> mean_loss = reduce_loss(loss, reduction="mean")  # 2.0

    >>> # Reduce to sum
    >>> sum_loss = reduce_loss(loss, reduction="sum")  # 6.0

    >>> # No reduction
    >>> unreduced = reduce_loss(loss, reduction=None)  # tensor([1.0, 2.0, 3.0])
    """
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum() if reduction == "sum" else loss
    )


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing for regularization.

    Label smoothing is a regularization technique that prevents the model from
    becoming overconfident in its predictions. Instead of using hard targets
    (1 for correct class, 0 for others), it uses soft targets where the
    correct class gets probability `1 - ε` and all other classes get `ε / (num_classes - 1)`.

    This helps improve generalization and makes the model more robust to
    label noise and overfitting.

    Attributes
    ----------
    eps : float
        The smoothing factor (epsilon) that controls the amount of smoothing.
    reduction : str
        The reduction method applied to the loss tensor.

    Examples
    --------
    >>> # Create label smoothing cross-entropy loss
    >>> loss_func = LabelSmoothingCrossEntropy(eps=0.1)
    >>>
    >>> # Use in training
    >>> outputs = model(inputs)  # Shape: (batch_size, num_classes)
    >>> targets = torch.tensor([0, 1, 2])  # Shape: (batch_size,)
    >>> loss = loss_func(outputs, targets)
    >>>
    >>> # With custom reduction
    >>> loss_func = LabelSmoothingCrossEntropy(eps=0.05, reduction="sum")
    """

    def __init__(self, eps: float = 0.1, reduction: str = "mean") -> None:
        """
        Initialize the label smoothing cross-entropy loss.

        Parameters
        ----------
        eps : float, default=0.1
            The smoothing factor. Should be between 0 and 1.
            - 0: No smoothing (standard cross-entropy)
            - 1: Maximum smoothing (uniform distribution)
        reduction : str, default="mean"
            The reduction method to apply to the loss tensor.
            Options: "mean", "sum", or "none".

        Notes
        -----
        The smoothing factor `eps` controls how much the target distribution
        is smoothed. A value of 0.1 means the correct class gets probability
        0.9 and the remaining 0.1 is distributed equally among other classes.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the label smoothing cross-entropy loss.

        Parameters
        ----------
        output : torch.Tensor
            The model's output logits. Shape: (batch_size, num_classes)
        target : torch.Tensor
            The target class indices. Shape: (batch_size,)

        Returns
        -------
        torch.Tensor
            The computed loss value.

        Notes
        -----
        The loss is computed as:
        ```
        loss = (1 - eps) * NLL_loss + eps * KL_divergence
        ```
        where NLL_loss is the negative log-likelihood loss and KL_divergence
        is the KL divergence between the predicted distribution and a uniform
        distribution over all classes.
        """
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.eps * (loss / c) + (1 - self.eps) * nll
