"""
Activation functions for neural networks.

This module provides custom activation functions that extend the standard
PyTorch activation functions with additional functionality and flexibility.

Classes
-------
GeneralRelu : nn.Module
    A generalized Rectified Linear Unit with configurable leaky slope,
    output subtraction, and maximum value clipping.

Notes
-----
All activation functions in this module are designed to be compatible with
PyTorch's nn.Module interface and can be used as drop-in replacements for
standard activation functions in neural network architectures.

Examples
--------
>>> import torch
>>> from cmn_ai.activations import GeneralRelu
>>>
>>> # Create a generalized ReLU with custom parameters
>>> act = GeneralRelu(leak=0.1, sub=0.4, maxv=6.0)
>>>
>>> # Apply to input tensor
>>> x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
>>> output = act(x)
>>> print(output)
tensor([-0.6000, -0.4500,  0.1000,  1.6000])
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class GeneralRelu(nn.Module):
    """
    A generalized Rectified Linear Unit (ReLU) activation function with optional
    leaky slope, output subtraction, and maximum value clipping.

    Parameters
    ----------
    leak : float, optional
        Negative slope for values less than zero, similar to LeakyReLU.
        If None (default), standard ReLU behavior is used (all negatives set to 0).
    sub : float, optional
        A constant value to subtract from the activation output after applying ReLU/LeakyReLU.
        If None (default), no subtraction is applied.
    maxv : float, optional
        Maximum value to clip the activation output to. If None (default), no clipping is applied.

    Attributes
    ----------
    leak : float or None
        The negative slope applied to negative inputs.
    sub : float or None
        The value subtracted from the activation output.
    maxv : float or None
        The upper bound for output clipping.

    Methods
    -------
    forward(x)
        Applies the configured activation transformation to the input tensor.

    Examples
    --------
    >>> import torch
    >>> act = GeneralReLU(leak=0.1, sub=0.4, maxv=6.0)
    >>> x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    >>> act(x)
    tensor([-0.6000, -0.4500,  0.1000,  1.6000])
    """

    def __init__(
        self, leak: float = 0.1, sub: float = 0.4, maxv: float | None = None
    ):
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.maxv = maxv

    def forward(self, x):
        """
        Apply the generalized ReLU activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor, possibly leaky for negative values, with optional
            subtraction and clipping applied.
        """
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x -= self.sub
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x
