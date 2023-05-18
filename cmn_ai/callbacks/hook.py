"""
Hooks are very useful to inspect what is happening during the forward and
backward passes such as computing stats of the activations and gradients.

The module contains the following classes:

- `Hook`: Registers forward or backward hook for a single module
- `Hooks`: Registers forward or backward hook for multiple modules
- `HooksCallback`: Use callbacks to register and manage hooks
- `ActivationStats`: Computes means/stds for either activation or gradients
    and plot the computed stats.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Iterable, Iterator

import fastcore.all as fc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.image import AxesImage
from torch import Tensor

from ..plot import get_grid, show_image
from .core import Callback


class Hook:
    """
    Register either a forward or a backward hook on the module.

    Parameters
    ----------
    module : nn.Module
        The module to register the hook on.
    func : Callable
        The hook to be registered.
    is_forward : bool, default=True
        Whether to register `func` as a forward or backward hook.
    """

    def __init__(
        self,
        module: nn.Module,
        func: Callable,
        is_forward: bool = True,
        **kwargs,
    ) -> None:
        self.is_forward = is_forward
        if self.is_forward:
            self.hook = module.register_forward_hook(
                partial(func, self, **kwargs)
            )
        else:
            self.hook = module.register_backward_hook(
                partial(func, self, **kwargs)
            )

    def __enter__(self) -> Hook:
        return self

    def __exit__(self) -> None:
        self.remove()

    def remove(self) -> None:
        self.hook.remove()

    def __del__(self) -> None:
        self.remove()


class Hooks:
    """"""

    """
    Register hooks on all modules.

    Parameters
    ----------
    modules : nn.Module or Iterable[nn.Module]
        The module to register the hook on.
    func : Callable
        The hook to be registered.
    is_forward : bool, default=True
        Whether to register `func` as a forward or backward hook.
    """

    def __init__(
        self,
        modules: nn.Module | Iterable[nn.Module],
        func: Callable,
        is_forward: bool = True,
        **kwargs,
    ) -> None:
        self.hooks = [
            Hook(module, func, is_forward, **kwargs) for module in modules
        ]

    def __getitem__(self, idx) -> Hook:
        return self.hooks[idx]

    def __len__(self) -> int:
        return len(self.hooks)

    def __iter__(self) -> Iterator:
        return iter(self.hooks)

    def __enter__(self, *args) -> Hooks:
        return self

    def __exit__(self, *args) -> None:
        self.remove()

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __del__(self) -> None:
        self.remove()


def compute_stats(
    hook: Hook,
    module: nn.Module,
    inp: Tensor,
    outp: Tensor,
    bins: int = 40,
    bins_range: list | tuple = (0, 10),
) -> None:
    """
    Compute the means, std, and histogram of `module` activations/gradients
    and the `hook` stats.

    Parameters
    ----------
    hook : Hook
        Registered hook on the provided module.
    module : nn.Module
        Module to compute the stats on.
    inp : Tensor
        Input of the module.
    outp : Tensor
        Output of the module.
    bins : int, default=40
        Number of histogram bins.
    bins_range : Iterable, default=(0, 10)
        lower/upper end of the histogram's bins range.
    """
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    if not hook.is_forward:
        inp = inp[0], outp = outp[0]
    hook.stats[0].append(outp.data.mean().cpu())
    hook.stats[1].append(outp.data.std().cpu())
    hook.stats[2].append(outp.data.cpu().histc(bins, *bins_range))


def get_hist(hook: Hook) -> Tensor:
    """Return matrix-ready for plotting heatmap of activations/gradients."""
    return torch.stack(hook.stats[2]).t().float().log1p()


def get_min(hook: Hook, bins_range: list | tuple) -> Tensor:
    """
    Compute the percentage of activations/gradients around zero from hook's
    histogram matrix.

    Parameters
    ----------
    hook : Hook
        Hook that has the stats of the activations
    bins_range : list | tuple
        Bins range around zero.

    Returns
    -------
    Tensor
        Percentage of the activations around zero.
    """
    res = torch.stack(hook.stats[2]).t().float()
    return res[slice(*bins_range)].sum(0) / res.sum(0)


class HooksCallback(Callback):
    """
    Base call to run hooks on modules as a callback.

    Parameters
    ----------
    hookfunc : Callable
        The hook to be registered.
    on_train : bool, default=True
        Whether to run the hook on modules during training.
    on_valid : bool, default=False
        Whether to run the hook on modules during validation.
    modules : nn.Module | Iterable[nn.Module] | None, default=None
        Modules to register the hook on. Default to all modules.
    is_forward : bool, default=True
        Whether to register `func` as a forward or backward hook.
    """

    def __init__(
        self,
        hookfunc: Callable,
        on_train: bool = True,
        on_valid: bool = False,
        modules: nn.Module | Iterable[nn.Module] | None = None,
        is_forward: bool = True,
    ) -> None:
        self.hookfunc = hookfunc
        self.on_train = on_train
        self.on_valid = on_valid
        self.modules = modules
        self.is_forward = is_forward

    def before_fit(self) -> None:
        if self.modules is None:
            self.modules = self.model.modules()
        self.hooks = Hooks(self.modules, self._hookfunc, self.is_forward)

    def _hookfunc(self, *args, **kwargs):
        if (self.on_train and self.training) or (
            self.on_valid and not self.training
        ):
            self.hookfunc(*args, **kwargs)

    def after_fit(self) -> None:
        self.hooks.remove()

    def __iter__(self) -> Iterator:
        return iter(self.hooks)

    def __len__(self) -> int:
        return len(self.hooks)


class ActivationStats(HooksCallback):
    """
    Plot the means, std, histogram, and dead activations of all modules'
    activations if `is_forward` else gradients.

    Parameters
    ----------
    modules : nn.Module | Iterable[nn.Module] | None, default=None
        Modules to register the hook on. Default to all modules.
    is_forward : bool, default=True
        Whether to register `func` as a forward or backward hook.
    bins : int, default=40
        Number of histogram bins.
    bins_range : Iterable, default=(0, 10)
        lower/upper end of the histogram's bins range.
    """

    def __init__(
        self,
        modules: nn.Module | Iterable[nn.Module] | None = None,
        is_forward: bool = True,
        bins: int = 40,
        bins_range: list | tuple = (0, 10),
    ) -> None:
        self.bins_range = bins_range
        super().__init__(
            partial(compute_stats, bins=bins, bins_range=bins_range),
            is_forward=is_forward,
            modules=modules,
        )

    def plot_hist(self, figsize=(11, 5)) -> None:
        _, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin="lower")

    def dead_chart(self, figsize=(11, 5)) -> None:
        _, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(get_min(h, self.bins_range))
            ax.set_ylim(0, 1)

    def plot_stats(self, figsize=(10, 4)) -> None:
        _, axes = plt.subplots(1, 2, figsize=figsize)
        for h in self:
            axes[0].plot(h.stats[0])
            axes[1].plot(h.stats[1])
        axes[0].set_title(
            f"Means of {'activations' if self.is_forward else 'gradients'}"
        )
        axes[1].set_title(
            f"Stdevs of {'activations' if self.is_forward else 'gradients'}"
        )
        plt.legend(fc.L.range(self))
