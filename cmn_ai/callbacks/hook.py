"""
Hooks for inspecting neural network activations and gradients during training.

This module provides utilities for registering hooks on PyTorch modules to inspect
what is happening during forward and backward passes. This is useful for computing
statistics of activations and gradients, debugging training issues, and monitoring
model behavior.

Hooks are very useful to inspect what is happening during the forward and backward
passes such as computing stats of the activations and gradients.

Functions
---------
compute_stats : Callable
    Compute means, std, and histogram of module activations/gradients.
get_hist : Callable
    Return matrix-ready for plotting heatmap of activations/gradients.
get_min : Callable
    Compute percentage of activations/gradients around zero from histogram.

Classes
-------
Hook : object
    Register either a forward or a backward hook on a single module.
Hooks : object
    Register hooks on multiple modules with context manager support.
HooksCallback : Callback
    Base class to run hooks on modules as a callback.
ActivationStats : HooksCallback
    Plot means, std, histogram, and dead activations of all modules.

Examples
--------
>>> # Register a hook on a single module
>>> hook = Hook(model.layer1, compute_stats)
>>> # Use as context manager
>>> with Hook(model.layer1, compute_stats):
...     output = model(input)

>>> # Register hooks on multiple modules
>>> hooks = Hooks(model.children(), compute_stats)
>>> hooks.remove()  # Clean up

>>> # Use as callback during training
>>> stats = ActivationStats(model, is_forward=True)
>>> learner.add_callback(stats)
>>> stats.plot_stats()  # Plot activation statistics
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor

from ..plot import get_grid, show_image
from ..utils.utils import listify
from .core import Callback


class Hook:
    """
    Register either a forward or a backward hook on a single module.

    This class provides a convenient way to register hooks on PyTorch modules
    and automatically handle cleanup. It can be used as a context manager
    for automatic hook removal.

    Attributes
    ----------
    is_forward : bool
        Whether the hook is a forward or backward hook.
    hook : torch.utils.hooks.RemovableHandle
        The registered hook handle for cleanup.

    Examples
    --------
    >>> hook = Hook(model.layer1, compute_stats)
    >>> # ... use the hook
    >>> hook.remove()

    >>> # Use as context manager
    >>> with Hook(model.layer1, compute_stats):
    ...     output = model(input)
    """

    def __init__(
        self,
        module: nn.Module,
        func: Callable,
        is_forward: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the hook.

        Parameters
        ----------
        module : nn.Module
            The module to register the hook on.
        func : Callable
            The hook function to be registered. Should accept (hook, module, input, output)
            for forward hooks or (hook, module, grad_input, grad_output) for backward hooks.
        is_forward : bool, default=True
            Whether to register `func` as a forward or backward hook.
        **kwargs
            Additional keyword arguments to pass to the hook function.
        """
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
        """
        Enter the context manager.

        Returns
        -------
        Hook
            Self reference for context manager usage.
        """
        return self

    def __exit__(self, *args) -> None:
        """
        Exit the context manager and remove the hook.
        """
        self.remove()

    def remove(self) -> None:
        """
        Remove the registered hook.

        This method removes the hook from the module and should be called
        to prevent memory leaks.
        """
        self.hook.remove()

    def __del__(self) -> None:
        """
        Destructor to ensure hook removal.
        """
        self.remove()


class Hooks:
    """
    Register hooks on multiple modules with convenient management.

    This class provides a container for multiple hooks with convenient
    iteration, indexing, and cleanup methods. It can be used as a context
    manager for automatic cleanup of all hooks.

    Attributes
    ----------
    hooks : list[Hook]
        List of registered hooks.

    Examples
    --------
    >>> hooks = Hooks(model.children(), compute_stats)
    >>> for hook in hooks:
    ...     print(hook.stats)
    >>> hooks.remove()

    >>> # Use as context manager
    >>> with Hooks(model.children(), compute_stats):
    ...     output = model(input)
    """

    def __init__(
        self,
        modules: Iterable[nn.Module],
        func: Callable,
        is_forward: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize hooks for multiple modules.

        Parameters
        ----------
        modules : Iterable[nn.Module]
            Iterable of modules to register hooks on.
        func : Callable
            The hook function to be registered on each module.
        is_forward : bool, default=True
            Whether to register `func` as a forward or backward hook.
        **kwargs
            Additional keyword arguments to pass to the hook function.
        """
        self.hooks = [
            Hook(module, func, is_forward, **kwargs) for module in modules
        ]

    def __getitem__(self, idx) -> Hook:
        """
        Get a hook by index.

        Parameters
        ----------
        idx : int
            Index of the hook to retrieve.

        Returns
        -------
        Hook
            The hook at the specified index.
        """
        return self.hooks[idx]

    def __len__(self) -> int:
        """
        Get the number of hooks.

        Returns
        -------
        int
            Number of registered hooks.
        """
        return len(self.hooks)

    def __iter__(self) -> Iterator[Hook]:
        """
        Iterate over all hooks.

        Returns
        -------
        Iterator[Hook]
            Iterator over all registered hooks.
        """
        return iter(self.hooks)

    def __enter__(self) -> Hooks:
        """
        Enter the context manager.

        Returns
        -------
        Hooks
            Self reference for context manager usage.
        """
        return self

    def __exit__(self, *args) -> None:
        """
        Exit the context manager and remove all hooks.
        """
        self.remove()

    def remove(self) -> None:
        """
        Remove all registered hooks.

        This method removes all hooks from their respective modules and
        should be called to prevent memory leaks.
        """
        for hook in self.hooks:
            hook.remove()

    def __del__(self) -> None:
        """
        Destructor to ensure all hooks are removed.
        """
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
    Compute the means, std, and histogram of module activations/gradients.

    This function is designed to be used as a hook function. It computes
    statistics of the module's output (activations for forward hooks,
    gradients for backward hooks) and stores them in the hook object.

    Parameters
    ----------
    hook : Hook
        The registered hook object where stats will be stored.
    module : nn.Module
        The module that the hook is registered on.
    inp : Tensor
        Input to the module (for forward hooks) or gradient input (for backward hooks).
    outp : Tensor
        Output from the module (for forward hooks) or gradient output (for backward hooks).
    bins : int, default=40
        Number of histogram bins.
    bins_range : list | tuple, default=(0, 10)
        Lower and upper bounds for the histogram bins.

    Notes
    -----
    The computed statistics are stored in `hook.stats` as a tuple of three lists:

    - `hook.stats[0]`: List of mean values
    - `hook.stats[1]`: List of standard deviation values
    - `hook.stats[2]`: List of histogram tensors
    """
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    if not hook.is_forward:
        inp, outp = inp[0], outp[0]
    hook.stats[0].append(outp.data.mean().cpu())
    hook.stats[1].append(outp.data.std().cpu())
    hook.stats[2].append(outp.data.cpu().histc(bins, *bins_range))


def get_hist(hook: Hook) -> Tensor:
    """
    Return matrix-ready for plotting heatmap of activations/gradients.

    Parameters
    ----------
    hook : Hook
        Hook object containing histogram statistics.

    Returns
    -------
    Tensor
        Matrix of histogram data ready for plotting as a heatmap.
        Shape is (bins, timesteps) with log1p applied for better visualization.
    """
    return torch.stack(hook.stats[2]).t().float().log1p()


def get_min(hook: Hook, bins_range: list | tuple) -> Tensor:
    """
    Compute the percentage of activations/gradients around zero.

    This function calculates what percentage of the activations or gradients
    fall within the specified range around zero, which can be useful for
    identifying "dead" neurons or gradients.

    Parameters
    ----------
    hook : Hook
        Hook object containing histogram statistics.
    bins_range : list | tuple
        Range of bins around zero to consider as "dead" activations/gradients.
        Should be a slice-like object (e.g., [0, 5] for bins 0-4).

    Returns
    -------
    Tensor
        Percentage of activations/gradients around zero for each timestep.
        Values range from 0 to 1.

    Examples
    --------
    >>> # Get percentage of activations in bins 0-4 (around zero)
    >>> dead_percentage = get_min(hook, [0, 5])
    >>> print(f"Dead activations: {dead_percentage.mean():.2%}")
    """
    res = torch.stack(hook.stats[2]).t().float()
    return res[slice(*bins_range)].sum(0) / res.sum(0)


class HooksCallback(Callback):
    """
    Base class to run hooks on modules as a callback.

    This class provides a convenient way to register and manage hooks
    during training using the callback system. It automatically handles
    hook registration before training and cleanup after training.

    Attributes
    ----------
    hookfunc : Callable
        The hook function to be registered on modules.
    on_train : bool
        Whether to run hooks during training.
    on_valid : bool
        Whether to run hooks during validation.
    modules : list[nn.Module]
        List of modules to register hooks on.
    is_forward : bool
        Whether to register forward or backward hooks.
    hooks : Hooks
        The hooks object managing all registered hooks.

    Examples
    --------
    >>> # Create a custom hook callback
    >>> class MyHookCallback(HooksCallback):
    ...     def __init__(self):
    ...         super().__init__(compute_stats, on_train=True, on_valid=False)
    >>>
    >>> callback = MyHookCallback()
    >>> learner.add_callback(callback)
    """

    def __init__(
        self,
        hookfunc: Callable,
        on_train: bool = True,
        on_valid: bool = False,
        modules: nn.Module | Iterable[nn.Module] | None = None,
        is_forward: bool = True,
    ) -> None:
        """
        Initialize the hooks callback.

        Parameters
        ----------
        hookfunc : Callable
            The hook function to be registered on modules.
        on_train : bool, default=True
            Whether to run the hook on modules during training.
        on_valid : bool, default=False
            Whether to run the hook on modules during validation.
        modules : nn.Module | Iterable[nn.Module] | None, default=None
            Modules to register the hook on. If None, uses all model children.
        is_forward : bool, default=True
            Whether to register forward or backward hooks.
        """
        super().__init__()
        self.hookfunc = hookfunc
        self.on_train = on_train
        self.on_valid = on_valid
        self.modules = listify(modules)
        self.is_forward = is_forward

    def before_fit(self) -> None:
        """
        Register hooks before training begins.

        If no modules are specified, registers hooks on all model children.
        """
        if not self.modules:
            self.modules = self.model.children()
        self.hooks = Hooks(self.modules, self._hookfunc, self.is_forward)

    def _hookfunc(self, *args, **kwargs):
        """
        Internal hook function that checks training/validation state.

        This method wraps the actual hook function and only calls it
        when the current phase (training/validation) matches the
        configured settings.
        """
        if (self.on_train and self.training) or (
            self.on_valid and not self.training
        ):
            self.hookfunc(*args, **kwargs)

    def after_fit(self) -> None:
        """
        Remove all hooks after training ends.
        """
        self.hooks.remove()

    def __iter__(self) -> Iterator[Hook]:
        """
        Iterate over all registered hooks.

        Returns
        -------
        Iterator[Hook]
            Iterator over all registered hooks.
        """
        return iter(self.hooks)

    def __len__(self) -> int:
        """
        Get the number of registered hooks.

        Returns
        -------
        int
            Number of registered hooks.
        """
        return len(self.hooks)


class ActivationStats(HooksCallback):
    """
    Plot activation/gradient statistics for all modules.

    This class automatically computes and can plot various statistics
    of module activations (or gradients if `is_forward=False`), including
    means, standard deviations, histograms, and dead activation percentages.

    Attributes
    ----------
    bins : int
        Number of histogram bins.
    bins_range : list | tuple
        Range for histogram bins.

    Examples
    --------
    >>> # Monitor activation statistics during training
    >>> stats = ActivationStats(model, is_forward=True)
    >>> learner.add_callback(stats)
    >>>
    >>> # After training, plot the statistics
    >>> stats.plot_stats()
    >>> stats.plot_hist()
    >>> stats.dead_chart([0, 5])  # Show dead activations in bins 0-4
    """

    def __init__(
        self,
        modules: nn.Module | Iterable[nn.Module] | None = None,
        is_forward: bool = True,
        bins: int = 40,
        bins_range: list | tuple = (0, 10),
    ) -> None:
        """
        Initialize the activation statistics callback.

        Parameters
        ----------
        modules : nn.Module | Iterable[nn.Module] | None, default=None
            Modules to register the hook on. If None, uses all model children.
        is_forward : bool, default=True
            Whether to monitor activations (True) or gradients (False).
        bins : int, default=40
            Number of histogram bins.
        bins_range : list | tuple, default=(0, 10)
            Lower and upper bounds for histogram bins.
        """
        super().__init__(
            partial(compute_stats, bins=bins, bins_range=bins_range),
            modules=modules,
            is_forward=is_forward,
        )
        self.bins = bins
        self.bins_range = bins_range

    def plot_hist(self, figsize=(11, 5)) -> None:
        """
        Plot histogram of activations/gradients as a heatmap.

        Creates a heatmap visualization where each row represents a histogram
        bin and each column represents a timestep during training.

        Parameters
        ----------
        figsize : tuple, default=(11, 5)
            Width and height of the figure in inches.
        """
        _, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin="lower", cmap="viridis")

    def dead_chart(self, bins_range, figsize=(11, 5)) -> None:
        """
        Plot a line chart of the "dead" activations percentage.

        Shows the percentage of activations/gradients that fall within
        the specified range around zero over time, which can help identify
        when neurons become inactive.

        Parameters
        ----------
        bins_range : list | tuple
            Range of bins around zero to consider as "dead" activations/gradients.
            Should be a slice-like object (e.g., [0, 5] for bins 0-4).
        figsize : tuple, default=(11, 5)
            Width and height of the figure in inches.
        """
        _, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(get_min(h, bins_range))
            ax.set_ylim(0, 1)

    def plot_stats(self, figsize=(10, 4)) -> None:
        """
        Plot means and standard deviations of activations/gradients.

        Creates two subplots showing the mean and standard deviation
        of activations/gradients for each layer over time.

        Parameters
        ----------
        figsize : tuple, default=(10, 4)
            Width and height of the figure in inches.
        """
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
        plt.legend(range(len(self)))
