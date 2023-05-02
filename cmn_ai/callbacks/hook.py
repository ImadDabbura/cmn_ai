from functools import partial
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor


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
        **kwargs
    ):
        self.is_forward = is_forward
        if self.is_forward:
            self.hook = module.register_forward_hook(
                partial(func, self, **kwargs)
            )
        else:
            self.hook = module.register_backward_hook(
                partial(func, self, **kwargs)
            )

    def __enter__(self):
        return self

    def __exit__(self):
        self.remove()

    def remove(self):
        self.hook.remove()

    def __del__(self):
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
        **kwargs
    ):
        self.hooks = [
            Hook(module, func, is_forward, **kwargs) for module in modules
        ]

    def __getitem__(self, idx):
        return self.hooks[idx]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove()


def compute_stats(
    hook: Hook,
    module: nn.Module,
    inp: Tensor,
    outp: Tensor,
    bins: int,
    hist_range: list | tuple = (0, 10),
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
    bins : int
        Number of histogram bins.
    hist_range : Iterable, optional
        lower/upper end of the histogram's bins range.
    """
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    if not hook.is_forward:
        inp = inp[0], outp = outp[0]
    hook.stats[0].append(outp.data.mean().cpu())
    hook.stats[1].append(outp.data.std().cpu())
    hook.stats[2].append(outp.data.cpu().histc(bins, *hist_range))


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
