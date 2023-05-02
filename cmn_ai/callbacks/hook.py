from functools import partial
from typing import Callable, Iterable

import torch
import torch.nn as nn


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
