from functools import partial
from typing import Callable

import torch
import torch.nn as nn


class Hook:
    """Register a hook into the module (forward/backward)."""

    def __init__(
        self,
        module: nn.Module,
        func: Callable,
        is_forward: bool = True,
        **kwargs
    ):
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
