from __future__ import annotations

import gc
import random
import sys
import traceback
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def listify(obj: Any) -> list:
    """
    Change type of any object into a list.

    Parameters
    ----------
    obj : Any
        Object to turn into a list.

    Returns
    -------
    out : list
        Returns list of the provided `obj`.
    """
    if obj is None:
        return []
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, str):
        return [obj]
    elif isinstance(obj, Iterable):
        return list(obj)
    return [obj]


def tuplify(obj: Any) -> tuple:
    """
    Change type of any object into a tuple.

    Parameters
    ----------
    obj : Any
        Object to turn into a tuple.

    Returns
    -------
    out : tuple
        Returns tuple of the provided `obj`.
    """
    if isinstance(obj, tuple):
        return obj
    return tuple(listify(obj))


def setify(obj: any) -> set:
    """
    Change type of any object into a set.

    Parameters
    ----------
    obj : Any
        Object to turn into a set.

    Returns
    -------
    out : set
        Returns set of the provided `obj`.
    """
    if isinstance(obj, set):
        return obj
    return set(listify(obj))


def uniqueify(x: Iterable, sort: bool = False) -> list:
    """
    Returns a list unique elements in any iterable, optionally sorted.

    Parameters
    ----------
    x : Iterable
        iterable to get unique elements from.
    sort : bool, default=False
        whether to sort the unique elements in the list.

    Returns
    -------
    output : list
        List containing the unique elements.
    """
    output = listify(setify(x))
    if sort:
        output.sort()
    return output


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set seeds for generating random numbers for pytorch, numpy, and random
    packages.

    Parameters
    ----------
    seed : int, default=42
        Desired seed.
    deterministic : bool, default=False
        Whether pytorch uses deterministic algorithms.
    """
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def clean_ipython_history():
    # Code in this function mainly copied from IPython source
    if "get_ipython" not in globals():
        return
    ip = get_ipython()  # noqa: F821
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop("_i" + repr(n), None)
    user_ns.update(dict(_i="", _ii="", _iii=""))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [""] * pc
    hm.input_hist_raw[:] = [""] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ""


def clean_traceback():
    # Very useful especially when traceback has big tensors
    # attached to a traceback while the operation raised Exception
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")


def clean_memory():
    clean_traceback()
    clean_ipython_history()
    gc.collect()
    torch.cuda.empty_cache()
