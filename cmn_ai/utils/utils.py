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
