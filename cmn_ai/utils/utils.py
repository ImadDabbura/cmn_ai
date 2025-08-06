"""
Useful Python utilities for data manipulation, memory management, and system configuration.

This module provides utility functions for common data science and machine learning tasks.
It includes functions for type conversion, data processing, random seed management,
memory cleanup, and print configuration.

Functions
---------
listify : Convert any object into a list
tuplify : Convert any object into a tuple
setify : Convert any object into a set
uniqueify : Return a list of unique elements from an iterable
set_seed : Set random seeds for reproducibility across multiple libraries
clean_memory : Perform comprehensive memory cleanup
clean_traceback : Clean memory used by traceback objects
clean_ipython_history : Clean IPython command history to free memory
set_printoptions : Set print options for NumPy and PyTorch

Notes
-----
This module is designed to work with PyTorch and NumPy, providing utilities
that are commonly needed in machine learning workflows. The memory cleanup
functions are particularly useful when working with large tensors on GPU.

Examples
--------
>>> from cmn_ai.utils.utils import listify, set_seed, clean_memory
>>>
>>> # Type conversion
>>> listify([1, 2, 3])
[1, 2, 3]
>>>
>>> # Set random seed for reproducibility
>>> set_seed(42)
>>>
>>> # Clean memory after heavy computations
>>> clean_memory()
"""

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
    Convert any object into a list.

    This function handles various input types and converts them to a list format.
    None values are converted to empty lists, strings are wrapped in a list,
    and iterables are converted to lists.

    Parameters
    ----------
    obj : Any
        Object to convert into a list. Can be None, a string, an iterable,
        or any other object.

    Returns
    -------
    list
        List representation of the input object.

    Examples
    --------
    >>> listify(None)
    []
    >>> listify("hello")
    ['hello']
    >>> listify([1, 2, 3])
    [1, 2, 3]
    >>> listify((1, 2, 3))
    [1, 2, 3]
    >>> listify(42)
    [42]
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, Iterable):
        return list(obj)
    return [obj]


def tuplify(obj: Any) -> tuple:
    """
    Convert any object into a tuple.

    This function converts various input types to a tuple format by first
    converting to a list using listify, then converting to a tuple.

    Parameters
    ----------
    obj : Any
        Object to convert into a tuple. Can be None, a string, an iterable,
        or any other object.

    Returns
    -------
    tuple
        Tuple representation of the input object.

    Examples
    --------
    >>> tuplify(None)
    ()
    >>> tuplify("hello")
    ('hello',)
    >>> tuplify([1, 2, 3])
    (1, 2, 3)
    >>> tuplify((1, 2, 3))
    (1, 2, 3)
    """
    if isinstance(obj, tuple):
        return obj
    return tuple(listify(obj))


def setify(obj: Any) -> set:
    """
    Convert any object into a set.

    This function converts various input types to a set format by first
    converting to a list using listify, then converting to a set.

    Parameters
    ----------
    obj : Any
        Object to convert into a set. Can be None, a string, an iterable,
        or any other object.

    Returns
    -------
    set
        Set representation of the input object.

    Examples
    --------
    >>> setify(None)
    set()
    >>> setify("hello")
    {'hello'}
    >>> setify([1, 2, 2, 3])
    {1, 2, 3}
    >>> setify({1, 2, 3})
    {1, 2, 3}
    """
    if isinstance(obj, set):
        return obj
    return set(listify(obj))


def uniqueify(x: Iterable, sort: bool = False) -> list:
    """
    Return a list of unique elements from an iterable.

    This function removes duplicates from an iterable and optionally sorts
    the resulting list.

    Parameters
    ----------
    x : Iterable
        Iterable to extract unique elements from.
    sort : bool, default=False
        Whether to sort the unique elements in ascending order.

    Returns
    -------
    list
        List containing unique elements from the input iterable.

    Examples
    --------
    >>> uniqueify([1, 2, 2, 3, 1])
    [1, 2, 3]
    >>> uniqueify([3, 1, 2, 2, 1], sort=True)
    [1, 2, 3]
    >>> uniqueify("hello")
    ['h', 'e', 'l', 'o']
    """
    output = listify(setify(x))
    if sort:
        output.sort()
    return output


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.

    This function sets the random seed for PyTorch, NumPy, and Python's
    random module to ensure reproducible results across runs.

    Parameters
    ----------
    seed : int, default=42
        Random seed value to use for all random number generators.
    deterministic : bool, default=False
        Whether PyTorch should use deterministic algorithms. When True,
        this may impact performance but ensures reproducibility.

    Notes
    -----
    Setting deterministic=True in PyTorch may significantly slow down
    some operations, especially on GPU. Use only when absolute
    reproducibility is required.

    Examples
    --------
    >>> set_seed(42)
    >>> set_seed(123, deterministic=True)
    """
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def clean_ipython_history() -> None:
    """
    Clean IPython command history to free memory.

    This function clears the IPython history, which is particularly useful
    when working with large tensors or objects that consume significant
    memory. The implementation is based on IPython source code.

    Notes
    -----
    This function only works when running in an IPython environment.
    If not in IPython, the function returns without doing anything.

    Examples
    --------
    >>> clean_ipython_history()  # Clears history if in IPython
    """
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


def clean_traceback() -> None:
    """
    Clean memory used by traceback objects.

    This function clears traceback objects that may hold references to
    large tensors or objects, preventing them from being garbage collected.
    This is particularly useful when dealing with GPU memory issues after
    exceptions.

    Notes
    -----
    When exceptions occur with large tensors, the traceback may keep
    references to these tensors in GPU memory, leading to out-of-memory
    errors even after attempting to clear GPU cache.

    Examples
    --------
    >>> try:
    ...     # Some operation that might fail
    ...     pass
    ... except Exception:
    ...     clean_traceback()  # Clean up traceback memory
    """
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")


def clean_memory() -> None:
    """
    Perform comprehensive memory cleanup.

    This function performs a complete memory cleanup by:
    1. Cleaning traceback objects
    2. Clearing IPython history
    3. Running garbage collection
    4. Emptying PyTorch's CUDA cache

    Notes
    -----
    This is a comprehensive cleanup function that should be called when
    experiencing memory issues, especially when working with large
    tensors on GPU.

    Examples
    --------
    >>> clean_memory()  # Perform complete memory cleanup
    """
    clean_traceback()
    clean_ipython_history()
    gc.collect()
    torch.cuda.empty_cache()


def set_printoptions(
    precision: int = 2, linewidth: int = 125, sci_mode: bool = False
) -> None:
    """
    Set print options for NumPy and PyTorch.

    This function configures the display format for both NumPy arrays and
    PyTorch tensors to ensure consistent output formatting.

    Parameters
    ----------
    precision : int, default=2
        Number of decimal digits to display for floating point numbers.
    linewidth : int, default=125
        Maximum number of characters per line before wrapping.
    sci_mode : bool, default=False
        Whether to use scientific notation for floating point numbers.
        When True, numbers are displayed in scientific notation (e.g., 1e-3).

    Examples
    --------
    >>> set_printoptions(precision=4, linewidth=80)
    >>> set_printoptions(precision=3, sci_mode=True)
    """
    torch.set_printoptions(
        precision=precision, linewidth=linewidth, sci_mode=sci_mode
    )
    np.set_printoptions(precision=precision, linewidth=linewidth)
