from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def listify(obj: Any) -> list:
    "Change type of any object into a list."
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
    "Change type of any object into a tuple."
    if isinstance(obj, tuple):
        return obj
    return tuple(listify(obj))
