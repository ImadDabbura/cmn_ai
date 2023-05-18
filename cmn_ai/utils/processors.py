"""
Processors are meant as transformers for DL datasets that can be used by
Dataset object to transform objects before returning a requested item.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .utils import uniqueify


class Processor:
    """Base class for all processors."""

    def process(self, item):
        return item


class CategoryProcessor(Processor):
    """
    Create a vocabulary from training data and use it to numericalize
    categories/text.
    """

    def __init__(self) -> None:
        self.vocab = None

    def __call__(self, items: Sequence[str] | str) -> list[int] | int:
        """
        Create a vocabulary from items if it doesn't already exist and return
        their numerical IDs.

        Parameters
        ----------
        items : Sequence[int] | int
            Data to numericalize.

        Returns
        -------
        list[int] | int
            Numerical IDs of items.
        """
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi = {k: v for v, k in enumerate(self.vocab)}
        return self.process(items)

    def process(self, items: Iterable[str] | str) -> list[int] | int:
        if isinstance(items, Iterable):
            return [self.otoi[item] for item in items]
        return self.otoi[items]

    def deprocess(self, idxs: Iterable[str] | int) -> list[str] | str:
        if isinstance(idxs, Iterable):
            return [self.vocab[idx] for idx in idxs]
        return self.vocab[idxs]
