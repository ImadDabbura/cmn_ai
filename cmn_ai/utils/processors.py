"""
Processors are meant as transformers for DL datasets that can be used by
Dataset object to transform objects before returning a requested item.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .utils import uniqueify


class Processor:
    """Base class for all processors."""


class CategoryProcessor(Processor):
    """
    Create a vocabulary from training data and use it to numericalize
    categories/text.

    Attributes
    ----------
    vocab : Sequence[str]
        Vocabulary used for numericalizing tokens.
    otoi : Dict[str, int]
        Mapping of tokens to their integer indices.
    """

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        """
        Parameters
        ----------
        vocab : Sequence[str] | None, default=None
            Vocabulary to use for numericalizing tokens.
        """
        self.vocab = vocab
        self.otoi = (
            None
            if self.vocab is None
            else {k: v for v, k in enumerate(self.vocab)}
        )

    def __call__(self, items: Sequence[str] | str) -> list[int] | int:
        """
        Create a vocabulary from items if it doesn't already exist and
        return their numerical IDs.

        Parameters
        ----------
        items : Sequence[str] | str
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
        """
        Numericalize item(s).

        Parameters
        ----------
        items : Iterable[str] | str
            Items to numericalize.

        Returns
        -------
        list[int] | int
            Numerical IDs of passed items.
        """
        if isinstance(items, Iterable):
            return [self.otoi[item] for item in items]
        return self.otoi[items]

    def deprocess(self, idxs: Iterable[int] | int) -> list[str] | str:
        """
        Denumericalize item(s) by converting IDs to actual tokens.

        Parameters
        ----------
        idxs : Iterable[int] | int
            IDs to denumricalize.

        Returns
        -------
        list[str] | str
            Tokens that correspond for each ID.
        """
        if isinstance(idxs, Iterable):
            return [self.vocab[idx] for idx in idxs]
        return self.vocab[idxs]
