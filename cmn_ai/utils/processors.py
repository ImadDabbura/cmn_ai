"""
Data processors for deep learning pipelines.

This module provides processor classes designed to transform data before it is
fed into deep learning models. Processors implement common data preprocessing
patterns such as tokenization, numericalization, and vocabulary management for
text and other sequential data.

The processors follow a consistent interface pattern:

1. Initialize with configuration parameters
2. Fit to training data to learn vocabulary/transformations
3. Transform data to numerical representations
4. Reverse transform (decode) numerical data back to original form

All processors support:

- Vocabulary management with automatic building from training data
- Frequency filtering to remove low-frequency tokens below threshold
- Size limits to respect maximum vocabulary size constraints
- Reserved tokens for special tokens (padding, unknown, etc.)
- Immediate usability when processors have predefined vocabularies
- Modern typing with Python 3.10+ union syntax

Classes
-------
Processor
    Base class for all processors. Defines the processor interface.

NumericalizeProcessor
    Converts tokens to numerical indices using a learned vocabulary.
    Supports frequency filtering, vocabulary size limits, and reserved tokens.
    Can be initialized with a predefined vocabulary for immediate use.

Examples
--------
Basic token numericalization workflow:

>>> from cmn_ai.utils.processors import NumericalizeProcessor
>>> processor = NumericalizeProcessor(min_freq=2, max_vocab=1000)
>>> training_data = [["hello", "world"], ["hello", "there"], ["world", "peace"]]
>>> processor.fit(training_data)
>>> indices = processor.process(["hello", "world"])
>>> print(indices)
[1, 2]
>>> tokens = processor.deprocess(indices)
>>> print(tokens)
['hello', 'world']

Using a predefined vocabulary:

>>> vocab = ["<unk>", "<pad>", "hello", "world", "goodbye"]
>>> processor = NumericalizeProcessor(vocab=vocab)
>>> indices = processor.process(["hello", "world"])
>>> print(indices)
[2, 3]

Including reserved tokens in vocabulary:

>>> processor = NumericalizeProcessor(
...     max_vocab=100,
...     min_freq=1,
...     reserved_tokens=["<pad>", "<eos>", "<sos>"]
... )
>>> processor.fit(training_data)

Notes
-----
Processors include comprehensive error handling:

- `ValueError` is raised when attempting to use unfitted processors
- Unknown tokens are gracefully handled (returns unk_token index)
- Out-of-bounds indices are gracefully handled (returns unk_token)
- Vocabulary size constraints are strictly enforced

Implementation details:

- Vocabulary building respects both frequency thresholds and size limits
- Reserved tokens are prioritized over data tokens when space is limited
- The unk_token is always included and has the highest priority
- All processors use modern Python type hints with union syntax (|)

See Also
--------
cmn_ai.utils.utils : Utility functions used by processors
cmn_ai.text.data : Text data handling utilities
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .utils import listify, uniqueify


class Processor:
    """Base class for all processors."""


class NumericalizeProcessor(Processor):
    """
    A processor that converts tokens to numerical indices and vice versa.

    This processor builds a vocabulary from input tokens and provides methods
    to convert between tokens and their corresponding numerical indices.

    Parameters
    ----------
    vocab : List[str] | None, default=None
        Pre-defined vocabulary. If None, vocabulary will be built from input data.
    max_vocab : int, default=60000
        Maximum vocabulary size.
    min_freq : int, default=2
        Minimum frequency threshold for tokens to be included in vocabulary.
    reserved_tokens : str | List[str] | None, default=None
        Reserved tokens to always include in vocabulary (e.g., special tokens).
    unk_token : str, default="<unk>"
        Token to use for unknown/out-of-vocabulary items.

    Attributes
    ----------
    vocab : List[str]
        The vocabulary list mapping indices to tokens.
    token_to_idx : Dict[str, int]
        Mapping from tokens to their indices.
    idx_to_token : Dict[int, str]
        Mapping from indices to their tokens.
    is_fitted : bool
        Whether the processor has been fitted with data.

    Examples
    --------
    >>> processor = NumericalizeProcessor(min_freq=1)
    >>> tokens = [["hello", "world"], ["hello", "there"]]
    >>> indices = processor(tokens)
    >>> processor.deprocess(indices[0])
    ['hello', 'world']
    """

    def __init__(
        self,
        vocab: list[str] | None = None,
        max_vocab: int = 60000,
        min_freq: int = 2,
        reserved_tokens: str | list[str] | None = None,
        unk_token: str = "<unk>",
    ) -> None:
        self.vocab = vocab
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.reserved_tokens = reserved_tokens
        self.unk_token = unk_token

        # Internal state
        self._token_to_idx: dict[str, int] | None = None
        self._idx_to_token: dict[int, str] | None = None

        # If vocab is provided, the processor is immediately fitted
        if self.vocab is not None:
            self._create_mappings()
            self._is_fitted = True
        else:
            self._is_fitted = False

    def __call__(self, items: list[str | list[str]]) -> list[int | list[int]]:
        """
        Process a list of items, building vocabulary if needed.

        Parameters
        ----------
        items : List[str | List[str]]
            List of tokens or token sequences to process.

        Returns
        -------
        List[int | List[int]]
            List of corresponding indices or index sequences.
        """
        if not self.is_fitted:
            self._build_vocabulary(items)

        return [self.process(item) for item in items]

    def _build_vocabulary(self, items: list[str | list[str]]) -> None:
        """
        Build vocabulary from input items.

        Parameters
        ----------
        items : List[str | List[str]]
            Input items to build vocabulary from.
        """
        # If vocab is already set (from __init__), mappings are already created
        if self.vocab is not None and self._is_fitted:
            return

        # Count token frequencies
        self.token_freqs = Counter()
        for item in items:
            if isinstance(item, (list, tuple)):
                self.token_freqs.update(item)
            else:
                self.token_freqs[item] += 1

        # Build vocabulary respecting max_vocab limit
        reserved = listify(self.reserved_tokens)

        # Start with unk_token (always include it)
        vocab_tokens = [self.unk_token]
        remaining_slots = self.max_vocab - 1

        # Add reserved tokens up to remaining slots
        for token in reserved:
            if token != self.unk_token and remaining_slots > 0:
                vocab_tokens.append(token)
                remaining_slots -= 1

        # Add common tokens from data up to remaining slots
        reserved_set = set(vocab_tokens)  # All tokens we've added so far
        for token, count in self.token_freqs.most_common():
            if remaining_slots <= 0:
                break
            if count < self.min_freq:
                break
            if token not in reserved_set:
                vocab_tokens.append(token)
                reserved_set.add(token)
                remaining_slots -= 1

        # Build final vocabulary (uniqueify and sort)
        self.vocab = uniqueify(vocab_tokens, sort=True)

        self._create_mappings()
        self._is_fitted = True

    def _create_mappings(self) -> None:
        """Create token-to-index and index-to-token mappings."""
        if self.vocab is None:
            raise ValueError("Vocabulary must be set before creating mappings")

        self._token_to_idx = {
            token: idx for idx, token in enumerate(self.vocab)
        }
        self._idx_to_token = {
            idx: token for idx, token in enumerate(self.vocab)
        }

    def process(self, items: str | list[str]) -> int | list[int]:
        """
        Convert tokens to indices.

        Parameters
        ----------
        items : str | List[str]
            Token or list of tokens to convert.

        Returns
        -------
        int | List[int]
            Corresponding index or list of indices.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before processing")

        if isinstance(items, (list, tuple)):
            return [
                self._token_to_idx.get(item, self.unk_idx) for item in items
            ]
        return self._token_to_idx.get(items, self.unk_idx)

    def deprocess(self, indices: int | list[int]) -> str | list[str]:
        """
        Convert indices back to tokens.

        Parameters
        ----------
        indices : int | List[int]
            Index or list of indices to convert.

        Returns
        -------
        str | List[str]
            Corresponding token or list of tokens.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before deprocessing")

        if isinstance(indices, (list, tuple)):
            return [
                self._idx_to_token.get(idx, self.unk_token) for idx in indices
            ]
        return self._idx_to_token.get(indices, self.unk_token)

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        if self.vocab is None:
            return 0
        return len(self.vocab)

    def __getitem__(self, tokens: str | list[str]) -> int | list[int]:
        """
        Get indices for tokens using bracket notation.

        Parameters
        ----------
        tokens : str | List[str]
            Token or list of tokens.

        Returns
        -------
        int | List[int]
            Corresponding index or list of indices.
        """
        return self.process(tokens)

    def fit(self, items: list[str | list[str]]) -> NumericalizeProcessor:
        """
        Fit the processor to the data without processing.

        Parameters
        ----------
        items : List[str | List[str]]
            Items to build vocabulary from.

        Returns
        -------
        NumericalizeProcessor
            Self for method chaining.
        """
        self._build_vocabulary(items)
        return self

    def get_token(self, idx: int) -> str:
        """
        Get token for a given index.

        Parameters
        ----------
        idx : int
            Index to look up.

        Returns
        -------
        str
            Corresponding token.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before getting tokens")
        return self._idx_to_token.get(idx, self.unk_token)

    def get_index(self, token: str) -> int:
        """
        Get index for a given token.

        Parameters
        ----------
        token : str
            Token to look up.

        Returns
        -------
        int
            Corresponding index.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before getting indices")
        return self._token_to_idx.get(token, self.unk_idx)

    @property
    def token_to_idx(self) -> dict[str, int]:
        """Get the token-to-index mapping."""
        if not self.is_fitted:
            raise ValueError(
                "Processor must be fitted before accessing mappings"
            )
        return self._token_to_idx

    @property
    def idx_to_token(self) -> dict[int, str]:
        """Get the index-to-token mapping."""
        if not self.is_fitted:
            raise ValueError(
                "Processor must be fitted before accessing mappings"
            )
        return self._idx_to_token

    @property
    def is_fitted(self) -> bool:
        """Check if the processor has been fitted."""
        return self._is_fitted

    @property
    def unk_idx(self) -> int:
        """Get the index of the unknown token."""
        if not self.is_fitted:
            raise ValueError(
                "Processor must be fitted before accessing unk_idx"
            )
        # If unk_token is not in vocabulary, return index 0 as fallback
        return self._token_to_idx.get(self.unk_token, 0)

    @property
    def unk(self) -> int:
        """Alias for unk_idx for backward compatibility."""
        return self.unk_idx
