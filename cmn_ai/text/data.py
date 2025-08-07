"""
Text data utilities for text processing and dataset management.

This module provides utilities for working with text data in machine learning
pipelines. It includes classes and functions for loading, managing, and
processing text datasets with support for various text file formats and
transformations.

The module is designed to work seamlessly with text files and integrates
with the broader cmn_ai data processing pipeline.

Functions
---------
None

Classes
-------
TextList : Text dataset container for managing collections of text files

Notes
-----
This module extends the base data utilities to provide text-specific
functionality for natural language processing workflows.

Examples
--------
>>> from cmn_ai.text.data import TextList
>>> # Create text dataset from directory
>>> texts = TextList.from_files('./data/texts')
>>> # Access first text file
>>> text = texts[0]  # Returns string content
>>> print(len(text))  # Number of characters
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

from ..utils.data import ItemList, get_files


class TextList(ItemList):
    """
    Text dataset container for managing collections of text files.

    A specialized ItemList subclass for handling text datasets. It provides
    functionality to load and manage collections of text files with support
    for various text formats, transformations, and file discovery.

    Attributes
    ----------
    path : Path
        Base path where the text files are located.
    tfms : callable, optional
        Transformations to apply to text when retrieved.
    data : list[Path]
        List of text file paths.
    encoding : str
        Character encoding used to read text files.

    Notes
    -----
    This class extends ItemList to provide text-specific functionality.
    Text files are loaded and decoded when accessed.

    Examples
    --------
    >>> # Create TextList from a directory
    >>> text_list = TextList.from_files('./data/texts')
    >>> # Access first text file
    >>> text = text_list[0]  # Returns string content
    >>> # Create with specific encoding
    >>> text_list = TextList.from_files('./data', encoding='latin-1')
    >>> # Create with transformations
    >>> def preprocess(text): return text.lower().strip()
    >>> text_list = TextList.from_files('./data', tfms=preprocess)
    """

    @classmethod
    def from_files(
        cls,
        path: str | Path,
        extensions: Iterable[str] | str = ".txt",
        include: Iterable[str] | None = None,
        recurse: bool = True,
        tfms: Callable | None = None,
        encoding: str = "utf8",
    ) -> TextList:
        """
        Create a TextList from files in a directory.

        Factory method to create a TextList by discovering text files
        in the specified directory. Supports various text formats and
        recursive directory traversal.

        Parameters
        ----------
        path : str or Path
            Path for the root directory to search for text files.
        extensions : str or iterable of str, default=".txt"
            File extensions to include. Defaults to .txt files.
        include : iterable of str, optional
            Top-level directory(ies) under `path` to use for searching files.
            If None, searches all directories.
        recurse : bool, default=True
            Whether to search subdirectories recursively.
        tfms : callable, optional
            Transformations to apply to text before returning them.
        encoding : str, default="utf8"
            Character encoding used to read text files.

        Returns
        -------
        TextList
            New TextList instance containing discovered text files.

        Examples
        --------
        >>> # Create from directory with default settings
        >>> texts = TextList.from_files('./data/texts')
        >>> # Create with specific extensions
        >>> texts = TextList.from_files('./data', extensions=['.txt', '.md'])
        >>> # Create with custom encoding
        >>> texts = TextList.from_files('./data', encoding='latin-1')
        >>> # Create without recursion
        >>> texts = TextList.from_files('./data', recurse=False)
        """
        return cls(
            get_files(path, extensions, include, recurse),
            path,
            tfms,
            encoding=encoding,
        )

    def get(self, i):
        """Returns text in the file as string if `i` is path to a file."""
        if isinstance(i, Path):
            with open(i, encoding=self.encoding) as f:
                return f.read()
        return i
