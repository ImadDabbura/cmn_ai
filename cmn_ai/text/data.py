from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from ..utils.data import ItemList, get_files


class TextList(ItemList):
    """
    Build a text list from list of files in the `path` end with
    `extensions`, optionally recursively.
    """

    @classmethod
    def from_files(
        cls,
        path: str | Path,
        extensions: Iterable[str] | str = ".txt",
        include: Iterable[str] | None = None,
        recurse: bool = True,
        tfms: Callable | None = None,
        encoding="utf8",
    ) -> TextList:
        """
        Parameters
        ----------
        path : str | Path
            Path for the root directory to search for files.
        extensions : str | Iterable[str] | None, default=IMAGE_EXTENSIONS
            Suffixes of filenames to look for.
        include : Iterable[str] | None, default=None
            Top-level Director(y|ies) under `path` to use to search for
            files.
        recurse : bool, default=True
            Whether to search subdirectories recursively.
        tfms : Callable | None, default=None
            Transformations to apply items before returning them.
        encoding : str, default=utf8
            Name of encoding used to decode files.
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
