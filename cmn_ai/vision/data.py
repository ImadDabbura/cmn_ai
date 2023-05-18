from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Callable, Iterable

import PIL

from ..utils.data import ItemList, get_files

IMAGE_EXTENSIONS = [
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
]


class ImageList(ItemList):
    """
    Build an image list from list of files in the `path` end with
    extensions, optionally recursively.

    Parameters
    ----------
    path : str | Path
        Path for the root directory to search for files.
    extensions : str | Iterable[str] | None, default=IMAGE_EXTENSIONS
        Suffixes of filenames to look for.
    include : Iterable[str] | None, default=None
        Top-level Director(y|ies) under `path` to use to search for files.
    recurse : bool, default=True
        Whether to search subdirectories recursively.
    tfms : Callable | None, default=None
        Transformations to apply items before returning them.
    Returns
    -------
    list[str]
        List of filenames that ends with `extensions` under `path`.
    """

    @classmethod
    def from_files(
        cls,
        path: str | Path,
        extensions: Iterable[str] | str = IMAGE_EXTENSIONS,
        include: Iterable[str] | None = None,
        recurse: bool = True,
        tfms: Callable | None = None,
        **kwargs,
    ) -> ImageList:
        return cls(
            get_files(path, extensions, include, recurse), path, tfms, **kwargs
        )

    def get(self, item) -> PIL.Image:
        """Open an image using PIL."""
        return PIL.Image.open(item)
