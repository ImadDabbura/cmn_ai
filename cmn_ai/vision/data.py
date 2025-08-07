"""
Vision data utilities for image processing and dataset management.

This module provides utilities for working with image data in machine learning
pipelines. It includes classes and functions for loading, managing, and
processing image datasets with support for various image formats and
transformations.

The module is designed to work seamlessly with PIL (Python Imaging Library)
and integrates with the broader cmn_ai data processing pipeline.

Functions
---------
None

Classes
-------
ImageList : Image dataset container for managing collections of image files

Constants
---------
IMAGE_EXTENSIONS : list of str
    List of supported image file extensions derived from mimetypes.

Notes
-----
This module extends the base data utilities to provide vision-specific
functionality for image processing workflows.

Examples
--------
>>> from cmn_ai.vision.data import ImageList
>>> # Create image dataset from directory
>>> images = ImageList.from_files('./data/images')
>>> # Access first image
>>> img = images[0]  # Returns PIL.Image object
>>> print(img.size)  # (width, height)
"""

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
    Image dataset container for managing collections of image files.

    A specialized ItemList subclass for handling image datasets. It provides
    functionality to load and manage collections of image files with support
    for various image formats, transformations, and file discovery.

    Attributes
    ----------
    path : Path
        Base path where the image files are located.
    tfms : callable, optional
        Transformations to apply to images when retrieved.
    data : list[Path]
        List of image file paths.

    Notes
    -----
    This class extends ItemList to provide image-specific functionality.
    Images are loaded using PIL (Python Imaging Library) when accessed.

    Examples
    --------
    >>> # Create ImageList from a directory
    >>> image_list = ImageList.from_files('./data/images')
    >>> # Access first image
    >>> img = image_list[0]  # Returns PIL.Image object
    >>> # Create with transformations
    >>> from torchvision import transforms
    >>> tfms = transforms.Compose([transforms.Resize((224, 224))])
    >>> image_list = ImageList.from_files('./data', tfms=tfms)
    """

    @classmethod
    def from_files(
        cls,
        path: str | Path,
        extensions: Iterable[str] | str = IMAGE_EXTENSIONS,
        include: Iterable[str] | None = None,
        recurse: bool = True,
        tfms: Callable | None = None,
    ) -> ImageList:
        """
        Create an ImageList from files in a directory.

        Factory method to create an ImageList by discovering image files
        in the specified directory. Supports various image formats and
        recursive directory traversal.

        Parameters
        ----------
        path : str or Path
            Path for the root directory to search for image files.
        extensions : str or iterable of str, default=IMAGE_EXTENSIONS
            File extensions to include. Defaults to all supported image formats.
        include : iterable of str, optional
            Top-level directory(ies) under `path` to use for searching files.
            If None, searches all directories.
        recurse : bool, default=True
            Whether to search subdirectories recursively.
        tfms : callable, optional
            Transformations to apply to images before returning them.

        Returns
        -------
        ImageList
            New ImageList instance containing discovered image files.

        Examples
        --------
        >>> # Create from directory with default settings
        >>> images = ImageList.from_files('./data/images')
        >>> # Create with specific extensions
        >>> images = ImageList.from_files('./data', extensions=['.jpg', '.png'])
        >>> # Create without recursion
        >>> images = ImageList.from_files('./data', recurse=False)
        """
        return cls(get_files(path, extensions, include, recurse), path, tfms)

    def get(self, item: Path) -> PIL.Image.Image:
        """
        Load and return an image from file path.

        Opens an image file using PIL and returns the PIL Image object.
        This method is called automatically when accessing items from the
        ImageList.

        Parameters
        ----------
        item : Path
            File path to the image to load.

        Returns
        -------
        PIL.Image.Image
            Loaded image object.

        Notes
        -----
        The image is loaded in its original format. If transformations
        are specified in the ImageList, they will be applied after loading.

        Examples
        --------
        >>> image_list = ImageList.from_files('./data')
        >>> img = image_list.get(Path('./data/image.jpg'))
        >>> print(img.size)  # (width, height)
        """
        return PIL.Image.open(item)
