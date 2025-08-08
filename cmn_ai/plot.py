"""
This module provides different plotting utilities that are commonly used in ML
context such as plotting a grid of images or show a single image.
"""

from __future__ import annotations

import math
from itertools import zip_longest

import fastcore.all as fc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage


@fc.delegates(plt.Axes.imshow)
def show_image(
    image,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    noframe: bool = True,
    **kwargs,
) -> AxesImage:
    """
    Show a PIL or PyTorch image on `ax`.

    Parameters
    ----------
    image : PIL image or array-like
        Image data.
    ax : Axes
        Axes to plot the image on.
    figsize : tuple, default=None
        Width, height in inches of the returned Figure
    title : str, default=None
        Title of the image
    noframe : bool, default=True
        Whether to remove axis from the plotted image.

    Returns
    -------
    ax : AxesImage
        Plotted image on `ax`.
    """
    if fc.hasattrs(image, ("cpu", "permute", "detach")):
        image = image.detach().cpu()
        if len(image.shape) == 3 and image.shape[0] < 5:
            image = image.permute(1, 2, 0)
    elif not isinstance(image, np.ndarray):
        image = np.array(image)
    if image.shape[-1] == 1:
        image = image[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if noframe:
        ax.axis("off")
    return ax


@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple = None,
    imsize: int = 3,
    suptitle: str = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    A figure and set of subplots to display images of `imsize` inches.

    Parameters
    ----------
    nrows : int, default=1
        Number of rows in returned axes grid.
    ncols : int, default=1
        Number of columns in returned axes grid.
    figsize : tuple, default=None
        Width, height in inches of the returned Figure.
    imsize : int, default=3
        Size (in inches) of images that will be displayed in the returned figure.
    suptitle : str, default=None
        Title of the Figure.

    Returns
    -------
    fig : Figure
        Top level container for all axes.
    ax : array of Axes
        Array of axes.
    """
    if figsize is None:
        figsize = (ncols * imsize, nrows * imsize)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows * ncols == 1:
        ax = np.array([ax])
    return fig, ax


@fc.delegates(subplots)
def get_grid(
    n: int,
    nrows: int = None,
    ncols: int = None,
    title: str = None,
    weight: str = "bold",
    size: int = 14,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Return a grid of `n` axes, `rows` by `cols`. `nrows` and `ncols` are mutually
    exclusive. This means you only need to pass one of them and the other will be
    inferred.

    Parameters
    ----------
    n : int
        Number of axes.
    nrows : int, optional
        Number of rows, default=`int(math.sqrt(n))`.
    ncols : int, optional
        Number of columns, default=`ceil(n/rows)`.
    title : str, optional
        Title of the Figure.
    weight : str, default='bold'
        Title font weight.
    size : int, default=14
        Title font size.

    Returns
    -------
    fig : Figure
        Top level container for all axes.
    ax : array of Axes
        Array of axes.
    """
    if nrows:
        ncols = ncols or int(np.floor(n / nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n / ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n / nrows))
    fig, axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows * ncols):
        axs.flat[i].set_axis_off()
    if title is not None:
        fig.suptitle(title, weight=weight, size=size)
    return fig, axs


@fc.delegates(subplots)
def show_images(
    images: list,
    nrows: int | None = None,
    ncols: int | None = None,
    titles: list | None = None,
    **kwargs,
) -> None:
    """
    Show all images as subplots with `nrows` x `ncols` using `titles`.

    Parameters
    ----------
    images : list or array-like
        List of images to show.
    nrows : int, default=None
        Number of rows in the grid.
    ncols : int, default=None
        Number of columns in the grid.
    titles : list, default=None
        List of titles for each image.
    """
    axs = get_grid(len(images), nrows, ncols, **kwargs)[1].flat
    for im, t, ax in zip_longest(images, titles or [], axs):
        show_image(im, ax=ax, title=t)
