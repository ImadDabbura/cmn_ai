import math
from itertools import zip_longest

import fastcore.all as fc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure


@fc.delegates(plt.Axes.imshow)
def show_image(
    image,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    noframe: bool = True,
    **kwargs,
) -> plt.image.AxesImage:
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
    ax : plt.image.AxesImage
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
