"""
Core vision functionality for computer vision tasks.

This module provides specialized learners and utilities for computer vision tasks.
It extends the base Learner class with vision-specific functionality such as
batch visualization and image processing capabilities.

Classes
-------
VisionLearner : Learner
    Learner specialized for computer vision tasks with batch visualization.

Examples
--------
>>> # Create a vision learner
>>> learner = VisionLearner(model, dls, loss_func, opt_func)
>>>
>>> # Show a batch of images
>>> learner.show_batch(sample_sz=4, figsize=(12, 8))
"""

import fastcore.all as fc

from ..callbacks.training import SingleBatchForwardCallback
from ..learner import Learner
from ..plot import show_images
from ..utils.utils import listify


class VisionLearner(Learner):
    """
    Learner specialized for computer vision tasks.

    This class extends the base Learner with vision-specific functionality,
    particularly for visualizing batches of images during training and inference.
    It provides convenient methods for displaying image data and monitoring
    training progress in computer vision applications.

    Examples
    --------
    >>> # Create a vision learner with a CNN model
    >>> model = torchvision.models.resnet18(pretrained=True)
    >>> learner = VisionLearner(model, dls, loss_func, opt_func)
    >>>
    >>> # Show a batch of images
    >>> learner.show_batch(sample_sz=4)
    >>>
    >>> # Train the model
    >>> learner.fit_one_cycle(10)
    """

    @fc.delegates(show_images)
    def show_batch(self, sample_sz=1, callbacks=None, **kwargs):
        """
        Show a batch of images for visualization.

        This method runs a single forward pass through the model and displays
        the input images. It's useful for inspecting the data being fed to
        the model and verifying data preprocessing.

        Parameters
        ----------
        sample_sz : int, default=1
            Number of input samples to show from the batch.
        callbacks : Iterable[Callback] | None, default=None
            Additional callbacks to add temporarily for this visualization.
            These callbacks will be removed after the method completes.
        **kwargs
            Additional keyword arguments passed to `show_images` function.
            Common options include:
            - figsize : tuple, default=(10, 10)
                Figure size in inches (width, height)
            - nrows : int, default=None
                Number of rows in the grid
            - ncols : int, default=None
                Number of columns in the grid
            - title : str, default=None
                Title for the figure
            - cmap : str, default=None
                Colormap for grayscale images

        Notes
        -----
        This method temporarily adds a `SingleBatchForwardCallback` to ensure
        only one batch is processed, regardless of the current training state.
        The method will automatically clean up any additional callbacks that
        were passed in.

        Examples
        --------
        >>> # Show 4 images from the current batch
        >>> learner.show_batch(sample_sz=4, figsize=(12, 8))
        >>>
        >>> # Show images with custom grid layout
        >>> learner.show_batch(sample_sz=9, nrows=3, ncols=3)
        >>>
        >>> # Show images with a title
        >>> learner.show_batch(sample_sz=2, title="Training Batch")
        """
        self.fit(
            1, callbacks=[SingleBatchForwardCallback()] + listify(callbacks)
        )
        show_images(self.xb[0][:sample_sz], **kwargs)
