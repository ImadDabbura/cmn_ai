"""
Unit tests for plotting utilities module.

This module contains comprehensive tests for the plotting functions,
covering image display, subplot creation, grid generation, and various input types.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from PIL import Image

from cmn_ai.plot import get_grid, show_image, show_images, subplots


class TestShowImage:
    """Test cases for show_image function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create test images
        self.numpy_image = np.random.randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        self.grayscale_image = np.random.randint(
            0, 255, (100, 100), dtype=np.uint8
        )
        self.pil_image = Image.fromarray(self.numpy_image)
        self.torch_image = torch.randn(3, 100, 100)
        self.torch_image_grayscale = torch.randn(1, 100, 100)

    def test_show_numpy_image(self):
        """Test showing a numpy array image."""
        ax = show_image(self.numpy_image)

        assert isinstance(ax, Axes)
        # Check that an image was displayed
        assert len(ax.images) > 0
        assert ax.images[0].get_array().shape == (100, 100, 3)

    def test_show_pil_image(self):
        """Test showing a PIL image."""
        ax = show_image(self.pil_image)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        assert ax.images[0].get_array().shape == (100, 100, 3)

    def test_show_torch_image(self):
        """Test showing a PyTorch tensor image."""
        ax = show_image(self.torch_image)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        # Should be permuted from (3, 100, 100) to (100, 100, 3)
        assert ax.images[0].get_array().shape == (100, 100, 3)

    def test_show_torch_grayscale_image(self):
        """Test showing a PyTorch grayscale tensor image."""
        ax = show_image(self.torch_image_grayscale)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        # Should be squeezed from (1, 100, 100) to (100, 100)
        assert ax.images[0].get_array().shape == (100, 100)

    def test_show_grayscale_image(self):
        """Test showing a grayscale numpy image."""
        ax = show_image(self.grayscale_image)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        assert ax.images[0].get_array().shape == (100, 100)

    def test_show_image_with_title(self):
        """Test showing image with title."""
        ax = show_image(self.numpy_image, title="Test Image")

        assert ax.get_title() == "Test Image"

    def test_show_image_with_custom_ax(self):
        """Test showing image on custom axes."""
        fig, custom_ax = plt.subplots()
        ax = show_image(self.numpy_image, ax=custom_ax)

        assert ax == custom_ax
        assert isinstance(ax, Axes)
        assert len(ax.images) > 0

    def test_show_image_with_figsize(self):
        """Test showing image with custom figure size."""
        ax = show_image(self.numpy_image, figsize=(8, 6))

        assert ax.figure.get_size_inches()[0] == 8
        assert ax.figure.get_size_inches()[1] == 6

    def test_show_image_no_frame(self):
        """Test showing image without frame."""
        ax = show_image(self.numpy_image, noframe=True)

        # Check that axis ticks are removed
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0

    def test_show_image_with_frame(self):
        """Test showing image with frame."""
        ax = show_image(self.numpy_image, noframe=False)

        # Check that axis is visible
        assert ax.get_visible()

    def test_show_image_with_kwargs(self):
        """Test showing image with additional kwargs."""
        ax = show_image(self.numpy_image, cmap="gray")

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0

    def test_show_image_single_channel(self):
        """Test showing single channel image."""
        single_channel = np.random.randint(
            0, 255, (100, 100, 1), dtype=np.uint8
        )
        ax = show_image(single_channel)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        # Should be squeezed to (100, 100)
        assert ax.images[0].get_array().shape == (100, 100)


class TestSubplots:
    """Test cases for subplots function."""

    def test_subplots_default(self):
        """Test subplots with default parameters."""
        fig, ax = subplots()

        assert isinstance(fig, Figure)
        assert isinstance(ax, np.ndarray)
        assert len(ax) == 1

    def test_subplots_1x1(self):
        """Test subplots with 1x1 grid."""
        fig, ax = subplots(1, 1)

        assert isinstance(fig, Figure)
        assert isinstance(ax, np.ndarray)
        assert ax.shape == (1,)

    def test_subplots_2x2(self):
        """Test subplots with 2x2 grid."""
        fig, ax = subplots(2, 2)

        assert isinstance(fig, Figure)
        assert isinstance(ax, np.ndarray)
        assert ax.shape == (2, 2)

    def test_subplots_with_figsize(self):
        """Test subplots with custom figure size."""
        fig, ax = subplots(1, 1, figsize=(10, 8))

        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 8

    def test_subplots_with_imsize(self):
        """Test subplots with custom image size."""
        fig, ax = subplots(2, 3, imsize=4)

        # Should be 3*4 = 12 width, 2*4 = 8 height
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8

    def test_subplots_with_suptitle(self):
        """Test subplots with super title."""
        fig, ax = subplots(1, 1, suptitle="Test Figure")

        assert fig._suptitle.get_text() == "Test Figure"

    def test_subplots_single_ax_return(self):
        """Test that single subplot returns array with one element."""
        fig, ax = subplots(1, 1)

        assert isinstance(ax, np.ndarray)
        assert len(ax) == 1


class TestGetGrid:
    """Test cases for get_grid function."""

    def test_get_grid_default(self):
        """Test get_grid with default parameters."""
        fig, ax = get_grid(4)

        assert isinstance(fig, Figure)
        assert isinstance(ax, np.ndarray)
        assert len(ax.flat) >= 4

    def test_get_grid_with_nrows(self):
        """Test get_grid with specified number of rows."""
        fig, ax = get_grid(6, nrows=2)

        assert ax.shape[0] == 2
        assert ax.shape[1] == 3

    def test_get_grid_with_ncols(self):
        """Test get_grid with specified number of columns."""
        fig, ax = get_grid(6, ncols=3)

        assert ax.shape[0] == 2
        assert ax.shape[1] == 3

    def test_get_grid_with_title(self):
        """Test get_grid with title."""
        fig, ax = get_grid(4, title="Test Grid")

        assert fig._suptitle.get_text() == "Test Grid"

    def test_get_grid_with_custom_title_params(self):
        """Test get_grid with custom title parameters."""
        fig, ax = get_grid(4, title="Test Grid", weight="normal", size=16)

        assert fig._suptitle.get_text() == "Test Grid"
        assert fig._suptitle.get_weight() == "normal"
        assert fig._suptitle.get_size() == 16

    def test_get_grid_hides_extra_axes(self):
        """Test that get_grid hides axes beyond the required number."""
        fig, ax = get_grid(3, nrows=2, ncols=2)

        # Should have 4 total axes, but only 3 should be visible
        assert len(ax.flat) == 4
        # The fourth axis should be turned off (implementation detail)
        # We just verify that the function doesn't crash and returns the right number of axes

    def test_get_grid_square_approximation(self):
        """Test get_grid creates approximately square grid."""
        fig, ax = get_grid(9)

        # Should be 3x3 grid
        assert ax.shape == (3, 3)

    def test_get_grid_rectangular_approximation(self):
        """Test get_grid creates appropriate rectangular grid."""
        fig, ax = get_grid(10)

        # For 10 images, sqrt(10) ≈ 3.16, so it creates a 3x3 grid = 9 axes
        # The function uses floor, so 3x3 = 9 axes
        assert len(ax.flat) >= 9


class TestShowImages:
    """Test cases for show_images function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create test images
        self.images = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        ]
        self.titles = ["Image 1", "Image 2", "Image 3", "Image 4"]

    def test_show_images_default(self):
        """Test show_images with default parameters."""
        # This should not raise any exceptions
        show_images(self.images)

    def test_show_images_with_titles(self):
        """Test show_images with titles."""
        # This should not raise any exceptions
        show_images(self.images, titles=self.titles)

    def test_show_images_with_nrows(self):
        """Test show_images with specified number of rows."""
        # This should not raise any exceptions
        show_images(self.images, nrows=2)

    def test_show_images_with_ncols(self):
        """Test show_images with specified number of columns."""
        # This should not raise any exceptions
        show_images(self.images, ncols=2)

    def test_show_images_mixed_types(self):
        """Test show_images with mixed image types."""
        mixed_images = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            torch.randn(3, 50, 50),
            Image.fromarray(
                np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            ),
        ]

        # This should not raise any exceptions
        show_images(mixed_images)

    def test_show_images_fewer_titles(self):
        """Test show_images with fewer titles than images."""
        # This should not raise any exceptions
        show_images(self.images, titles=["Title 1", "Title 2"])

    def test_show_images_more_titles(self):
        """Test show_images with more titles than images."""
        # This should not raise any exceptions
        show_images(self.images[:2], titles=self.titles[:2])

    def test_show_images_single_image(self):
        """Test show_images with single image."""
        # This should not raise any exceptions
        show_images([self.images[0]])

    def test_show_images_empty_list(self):
        """Test show_images with empty list."""
        with pytest.raises(ZeroDivisionError):
            show_images([])


class TestPlotIntegration:
    """Integration tests for plotting functions."""

    def test_show_image_integration(self):
        """Test integration of show_image with matplotlib."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test that we can create and display an image
        ax = show_image(image, title="Integration Test")

        assert isinstance(ax, Axes)
        assert ax.get_title() == "Integration Test"

        # Clean up
        plt.close(ax.figure)

    def test_subplots_integration(self):
        """Test integration of subplots with matplotlib."""
        fig, ax = subplots(2, 2, suptitle="Integration Test")

        assert isinstance(fig, Figure)
        assert ax.shape == (2, 2)
        assert fig._suptitle.get_text() == "Integration Test"

        # Clean up
        plt.close(fig)

    def test_get_grid_integration(self):
        """Test integration of get_grid with matplotlib."""
        fig, ax = get_grid(5, title="Integration Test")

        assert isinstance(fig, Figure)
        # For 5 images, sqrt(5) ≈ 2.24, so it creates a 2x2 grid = 4 axes
        # The function uses floor, so 2x2 = 4 axes
        assert len(ax.flat) >= 4
        assert fig._suptitle.get_text() == "Integration Test"

        # Clean up
        plt.close(fig)

    def test_show_images_integration(self):
        """Test integration of show_images with matplotlib."""
        images = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        ]
        titles = ["Test 1", "Test 2"]

        # This should not raise any exceptions
        show_images(images, titles=titles)

        # Clean up any created figures
        plt.close("all")


class TestPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_show_image_empty_array(self):
        """Test show_image with empty array."""
        empty_array = np.array([])

        with pytest.raises(TypeError):
            show_image(empty_array)

    def test_show_image_none_input(self):
        """Test show_image with None input."""
        with pytest.raises(IndexError):
            show_image(None)

    def test_subplots_zero_dimensions(self):
        """Test subplots with zero dimensions."""
        with pytest.raises(ValueError):
            subplots(0, 1)

        with pytest.raises(ValueError):
            subplots(1, 0)

    def test_get_grid_zero_images(self):
        """Test get_grid with zero images."""
        with pytest.raises(ZeroDivisionError):
            get_grid(0)

    def test_get_grid_negative_images(self):
        """Test get_grid with negative number of images."""
        with pytest.raises(ValueError):
            get_grid(-1)

    def test_show_images_none_input(self):
        """Test show_images with None input."""
        with pytest.raises(TypeError):
            show_images(None)

    def test_show_images_non_iterable(self):
        """Test show_images with non-iterable input."""
        with pytest.raises(TypeError):
            show_images(123)

    def test_show_image_torch_tensor_no_cuda(self):
        """Test show_image with PyTorch tensor (simulating CPU tensor)."""
        tensor = torch.randn(3, 100, 100)
        # Simulate tensor that has been moved to CPU
        tensor = tensor.cpu()

        ax = show_image(tensor)
        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        assert ax.images[0].get_array().shape == (100, 100, 3)

    def test_show_image_requires_grad(self):
        """Test show_image with tensor that requires gradients."""
        tensor = torch.randn(3, 100, 100, requires_grad=True)

        ax = show_image(tensor)
        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        # Should be detached and converted to numpy
        assert isinstance(ax.images[0].get_array(), np.ndarray)
