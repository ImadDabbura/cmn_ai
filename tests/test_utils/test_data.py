"""
Unit tests for cmn_ai.utils.data module.

Tests for data utilities including device management, file operations,
data splitting, and container classes.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from cmn_ai.utils.data import (
    ItemList,
    LabeledData,
    ListContainer,
    SplitData,
    compose,
    get_files,
    grandparent_splitter,
    parent_labeler,
    random_splitter,
    split_by_func,
    to_cpu,
    to_device,
)


class TestDeviceManagement:
    """Test device management functions."""

    def test_to_device_tensor(self):
        """Test to_device with a single tensor."""
        tensor = torch.randn(3, 3)
        result = to_device(tensor, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_to_device_list(self):
        """Test to_device with a list of tensors."""
        tensors = [torch.randn(2, 2), torch.randn(2, 2)]
        result = to_device(tensors, "cpu")
        assert isinstance(result, list)
        assert all(t.device.type == "cpu" for t in result)

    def test_to_device_dict(self):
        """Test to_device with a dictionary of tensors."""
        tensors = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
        result = to_device(tensors, "cpu")
        assert isinstance(result, dict)
        assert all(t.device.type == "cpu" for t in result.values())

    def test_to_device_already_on_device(self):
        """Test to_device when tensor is already on target device."""
        tensor = torch.randn(3, 3, device="cpu")
        result = to_device(tensor, "cpu")
        assert result is tensor  # Should return the same object

    def test_to_cpu(self):
        """Test to_cpu function."""
        if torch.cuda.is_available():
            tensor = torch.randn(3, 3, device="cuda")
            result = to_cpu(tensor)
            assert result.device.type == "cpu"
        else:
            tensor = torch.randn(3, 3)
            result = to_cpu(tensor)
            assert result.device.type == "cpu"


class TestCompose:
    """Test function composition."""

    def test_compose_single_function(self):
        """Test compose with a single function."""

        def add_one(x):
            return x + 1

        result = compose(5, add_one)
        assert result == 6

    def test_compose_multiple_functions(self):
        """Test compose with multiple functions."""

        def add_one(x):
            return x + 1

        def multiply_two(x):
            return x * 2

        result = compose(5, [add_one, multiply_two])
        assert result == 12

    def test_compose_with_order_attribute(self):
        """Test compose with functions having order attribute."""

        def func1(x):
            return x + 1

        func1.order = 2

        def func2(x):
            return x * 2

        func2.order = 1

        result = compose(5, [func1, func2])
        assert result == 11  # func2 (order=1) should be applied first

    def test_compose_with_args_kwargs(self):
        """Test compose with additional arguments."""

        def add_value(x, value):
            return x + value

        result = compose(5, add_value, value=3)
        assert result == 8


class TestFileOperations:
    """Test file operations."""

    def test_get_files_basic(self):
        """Test get_files with basic functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            temp_path = Path(temp_dir)
            (temp_path / "test1.txt").touch()
            (temp_path / "test2.txt").touch()
            (temp_path / "test3.csv").touch()

            # Test with .txt extension
            files = get_files(temp_dir, extensions=[".txt"])
            assert len(files) == 2
            assert all(f.suffix == ".txt" for f in files)

    def test_get_files_recursive(self):
        """Test get_files with recursive search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test1.txt").touch()
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "test2.txt").touch()

            files = get_files(temp_dir, extensions=[".txt"], recurse=True)
            assert len(files) == 2

    def test_get_files_no_recursion(self):
        """Test get_files without recursion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test1.txt").touch()
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "test2.txt").touch()

            files = get_files(temp_dir, extensions=[".txt"], recurse=False)
            assert len(files) == 1

    def test_get_files_with_include(self):
        """Test get_files with include parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "dir1").mkdir()
            (temp_path / "dir2").mkdir()
            (temp_path / "dir1" / "test1.txt").touch()
            (temp_path / "dir2" / "test2.txt").touch()

            files = get_files(
                temp_dir, extensions=[".txt"], include=["dir1"], recurse=True
            )
            assert len(files) == 1
            assert "dir1" in str(files[0])


class TestSplitters:
    """Test data splitting functions."""

    def test_random_splitter(self):
        """Test random_splitter function."""
        # Test with p_valid = 0.0 (should always return False)
        with patch("numpy.random.random", return_value=0.5):
            result = random_splitter("test.txt", p_valid=0.0)
            assert result is False

        # Test with p_valid = 1.0 (should always return True)
        with patch("numpy.random.random", return_value=0.5):
            result = random_splitter("test.txt", p_valid=1.0)
            assert result is True

    def test_grandparent_splitter_valid(self):
        """Test grandparent_splitter with valid directory."""
        result = grandparent_splitter("train/cat/image.jpg")
        assert result is False

        result = grandparent_splitter("valid/dog/image.jpg")
        assert result is True

    def test_grandparent_splitter_custom_names(self):
        """Test grandparent_splitter with custom directory names."""
        result = grandparent_splitter(
            "train_data/cat/image.jpg", "val", "train_data"
        )
        assert result is False

        result = grandparent_splitter("val/dog/image.jpg", "val", "train_data")
        assert result is True

    def test_grandparent_splitter_neither(self):
        """Test grandparent_splitter with neither train nor valid."""
        result = grandparent_splitter("other/cat/image.jpg")
        assert result is None

    def test_split_by_func(self):
        """Test split_by_func function."""
        items = ["train/cat/1.jpg", "valid/dog/2.jpg", "train/cat/3.jpg"]
        train, valid = split_by_func(items, grandparent_splitter)

        assert len(train) == 2
        assert len(valid) == 1
        assert all("train" in item for item in train)
        assert all("valid" in item for item in valid)

    def test_parent_labeler(self):
        """Test parent_labeler function."""
        result = parent_labeler("data/cat/image.jpg")
        assert result == "cat"

        result = parent_labeler("data/dog/image.png")
        assert result == "dog"


class TestListContainer:
    """Test ListContainer class."""

    def test_init_with_list(self):
        """Test ListContainer initialization with list."""
        items = [1, 2, 3, 4, 5]
        container = ListContainer(items)
        assert len(container) == 5
        assert container.data == items

    def test_init_with_single_item(self):
        """Test ListContainer initialization with single item."""
        container = ListContainer(42)
        assert len(container) == 1
        assert container.data == [42]

    def test_repr_short_list(self):
        """Test ListContainer repr with short list."""
        container = ListContainer([1, 2, 3])
        repr_str = repr(container)
        assert "ListContainer: (3 items)" in repr_str
        assert "[1, 2, 3]" in repr_str

    def test_repr_long_list(self):
        """Test ListContainer repr with long list."""
        container = ListContainer(list(range(15)))
        repr_str = repr(container)
        assert "ListContainer: (15 items)" in repr_str
        assert "..." in repr_str  # Should truncate


class TestItemList:
    """Test ItemList class."""

    def test_init(self):
        """Test ItemList initialization."""
        items = ["file1.txt", "file2.txt"]
        item_list = ItemList(items, path="./data")

        assert len(item_list) == 2
        assert item_list.path == Path("./data")
        assert item_list.tfms is None

    def test_init_with_transforms(self):
        """Test ItemList initialization with transforms."""

        def transform(x):
            return x.upper()

        items = ["file1.txt", "file2.txt"]
        item_list = ItemList(items, path="./data", tfms=transform)

        assert item_list.tfms == transform

    def test_new(self):
        """Test ItemList new method."""
        items = ["file1.txt", "file2.txt"]
        item_list = ItemList(items, path="./data")

        new_items = ["file3.txt", "file4.txt"]
        new_list = item_list.new(new_items)

        assert isinstance(new_list, ItemList)
        assert new_list.data == new_items
        assert new_list.path == item_list.path
        assert new_list.tfms == item_list.tfms

    def test_get(self):
        """Test ItemList get method."""
        items = ["file1.txt", "file2.txt"]
        item_list = ItemList(items, path="./data")

        result = item_list.get("test.txt")
        assert (
            result == "test.txt"
        )  # Default implementation returns item as-is

    def test_getitem_with_transforms(self):
        """Test ItemList __getitem__ with transforms."""

        def transform(x):
            return x.upper()

        items = ["file1.txt", "file2.txt"]
        item_list = ItemList(items, path="./data", tfms=transform)

        result = item_list[0]
        assert result == "FILE1.TXT"

    def test_repr(self):
        """Test ItemList repr method."""
        items = ["file1.txt", "file2.txt"]
        item_list = ItemList(items, path="./data")

        repr_str = repr(item_list)
        assert "ItemList" in repr_str
        assert "Path:" in repr_str


class TestSplitData:
    """Test SplitData class."""

    def test_init(self):
        """Test SplitData initialization."""
        train_items = ItemList(["train1.txt", "train2.txt"])
        valid_items = ItemList(["valid1.txt", "valid2.txt"])

        split_data = SplitData(train_items, valid_items)

        assert split_data.train == train_items
        assert split_data.valid == valid_items

    def test_getattr_delegation(self):
        """Test SplitData attribute delegation to train."""
        train_items = ItemList(["train1.txt", "train2.txt"])
        valid_items = ItemList(["valid1.txt", "valid2.txt"])

        split_data = SplitData(train_items, valid_items)

        # Should delegate to train ItemList
        assert len(split_data.train) == 2
        assert split_data.path == train_items.path

    @classmethod
    def test_split_by_func(cls):
        """Test SplitData split_by_func classmethod."""
        items = ItemList(["train/cat/1.jpg", "valid/dog/2.jpg"])

        split_data = SplitData.split_by_func(items, grandparent_splitter)

        assert isinstance(split_data, SplitData)
        assert len(split_data.train) == 1
        assert len(split_data.valid) == 1

    def test_repr(self):
        """Test SplitData repr method."""
        train_items = ItemList(["train1.txt", "train2.txt"])
        valid_items = ItemList(["valid1.txt", "valid2.txt"])

        split_data = SplitData(train_items, valid_items)
        repr_str = repr(split_data)

        assert "SplitData" in repr_str
        assert "Train" in repr_str
        assert "Valid" in repr_str


class TestLabeledData:
    """Test LabeledData class."""

    def test_init(self):
        """Test LabeledData initialization."""
        x_items = ItemList(["image1.jpg", "image2.jpg"])
        y_items = ItemList(["cat", "dog"])

        labeled_data = LabeledData(x_items, y_items)

        assert labeled_data.x == x_items
        assert labeled_data.y == y_items
        assert labeled_data.proc_x is None
        assert labeled_data.proc_y is None

    def test_getitem(self):
        """Test LabeledData __getitem__ method."""
        x_items = ItemList(["image1.jpg", "image2.jpg"])
        y_items = ItemList(["cat", "dog"])

        labeled_data = LabeledData(x_items, y_items)

        x, y = labeled_data[0]
        assert x == "image1.jpg"
        assert y == "cat"

    def test_len(self):
        """Test LabeledData __len__ method."""
        x_items = ItemList(["image1.jpg", "image2.jpg"])
        y_items = ItemList(["cat", "dog"])

        labeled_data = LabeledData(x_items, y_items)

        assert len(labeled_data) == 2

    def test_repr(self):
        """Test LabeledData repr method."""
        x_items = ItemList(["image1.jpg", "image2.jpg"])
        y_items = ItemList(["cat", "dog"])

        labeled_data = LabeledData(x_items, y_items)
        repr_str = repr(labeled_data)

        assert "LabeledData" in repr_str
        assert "x:" in repr_str
        assert "y:" in repr_str

    @classmethod
    def test_label_by_func(cls):
        """Test LabeledData label_by_func classmethod."""
        items = ItemList(["image1.jpg", "image2.jpg"])

        def label_func(path):
            return "cat" if "1" in str(path) else "dog"

        labeled_data = LabeledData.label_by_func(items, label_func)

        assert isinstance(labeled_data, LabeledData)
        assert len(labeled_data) == 2
        assert labeled_data.y[0] == "cat"
        assert labeled_data.y[1] == "dog"


if __name__ == "__main__":
    pytest.main([__file__])
