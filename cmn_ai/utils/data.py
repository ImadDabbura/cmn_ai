"""
Data utilities for PyTorch-based machine learning workflows.

This module provides utilities for working with data in PyTorch-based machine learning
pipelines. It includes functions and classes for creating and managing DataLoaders,
device management for tensors and collections, data collation and preprocessing,
file system operations for data discovery, dataset splitting and labeling strategies,
and container classes for organizing data items.

The module is designed to work seamlessly with PyTorch's Dataset and DataLoader
classes, as well as Hugging Face datasets.

Functions
---------
get_dls : Create training and validation DataLoaders
to_device : Copy tensor(s) to specified device
to_cpu : Copy tensor(s) to CPU
collate_dict : Create collate function for Hugging Face Dataset dictionary
collate_device : Create collate function that moves batch to specified device
compose : Apply transformations in sequence to input
get_files : Get filenames in path with specified extensions
random_splitter : Randomly split items with specified probability
grandparent_splitter : Split items based on directory structure
split_by_func : Split items into train/valid lists using a function
parent_labeler : Label a file based on its parent directory
label_by_func : Label split data using a labeling function

Classes
-------
DataLoaders : Container for training and validation DataLoaders
ListContainer : Extended list with improved representation
ItemList : Base class for all types of datasets
SplitData : Split ItemList into train and validation data lists
LabeledData : Create labeled data with input and target ItemLists

Notes
-----
This module is part of the cmn_ai library and provides high-level abstractions
for common data processing tasks in machine learning workflows.

Examples
--------
>>> from cmn_ai.utils.data import get_files, to_device, compose, SplitData, LabeledData
>>> from cmn_ai.utils.data import ItemList, grandparent_splitter, parent_labeler
>>> # File operations
>>> files = get_files('./data', extensions=['.txt', '.csv'])
>>> # Device operations
>>> import torch
>>> tensor = torch.randn(3, 3)
>>> tensor_on_gpu = to_device(tensor, 'cuda')
>>> # Function composition
>>> def add_one(x): return x + 1
>>> def multiply_two(x): return x * 2
>>> result = compose(5, [add_one, multiply_two])  # Returns 12
>>> # Data splitting
>>> items = ItemList(['train/cat/1.jpg', 'valid/dog/2.jpg', 'train/cat/3.jpg'])
>>> split_data = SplitData.split_by_func(items, grandparent_splitter)
>>> print(f"Train: {len(split_data.train)}, Valid: {len(split_data.valid)}")
>>> # Labeled data
>>> x_items = ItemList(['image1.jpg', 'image2.jpg'])
>>> y_items = ItemList(['cat', 'dog'])
>>> labeled_data = LabeledData(x_items, y_items)
>>> x, y = labeled_data[0]  # Returns ('image1.jpg', 'cat')
>>> # Labeling with function
>>> items = ItemList(['data/cat/1.jpg', 'data/dog/2.jpg'])
>>> labeled = LabeledData.label_by_func(items, parent_labeler)
>>> print(labeled.y[0], labeled.y[1])  # 'cat' 'dog'
"""

from __future__ import annotations

import os
from collections import UserList
from collections.abc import Iterable, Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from datasets.dataset_dict import DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate

from .processors import Processor
from .utils import listify

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dls(
    train_ds: Dataset, valid_ds: Dataset, batch_size: int, **kwargs
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Creates two DataLoaders: one for training and one for validation. The
    validation DataLoader has twice the batch size and doesn't shuffle data.

    Parameters
    ----------
    train_ds : Dataset
        Training dataset.
    valid_ds : Dataset
        Validation dataset.
    batch_size : int
        Batch size for the training DataLoader. Validation DataLoader will
        use batch_size * 2.
    **kwargs : dict
        Additional keyword arguments passed to DataLoader constructor.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        A tuple containing (train_dataloader, valid_dataloader).

    Examples
    --------
    >>> train_ds = MyDataset(train_data)
    >>> valid_ds = MyDataset(valid_data)
    >>> train_dl, valid_dl = get_dls(train_ds, valid_ds, batch_size=32)
    """
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(
            valid_ds, batch_size=batch_size * 2, shuffle=False, **kwargs
        ),
    )


def to_device(
    x: Tensor | Iterable[Tensor] | Mapping[str, Tensor],
    device: str | torch.device = DEFAULT_DEVICE,
) -> Tensor | Iterable[Tensor] | Mapping[str, Tensor]:
    """
    Copy tensor(s) to specified device.

    Recursively moves tensors and collections of tensors to the specified device.
    If a tensor is already on the target device, returns the tensor itself.

    Parameters
    ----------
    x : Tensor or Iterable[Tensor] or Mapping[str, Tensor]
        Tensor or collection of tensors to move to device.
    device : str or torch.device, default='cuda' if available else 'cpu'
        Device to copy the tensor(s) to.

    Returns
    -------
    Tensor or Iterable[Tensor] or Mapping[str, Tensor]
        Copied tensor(s) on the specified device.

    Notes
    -----
    This function may fail if iterables contain non-tensor objects that
    don't have a `.to()` method.

    Examples
    --------
    >>> tensor = torch.randn(3, 3)
    >>> tensor_on_gpu = to_device(tensor, 'cuda')
    >>> batch = {'input': tensor, 'target': tensor}
    >>> batch_on_gpu = to_device(batch, 'cuda')
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return type(x)(to_device(o, device) for o in x)
    return x


def to_cpu(
    x: Tensor | Iterable[Tensor] | Mapping[str, Tensor],
) -> Tensor | Iterable[Tensor] | Mapping[str, Tensor]:
    """
    Copy tensor(s) to CPU.

    If a tensor is already on the CPU, returns the tensor itself; otherwise,
    returns a copy of the tensor on CPU.

    Parameters
    ----------
    x : Tensor or Iterable[Tensor] or Mapping[str, Tensor]
        Tensor or collection of tensors to move to CPU.

    Returns
    -------
    Tensor or Iterable[Tensor] or Mapping[str, Tensor]
        Copied tensor(s) on CPU.

    Examples
    --------
    >>> tensor = torch.randn(3, 3, device='cuda')
    >>> tensor_on_cpu = to_cpu(tensor)
    """
    return to_device(x, torch.device("cpu"))


def collate_dict(*keys: str) -> Callable:
    """
    Create a collate function for a Dataset dictionary.

    Creates a collate function that extracts specified keys from a batch
    and applies PyTorch's default collate function.

    Parameters
    ----------
    *keys : str
        Keys to extract from the batch dictionary.

    Returns
    -------
    callable
        Collate function that returns tuple of collated inputs.

    Examples
    --------
    >>> collate_fn = collate_dict('input_ids', 'attention_mask', 'labels')
    >>> dataloader = DataLoader(dataset, collate_fn=collate_fn)
    """
    get = itemgetter(*keys)

    def _f(batch):
        # default_collate(batch) -> dictionary where values are stacked
        # tensors
        # get() would return the tuple of stacked tensors instead of
        # dictionary
        return get(default_collate(batch))

    return _f


def collate_device(device: str | torch.device) -> Callable:
    """
    Create a collate function that moves batch to specified device.

    Parameters
    ----------
    device : str or torch.device
        Device to copy batch to.

    Returns
    -------
    callable
        Collate function that returns batch on specified device.

    Examples
    --------
    >>> collate_fn = collate_device('cuda')
    >>> dataloader = DataLoader(dataset, collate_fn=collate_fn)
    """

    def _f(batch):
        return to_device(default_collate(batch), device)

    return _f


class DataLoaders:
    """
    Container for training and validation DataLoaders.

    A convenience class that holds training and validation DataLoaders
    and provides easy access to them.

    Attributes
    ----------
    train : DataLoader
        Training DataLoader.
    valid : DataLoader
        Validation DataLoader.

    Examples
    --------
    >>> train_dl = DataLoader(train_ds, batch_size=32)
    >>> valid_dl = DataLoader(valid_ds, batch_size=32)
    >>> dls = DataLoaders(train_dl, valid_dl)
    >>> dls.train  # Access training DataLoader
    >>> dls.valid  # Access validation DataLoader
    """

    def __init__(self, *dls: DataLoader) -> None:
        """
        Initialize DataLoaders with training and validation DataLoaders.

        Parameters
        ----------
        *dls : DataLoader
            List of DataLoaders. First is assumed to be training,
            second is assumed to be validation.
        """
        # Assuming first dataloader is training dataloader and the second is
        # valid dataloader
        self.train, self.valid = dls[:2]

    @classmethod
    def from_dd(
        cls, dd: DatasetDict, batch_size: int = 32, **kwargs
    ) -> DataLoaders:
        """
        Create DataLoaders from Hugging Face Dataset dictionary.

        Parameters
        ----------
        dd : DatasetDict
            Hugging Face Dataset dictionary. Must have at least two datasets:
            train and valid/test datasets.
        batch_size : int, default=32
            Batch size passed to DataLoader.
        **kwargs : dict
            Additional keyword arguments passed to DataLoader.

        Returns
        -------
        DataLoaders
            DataLoaders instance with train and validation DataLoaders.

        Examples
        --------
        >>> from datasets import DatasetDict
        >>> dd = DatasetDict({'train': train_ds, 'validation': valid_ds})
        >>> dls = DataLoaders.from_dd(dd, batch_size=32)
        """
        return cls(
            *get_dls(
                *dd.values(),
                batch_size=batch_size,
                collate_fn=collate_dict(*dd["train"].features),
                **kwargs,
            )
        )


def compose(
    x: Any, funcs: Callable, *args, order: str = "order", **kwargs
) -> Any:
    """
    Apply transformations in sequence to input.

    Applies transformations in `funcs` to the input `x` in the specified order.
    Functions are sorted by their `order` attribute if present.

    Parameters
    ----------
    x : Any
        Input to transform.
    funcs : callable or iterable of callables
        Function(s) to apply to input.
    *args : tuple
        Positional arguments passed to each function.
    order : str, default='order'
        Attribute name used to determine function order.
    **kwargs : dict
        Keyword arguments passed to each function.

    Returns
    -------
    Any
        Transformed input.

    Examples
    --------
    >>> def add_one(x): return x + 1
    >>> def multiply_two(x): return x * 2
    >>> result = compose(5, [add_one, multiply_two])  # Returns 12
    """
    for func in sorted(listify(funcs), key=lambda o: getattr(o, order, 0)):
        x = func(x, *args, **kwargs)
    return x


def _get_files(
    path: Path, fs: list[str], extensions: Iterable | None = None
) -> list[Path]:
    """
    Get filenames in path that have specified extensions.

    Parameters
    ----------
    path : Path
        Directory path to search.
    fs : list of str
        List of filenames to filter.
    extensions : Iterable, optional
        Set of file extensions to include (without dot).

    Returns
    -------
    list of Path
        Filtered list of file paths.
    """
    path = Path(path)
    res = [
        path / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path: str | Path,
    extensions: str | Iterable[str] | None = None,
    include: Iterable[str] | None = None,
    recurse: bool = False,
) -> list[Path]:
    """
    Get filenames in path with specified extensions.

    Get filenames in `path` that have extension `extensions` starting
    with `path` and optionally recurse to subdirectories.

    Parameters
    ----------
    path : str or Path
        Path for the root directory to search for files.
    extensions : str or iterable of str, optional
        Suffixes of filenames to look for (with or without dot).
    include : iterable of str, optional
        Top-level directory(ies) under `path` to use for searching files.
    recurse : bool, default=False
        Whether to search subdirectories recursively.

    Returns
    -------
    list of Path
        List of file paths that end with specified extensions under `path`.

    Examples
    --------
    >>> files = get_files('./data', extensions=['.jpg', '.png'])
    >>> files = get_files('./data', extensions='.txt', recurse=True)
    """
    path = Path(path)
    extensions = {e.lower() for e in listify(extensions)}
    if recurse:
        res = []
        for i, (p, d, fs) in enumerate(os.walk(path)):
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            res += _get_files(p, fs, extensions)
        return res
    else:
        fs = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, fs, extensions)


class ListContainer(UserList):
    """
    Extended list with improved representation.

    Extends builtin list by changing the creation of the list from the given
    items and changing repr to return first 10 items along with total number of
    items and the class name. This will be the base class where other
    containers will inherit from.

    Attributes
    ----------
    data : list
        The underlying list data.

    Examples
    --------
    >>> container = ListContainer([1, 2, 3, 4, 5])
    >>> print(container)
    ListContainer: (5 items)
    [1, 2, 3, 4, 5]
    """

    def __init__(self, items: Any) -> None:
        """
        Initialize ListContainer with items.

        Parameters
        ----------
        items : Any
            Items to create list from.
        """
        self.data = listify(items)

    def __repr__(self) -> str:
        cls_nm = self.__class__.__name__
        res = f"{cls_nm}: ({len(self.data) :,} items)\n{self.data[:10]}"  # noqa: E231
        if len(self) > 10:
            res = res[:-1] + ", ...]"
        return res


class ItemList(ListContainer):
    """
    Base class for all types of datasets such as image, text, etc.

    A container class that holds items and provides functionality for
    applying transformations and retrieving items with transformations
    applied.

    Attributes
    ----------
    path : Path
        Path of the items that were used to create the list.
    tfms : callable, optional
        Transformations to apply on items before returning them.

    Examples
    --------
    >>> items = ItemList(['file1.jpg', 'file2.jpg'], path='./data')
    >>> items[0]  # Returns transformed item
    """

    def __init__(
        self,
        items: Sequence,
        path: str | Path = ".",
        tfms: Callable | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize ItemList with items, path, and transformations.

        Parameters
        ----------
        items : Sequence
            Items to create list.
        path : str or Path, default="."
            Path of the items that were used to create the list.
        tfms : callable, optional
            Transformations to apply on items before returning them.
        **kwargs : dict
            Additional attributes to set on the instance.
        """
        super().__init__(items)
        self.path = Path(path)
        self.tfms = tfms
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:
        return super().__repr__() + f"\nPath: {self.path.resolve()}"

    def new(self, items: Sequence, cls: ItemList | None = None) -> ItemList:
        """
        Create a new instance of the ItemList with items.

        Parameters
        ----------
        items : Sequence
            The items to create the list from.
        cls : ItemList, optional
            The class to instantiate. If None, the same class will be used.

        Returns
        -------
        ItemList
            The new instance of the ItemList.
        """
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, self.tfms)

    def get(self, item: Any) -> Any:
        """
        Get item without transformations.

        Every class that inherits from `ItemList` has to override this
        method to provide custom item retrieval logic.

        Parameters
        ----------
        item : Any
            Item to retrieve.

        Returns
        -------
        Any
            Retrieved item.
        """
        return item

    def _get(self, item: Any) -> Any:
        """
        Get item with transformations applied.

        Parameters
        ----------
        item : Any
            Item to retrieve.

        Returns
        -------
        Any
            Item with transformations applied.
        """
        return compose(self.get(item), self.tfms)

    def __getitem__(self, idx: int | slice) -> Any | list[Any]:
        """
        Get item(s) with transformations applied.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        Any or list of Any
            Item(s) with transformations applied.
        """
        items = super().__getitem__(idx)
        if isinstance(items, UserList):
            return [self._get(item) for item in items]
        return self._get(items)


def random_splitter(f_name: str, p_valid: float = 0.2) -> bool:
    """
    Randomly split items with specified probability.

    Randomly split items with `p_valid` probability to be in the
    validation set.

    Parameters
    ----------
    f_name : str
        Item's filename. Not used here, but left for API consistency
        with other splitters.
    p_valid : float, default=0.2
        Probability of the item to be in the validation set.

    Returns
    -------
    bool
        True if the item is in validation else False (training).

    Examples
    --------
    >>> splitter = lambda f: random_splitter(f, p_valid=0.3)
    >>> is_valid = splitter('file.jpg')  # Returns True or False
    """
    return np.random.random() < p_valid


def grandparent_splitter(
    f_name: str | Path,
    valid_name: str = "valid",
    train_name: str = "train",
) -> bool | None:
    """
    Split items based on directory structure.

    Split items based on whether they fall under validation or training
    directories. This assumes that the directory structure is
    train/label/items or valid/label/items.

    Parameters
    ----------
    f_name : str or Path
        Item's filename.
    valid_name : str, default="valid"
        Name of the directory that holds the validation items.
    train_name : str, default="train"
        Name of the directory that holds the training items.

    Returns
    -------
    bool or None
        True if the item is in validation, False if in training,
        None if neither.

    Examples
    --------
    >>> splitter = lambda f: grandparent_splitter(f, 'val', 'train')
    >>> is_valid = splitter('train/cat/image.jpg')  # Returns False
    >>> is_valid = splitter('val/dog/image.jpg')    # Returns True
    """
    gp = Path(f_name).parent.parent.name
    if gp == valid_name:
        return True
    if gp == train_name:
        return False
    return None


def split_by_func(items: Iterable, func: Callable) -> tuple[list, list]:
    """
    Split items into train/valid lists using a function.

    Parameters
    ----------
    items : Iterable
        Items to be split into train/valid.
    func : callable
        Split function to split items. Should return True for validation
        items, False for training items, and None to exclude items.

    Returns
    -------
    tuple[list, list]
        Train and valid item lists.

    Examples
    --------
    >>> files = ['train/cat/1.jpg', 'val/dog/2.jpg', 'train/cat/3.jpg']
    >>> train, valid = split_by_func(files, grandparent_splitter)
    """
    mask = [func(o) for o in items]
    # `None` values will be filtered out
    val = [o for o, m in zip(items, mask) if m]
    train = [o for o, m in zip(items, mask) if m is False]
    return train, val


class SplitData:
    """
    Split ItemList into train and validation data lists.

    A container class that holds training and validation ItemLists
    and provides functionality for creating DataLoaders from them.

    Attributes
    ----------
    train : ItemList
        Training items.
    valid : ItemList
        Validation items.

    Examples
    --------
    >>> train_items = ItemList(['train1.jpg', 'train2.jpg'])
    >>> valid_items = ItemList(['valid1.jpg', 'valid2.jpg'])
    >>> split_data = SplitData(train_items, valid_items)
    >>> train_dl, valid_dl = split_data.to_dls(batch_size=32)
    """

    def __init__(self, train: ItemList, valid: ItemList) -> None:
        """
        Initialize SplitData with training and validation ItemLists.

        Parameters
        ----------
        train : ItemList
            Training items.
        valid : ItemList
            Validation items.
        """
        self.train = train
        self.valid = valid

    def __getattr__(self, k: str) -> Any:
        """Delegate attribute access to training ItemList."""
        return getattr(self.train, k)

    # This is needed if we want to pickle SplitData objects and be able
    # to load it back without recursion errors
    def __setstate__(self, data: dict) -> None:
        """Set state for pickling."""
        self.__dict__.update(data)

    @classmethod
    def split_by_func(
        cls, item_list: ItemList, split_func: Callable
    ) -> SplitData:
        """
        Split ItemList by splitter function.

        Parameters
        ----------
        item_list : ItemList
            ItemList to split.
        split_func : callable
            Function to use for splitting items.

        Returns
        -------
        SplitData
            SplitData object with train and validation ItemLists.

        Examples
        --------
        >>> items = ItemList(['train/cat/1.jpg', 'val/dog/2.jpg'])
        >>> split_data = SplitData.split_by_func(items, grandparent_splitter)
        """
        # We need to use `data` attribute to get the item list so when
        # we index into it we get the element itself, not the
        # transformed element.
        train_files, val_files = split_by_func(item_list.data, split_func)
        # Because files would be of type list, change type to its
        # original type with the original path/transforms
        train_list, val_list = map(item_list.new, (train_files, val_files))
        return cls(train_list, val_list)

    def to_dls(
        self, batch_size: int = 32, **kwargs
    ) -> tuple[DataLoader, DataLoader]:
        """
        Create training and validation DataLoaders.

        Parameters
        ----------
        batch_size : int, default=32
            Batch size for DataLoaders.
        **kwargs : dict
            Additional keyword arguments passed to DataLoader.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            Training and validation DataLoaders.
        """
        return get_dls(self.train, self.valid, batch_size, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}\n---------\nTrain - {self.train}\n\n"
            f"Valid - {self.valid}\n"
        )


class LabeledData:
    """
    Create labeled data with input and target ItemLists.

    A container class that holds input (x) and target (y) ItemLists
    and provides functionality for processing and retrieving labeled data.

    Attributes
    ----------
    x : ItemList
        Input items to the model.
    y : ItemList
        Label items.
    proc_x : Processor or iterable of Processor, optional
        Input items processor(s).
    proc_y : Processor or iterable of Processor, optional
        Label items processor(s).

    Examples
    --------
    >>> x_items = ItemList(['image1.jpg', 'image2.jpg'])
    >>> y_items = ItemList(['cat', 'dog'])
    >>> labeled_data = LabeledData(x_items, y_items)
    >>> x, y = labeled_data[0]  # Get first labeled example
    """

    def __init__(
        self,
        x: ItemList,
        y: ItemList,
        proc_x: Processor | Iterable[Processor] | None = None,
        proc_y: Processor | Iterable[Processor] | None = None,
    ) -> None:
        """
        Initialize LabeledData with input and target ItemLists.

        Parameters
        ----------
        x : ItemList
            Input items to the model.
        y : ItemList
            Label items.
        proc_x : Processor or iterable of Processor, optional
            Input items processor(s).
        proc_y : Processor or iterable of Processor, optional
            Label items processor(s).
        """
        self.x = self.process(x, proc_x)
        self.y = self.process(y, proc_y)
        self.proc_x = proc_x
        self.proc_y = proc_y

    def process(
        self,
        item_list: ItemList,
        proc: Processor | Iterable[Processor] | None,
    ) -> ItemList:
        """
        Apply processors to an ItemList.

        Parameters
        ----------
        item_list : ItemList
            The ItemList to process.
        proc : Processor or iterable of Processor, optional
            The processor or list of processors to apply.

        Returns
        -------
        ItemList
            The processed ItemList.
        """
        return item_list.new(compose(item_list.data, proc))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n"

    def __getitem__(
        self, idx: int | slice
    ) -> tuple[Any, Any] | list[tuple[Any, Any]]:
        """
        Get labeled example(s) at index.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        tuple or list of tuple
            Labeled example(s) as (x, y) pairs.
        """
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        """Return the number of labeled examples."""
        return len(self.x)

    def x_obj(self, idx: int) -> Any:
        """
        Get input object at index after deprocessing.

        Parameters
        ----------
        idx : int
            Index of the input object to retrieve.

        Returns
        -------
        Any
            The input object at index idx after applying all processors
            in proc_x.
        """
        return self._obj(self.x, idx, self.proc_x)

    def y_obj(self, idx: int) -> Any:
        """
        Get label object at index after deprocessing.

        Parameters
        ----------
        idx : int
            Index of the label object to retrieve.

        Returns
        -------
        Any
            The label object at index idx after applying all processors
            in proc_y.
        """
        return self._obj(self.y, idx, self.proc_y)

    def _obj(
        self,
        items: ItemList,
        idx: int,
        procs: Processor | Iterable[Processor] | None,
    ) -> Any:
        """
        Get object at index after deprocessing.

        Parameters
        ----------
        items : ItemList
            ItemList to get object from.
        idx : int
            Index of object to retrieve.
        procs : Processor or iterable of Processor, optional
            Processors to deprocess with.

        Returns
        -------
        Any
            Deprocessed object.
        """
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deprocess(item)
        return item

    @staticmethod
    def _label_by_func(
        ds: ItemList, label_func: Callable, cls: type = ItemList
    ) -> ItemList:
        """
        Create ItemList from dataset using labeling function.

        Parameters
        ----------
        ds : ItemList
            Dataset to label.
        label_func : callable
            Function to apply to each item for labeling.
        cls : type, default=ItemList
            Class to use for creating the labeled ItemList.

        Returns
        -------
        ItemList
            Labeled ItemList.
        """
        return cls([label_func(o) for o in ds.data], path=ds.path)

    @classmethod
    def label_by_func(
        cls,
        item_list: ItemList,
        label_func: Callable,
        proc_x: Callable | None = None,
        proc_y: Callable | None = None,
    ) -> LabeledData:
        """
        Label an ItemList using a labeling function.

        Parameters
        ----------
        item_list : ItemList
            The ItemList to be labeled.
        label_func : callable
            The function to be used for labeling.
        proc_x : callable, optional
            The processor to be applied to the input data.
        proc_y : callable, optional
            The processor to be applied to the label data.

        Returns
        -------
        LabeledData
            The labeled ItemList.

        Examples
        --------
        >>> items = ItemList(['image1.jpg', 'image2.jpg'])
        >>> labeled = LabeledData.label_by_func(items, parent_labeler)
        """
        return cls(
            item_list,
            LabeledData._label_by_func(item_list, label_func),
            proc_x=proc_x,
            proc_y=proc_y,
        )


def parent_labeler(f_name: str | Path) -> str:
    """
    Label a file based on its parent directory.

    Parameters
    ----------
    f_name : str or Path
        Filename to get the parent directory.

    Returns
    -------
    str
        Name of the parent directory.

    Examples
    --------
    >>> label = parent_labeler('data/cat/image.jpg')  # Returns 'cat'
    >>> label = parent_labeler('data/dog/image.png')  # Returns 'dog'
    """
    return Path(f_name).parent.name


def label_by_func(
    splitted_data: SplitData,
    label_func: Callable,
    proc_x: Callable | None = None,
    proc_y: Callable | None = None,
) -> SplitData:
    """
    Label split data using a labeling function.

    Parameters
    ----------
    splitted_data : SplitData
        The split data to be labeled.
    label_func : callable
        The function to be used for labeling.
    proc_x : callable, optional
        The processor to be applied to the input data.
    proc_y : callable, optional
        The processor to be applied to the label data.

    Returns
    -------
    SplitData
        The labeled split data.

    Examples
    --------
    >>> split_data = SplitData(train_items, valid_items)
    >>> labeled_split = label_by_func(split_data, parent_labeler)
    """
    train = LabeledData.label_by_func(
        splitted_data.train, label_func, proc_x=proc_x, proc_y=proc_y
    )
    valid = LabeledData.label_by_func(
        splitted_data.valid, label_func, proc_x=proc_x, proc_y=proc_y
    )
    return SplitData(train, valid)
