"""
This module includes most of the utilities related to working with data
in general, and with `pytorch`'s related data in specific. It includes
functions from composing transforms and getting train/valid
`DataLoader`s to splitting and labeling `Dataset`s.
"""

from __future__ import annotations

import os
from collections import UserList
from collections.abc import Iterable, Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Mapping

import fastcore.all as fc
import numpy as np
import torch
from datasets.dataset_dict import DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate

from .processors import Processor
from .utils import listify

default_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dls(
    train_ds: Dataset, valid_ds: Dataset, batch_size: int, **kwargs
) -> tuple[DataLoader]:
    """
    Returns two dataloaders: 1 for training and 1 for validation. The
    validation dataloader has twice the batch size and doesn't shuffle
    data.
    """
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(
            valid_ds, batch_size=batch_size * 2, shuffle=False, **kwargs
        ),
    )


def to_device(
    x: Tensor | Iterable[Tensor] | Mapping[str, Tensor],
    device: str | torch.device = default_device,
):
    """
    Copy tensor(s) to device. If the tensor is already on the device,
    returns the tensor itself.

    Parameters
    ----------
    x : Tensor | Iterable[Tensor] | Mapping[str, Tensor]
        Tensor or collection of tensors to move to device.
    device : str | torch.device, default='cuda:0` if available else 'cpu'
        Device to copy the tensor to.

    Returns
    -------
    out : Tensor | Iterable[Tensor] | Mapping[str, Tensor]
        Copied tensor(s) on the `device`.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(to_device(o, device) for o in x)


def to_cpu(x: Tensor | Iterable[Tensor] | Mapping[str, Tensor]):
    """
    Copy tensor(s) to CPU. If a tensor is already on the CPU, returns
    the tensor itself; otherwise, returns a copy of the tensor.
    """
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype == torch.float16 else res


def collate_dict(ds: DatasetDict) -> Callable:
    """
    Collate inputs from HF Dataset dictionary and returns list of
    inputs after applying pytorch's default collate function.

    Parameters
    ----------
    ds : DatasetDict
        HF Dataset dictionary.

    Returns
    -------
    Callable
        Wrapper function that returns tuple of collated inputs.
    """
    get = itemgetter(*ds.features)

    def _f(batch):
        # default_collate(batch) -> dictionary where values are stacked
        # tensors
        # get() would return the tuple of stacked tensors instead of
        # dictionary
        return get(default_collate(batch))

    return _f


def collate_device(device: str | torch.device) -> Callable:
    """
    Collate inputs from batch and copy it to `device`.

    Parameters
    ----------
    device : str | torch.device
        Device to copy batch to.

    Returns
    -------
    Callable
        Wrapper function that returns tuple of collated inputs.
    """

    def _f(batch):
        return to_device(default_collate(batch), device)

    return _f


class DataLoaders:
    """Create train/valid DataLoaders."""

    def __init__(self, *dls) -> None:
        """
        Parameters
        ----------
        dls
            list of DataLoaders.
        """
        self.train, self.valid = dls[:2]

    @classmethod
    def from_dd(
        cls, dd: DatasetDict, batch_size: int, **kwargs
    ) -> DataLoaders:
        """
        Create train/valid data loaders from HF Dataset dictionary.

        Parameters
        ----------
        dd : DatasetDict
            HF Dataset dictionary.
        batch_size : int
            batch size passed to DataLoader.

        Returns
        -------
        tuple[DataLoader]
            Train/valid data loaders.
        """
        return cls(
            *get_dls(
                # TODO: dd may have other than train and valid datasets
                # and may also have only one train dataset
                # Either enforce at least two splits to be provided OR
                # restrict to the first two as train and valid OR make
                # the class/function more flexible
                *dd.values(),
                batch_size=batch_size,
                collate_fn=collate_dict(dd["train"]),
                **kwargs,
            )
        )


def compose(
    x: Any, funcs: Callable, *args, order: str = "order", **kwargs
) -> Any:
    """
    Applies transformations in `funcs` to the input `x` in  `order`.
    """
    for func in sorted(listify(funcs), key=lambda o: getattr(o, order, 0)):
        x = func(x, *args, **kwargs)
    return x


def _get_files(path, fs, extensions=None):
    """Get filenames in `path` that have extension `extensions`."""
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
    Get filenames in `path` that have extension `extensions` starting
    with `path` and optionally recurse to subdirectories.

    Parameters
    ----------
    path : str | Path
        Path for the root directory to search for files.
    extensions : str | Iterable[str] | None, default=None
        Suffixes of filenames to look for.
    include : Iterable[str] | None, default=None
        Top-level Director(y|ies) under `path` to use to search for files.
    recurse : bool, default=False
        Whether to search subdirectories recursively.

    Returns
    -------
    list[str]
        List of filenames that ends with `extensions` under `path`.
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
    Extend builtin list by changing the creation of the list from the given
    items and changing repr to return first 10 items along with total number of
    items and the class name. This will be the base class where other
    containers will inherit from.
    """

    def __init__(self, items) -> None:
        """
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
    """

    def __init__(
        self,
        items: Sequence,
        path: str | Path = ".",
        tfms: Callable | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        items : Sequence
            Items to create list.
        path : str | Path, default="."
            Path of the items that were used to create the list.
        tfms : Callable | None, default=None
            Transformations to apply on items before returning them.
        """
        super().__init__(items)
        self.path = Path(path)
        self.tfms = tfms
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:
        return super().__repr__() + f"\nPath: {self.path.resolve()}"

    def new(self, items: Sequence, cls: ItemList | None = None) -> ItemList:
        """
        Create a new instance of the `ItemList` with `items`.

        Parameters
        ----------
        items : Sequence
            The items to create the list from.
        cls : ItemList | None, default=None
            The class to instantiate. If None, the same class will be used.

        Returns
        -------
        ItemList
            The new instance of the `ItemList`.
        """
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, self.tfms)

    def get(self, item):
        """
        Every class that inherits from `ItemList` has to override this
        method.
        """
        return item

    def _get(self, item):
        """Returns items after applying all transforms `tfms`."""
        return compose(self.get(item), self.tfms)

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        if isinstance(items, UserList):
            return [self._get(item) for item in items]
        return self._get(items)


def random_splitter(f_name: str, p_valid: float = 0.2) -> bool:
    """
    Randomly split items with `p_valid` probability to be in the
    validation set.

    Parameters
    ----------
    f_name : str
        Item's filename. Not used here, but left for API consistency
        with other splitters.
    p_valid : float, optional
        Probability of the item to be in the validation set.

    Returns
    -------
    bool
        True if the item is in validation else False (training).
    """
    return np.random.random() < p_valid


def grandparent_splitter(
    f_name: str | Path, valid_name: str = "valid", train_name: str = "train"
) -> bool | None:
    """
    Split items based on whether they fall under validation or training
    directories. This assumes that the directory structure is
    train/label/items or valid/label/items.

    Parameters
    ----------
    f_name : str | Path
        Item's filename.
    valid_name : str, default="valid"
        Name of the directory that holds the validation items.
    train_name : str, default="train"
        Name of the directory that holds the training items.

    Returns
    -------
    bool | None
        True if the item is in validation else False (training).
        If neither, returns None.
    """
    gp = Path(f_name).parent.parent.name
    if gp == valid_name:
        return True
    if gp == train_name:
        return False
    return


def split_by_func(items: Iterable, func: Callable) -> tuple[list, list]:
    """
    Split items into train/valid lists using `func`.

    Parameters
    ----------
    items : Iterable
        Items to be split into train/valid.
    func : Callable
        Split function to split items.

    Returns
    -------
    tuple[list, list]
        Train and valid item lists.
    """
    mask = [func(o) for o in items]
    # `None` values will be filtered out
    val = [o for o, m in zip(items, mask) if m]
    train = [o for o, m in zip(items, mask) if m is False]
    return train, val


class SplitData:
    """
    Split Item list into train and validation data lists.
    """

    def __init__(self, train: ItemList, valid: ItemList) -> None:
        """
        Parameters
        ----------
        train : ItemList
            Training items.
        valid : ItemList
            Validation items.
        """
        self.train = train
        self.valid = valid

    def __getattr__(self, k):
        return getattr(self.train, k)

    # This is needed if we want to pickle SplitData objects and be able
    # to load it back without recursion errors
    def __setstate__(self, data):
        self.__dict__.update(data)

    @classmethod
    def split_by_func(
        cls, item_list: ItemList, split_func: Callable
    ) -> SplitData:
        """
        Split item list by splitter function and returns a SplitData
        object.
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
        Returns a tuple of training and validation DataLoaders using
        train and valid datasets.
        """
        return get_dls(self.train, self.valid, batch_size, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}\n---------\nTrain - {self.train}\n\n"
            f"Valid - {self.valid}\n"
        )


class LabeledData:
    """
    Create a labeled data and expose both x & y as item lists after
    passing them through all processors.
    """

    def __init__(
        self,
        x: ItemList,
        y: ItemList,
        proc_x: Processor | Iterable[Processor] | None = None,
        proc_y: Processor | Iterable[Processor] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        x : ItemList
            Input items to the model.
        y : ItemList
            Label items.
        proc_x : Processor | Iterable[Processor] | None, default=None
            Input items processor(s).
        proc_y : Processor | Iterable[Processor] | None, default=None
            Label items processor(s).
        """
        self.x = self.process(x, proc_x)
        self.y = self.process(y, proc_y)
        self.proc_x = proc_x
        self.proc_y = proc_y

    def process(self, item_list, proc):
        """
        Applies processors to an ItemList.

        Parameters
        ----------
        item_list : ItemList
            The ItemList to process.
        proc : Processor | Iterable[Processor]
            The processor or list of processors to apply.

        Returns
        -------
        ItemList
            The processed ItemList.
        """
        return item_list.new(compose(item_list.data, proc))

    def __repr__(self):
        return f"{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def x_obj(self, idx):
        """
        Returns the input object at index idx after applying all
        processors in proc_x.

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
        Returns the label object at index idx after applying all
        processors in proc_y.

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

    def _obj(self, items, idx, procs):
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deprocess(item)
        return item

    @staticmethod
    def _label_by_func(ds, label_func, cls=ItemList):
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
        label_func : Callable
            The function to be used for labeling.
        proc_x : Callable | None, default=None
            The processor to be applied to the input data.
        proc_y : Callable | None, default=None
            The processor to be applied to the label data.

        Returns
        -------
        LabeledData
            The labeled ItemList.
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
    f_name : str | Path
        Filename to get the parent directory.

    Returns
    -------
    str
        Name of the parent directory.
    """
    return Path(f_name).parent.name


def label_by_func(
    splitted_data: SplitData,
    label_func: Callable,
    proc_x: Callable | None = None,
    proc_y: Callable | None = None,
) -> SplitData:
    """
    Label splitted data using `label_func`.

    Parameters
    ----------
    splitted_data : SplitData
        The splitted data to be labeled.
    label_func : Callable
        The function to be used for labeling.
    proc_x : Callable | None, default=None
        The processor to be applied to the input data.
    proc_y : Callable | None, default=None
        The processor to be applied to the label data.

    Returns
    -------
    SplitData
        The labeled splitted data.
    """
    train = LabeledData.label_by_func(
        splitted_data.train, label_func, proc_x=proc_x, proc_y=proc_y
    )
    valid = LabeledData.label_by_func(
        splitted_data.valid, label_func, proc_x=proc_x, proc_y=proc_y
    )
    return SplitData(train, valid)
