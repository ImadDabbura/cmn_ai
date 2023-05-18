"""
This modules includes most of the utilities related to working data in general,
and with `pytorch` related data in specific. It includes functions from
composing transforms and getting train/valid `DataLoader`s to splitting and
labeling `Dataset`s.
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
from .utils import listify, setify


def get_dls(
    train_ds: Dataset, valid_ds: Dataset, batch_size: int, **kwargs
) -> tuple[DataLoader]:
    """
    Returns two dataloaders: 1 for training and 1 for 1 for validation. The
    validation dataloader has twice the batch size and doesn't shuffle data.
    """
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(
            valid_ds, batch_size=batch_size * 2, shuffle=False, **kwargs
        ),
    )


default_device = "cuda" if torch.cuda.is_available() else "cpu"


def to_device(
    x: Tensor | Iterable[Tensor] | Mapping[str, Tensor],
    device: str = default_device,
):
    """
    Copy tensor(s) to device. If the tensor is already on the device,
    returns the tensor itself.

    Parameters
    ----------
    x : Tensor | Iterable[Tensor] | Mapping[str, Tensor]
        Tensor or collection of tensors to move to devive.
    device : str, default='cuda` if available else 'cpu'
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
    Copy tensor(s) to CPU. If a tensor is already on the CPU, returns the
    tensor itself; otherwise, returns a copy of the tensor.
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
    Collate inputs from HF Dataset dictionary and returns list of inputs after
    applying pytorch's default collate function.

    Parameters
    ----------
    ds : DatasetDict
        HF Dataset dictionary.

    Returns
    -------
    function : tuple
        Wrapper function that returns tuple of collated inputs.
    """
    get = itemgetter(*ds.features)

    def _f(batch):
        return get(default_collate(batch))

    return _f


def collate_device(device: torch.device) -> Callable:
    """
    Collate inputs from batch and copy it to `device`.

    Parameters
    ----------
    device : torch.device
        Device to copy batch to.

    Returns
    -------
    function : callable
        Wrapper function that returns tuple of collated inputs.
    """

    def _f(batch):
        return to_device(default_collate(batch), device)

    return _f


class DataLoaders:
    def __init__(self, *dls):
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
        DataLoaders
            Train/valid data loaders.
        """
        return cls(
            *get_dls(
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
    Applies transformations in `funcs` to the input `x` in  `order` order.
    """
    for func in sorted(listify(funcs), key=lambda x: getattr(x, order, 0)):
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
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
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
    items and and changing repr to return first 10 items along with total
    number items and the class name. This will be the base class where other
    containers will inherit from.

    Parameters
    ----------
    items : Any
        Items to create list from.
    """

    def __init__(self, items) -> None:
        self.data = listify(items)

    def __repr__(self) -> str:
        res = (
            f"{self.__class__.__name__}: ({len(self.data):,} items)\n"
            f"{self.data[:10]}"
        )
        if len(self) > 10:
            res = res[:-1] + ", ...]"
        return res


class ItemList(ListContainer):
    """
    Base class for all type of datasets such as image, text, etc.

    Parameters
    ----------
    items : Sequence
        Items to create list.
    path : str | Path, default="."
        Path of the items that were used to create the list.
    tfms : Callable | None, default=None
        Transformations to apply items before returning them.
    """

    def __init__(
        self,
        items: Sequence,
        path: str | Path = ".",
        tfms: Callable | None = None,
    ) -> None:
        super().__init__(items)
        self.path = Path(path)
        self.tfms = tfms

    def __repr__(self) -> str:
        return super().__repr__() + f"\nPath: {self.path.resolve()}"

    def new(self, items, cls=None) -> ItemList:
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, self.tfms)

    def get(self, item):
        """Every class that inherits from ItemList has to override this method."""
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
    Randomly split items with `p_valid` probability to be in the validation set.

    Parameters
    ----------
    f_name : str
        Item's filename. Not used here, but left for API consistency with
        other splitters.
    p_valid : float, optional
        Probability of the item to be in the validation set.

    Returns
    -------
    bool
        Whether the item is in training or validation directories.
    """
    return np.random.random() < p_valid


def grandparent_splitter(
    f_name: str | Path, valid_name: str = "valid", train_name: str = "train"
) -> bool | None:
    """
    Split items based on whether they fall under validation or training
    directories. This assumes that the directory structure is train/label/items
    or valid/label/items.

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
        Whether the item is in training or validation directories.
        If neither, returns None.
    """
    gp = Path(f_name).parent.parent.name
    if gp == valid_name:
        return True
    elif gp == train_name:
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

    Parameters
    ----------
    train : ItemList
        Training items.
    valid : ItemList
        Validation items.
    """

    def __init__(self, train: ItemList, valid: ItemList) -> None:
        self.train = train
        self.valid = valid

    def __getattr__(self, k):
        return getattr(self.train, k)

    # This is needed if we want to pickle SplitData objects and be able to load
    # it back without recursion errors
    def __setstate__(self, data):
        self.__dict__.update(data)

    @classmethod
    def split_by_func(cls, item_list, split_func) -> SplitData:
        """
        Split item list by splitter function and returns a SplitData object.
        """
        train_files, val_files = split_by_func(item_list.data, split_func)
        train_list, val_list = map(item_list.new, (train_files, val_files))
        return cls(train_list, val_list)

    def to_dls(
        self, batch_size: int = 32, **kwargs
    ) -> tuple[DataLoader, DataLoader]:
        """
        Returns a tuple of training and validation DataLoaders object using
        train and valid datasets.
        """
        return get_dls(self.train, self.valid, batch_size, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}\n---------\nTrain - {self.train}\n\n"
            f"Valid - {self.valid}\n"
        )


class LabeledData:
    def __init__(
        self,
        x: ItemList,
        y: ItemList,
        proc_x: Processor | Iterable[Processor] | None = None,
        proc_y: Processor | Iterable[Processor] | None = None,
    ) -> None:
        """
        Create a labeled data and expose both x & y as item lists after passing
        them through all processors.

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
        return item_list.new(compose(item_list.data, proc))

    def __repr__(self):
        return f"{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def x_obj(self, idx):
        return self.obj(self.x, idx, self.proc_x)

    def y_obj(self, idx):
        return self.obj(self.y, idx, self.proc_y)

    def obj(self, items, idx, procs):
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deprocess(item)
        return item

    @staticmethod
    def _label_by_func(ds, label_func, cls=ItemList):
        return cls([label_func(o) for o in ds.data], path=ds.path)

    @classmethod
    def label_by_func(
        cls, item_list, label_func, proc_x=None, proc_y=None
    ) -> LabeledData:
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
    """
    return Path(f_name).parent.name


def label_by_func(
    splitted_data, label_func, proc_x=None, proc_y=None
) -> SplitData:
    """Label splitted data using `label_func`."""
    train = LabeledData.label_by_func(
        splitted_data.train, label_func, proc_x=proc_x, proc_y=proc_y
    )
    valid = LabeledData.label_by_func(
        splitted_data.valid, label_func, proc_x=proc_x, proc_y=proc_y
    )
    return SplitData(train, valid)
