import os
from collections import UserList
from collections.abc import Iterable, Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Mapping

import fastcore.all as fc
import torch
from datasets.dataset_dict import DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate

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
    def from_dd(cls, dd: DatasetDict, batch_size: int, **kwargs):
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


def compose(x: Any, funcs: Callable, *args, order: str = "_order", **kwargs):
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

    def __init__(self, items):
        self.data = listify(items)

    def __repr__(self):
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
    ):
        super().__init__(items)
        self.path = Path(path)
        self.tfms = tfms

    def __repr__(self):
        return super().__repr__() + f"\nPath: {self.path.resolve()}"

    def new(self, items, cls=None):
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
