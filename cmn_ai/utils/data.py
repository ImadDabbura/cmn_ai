from collections.abc import Iterable
from operator import itemgetter
from typing import Mapping

import torch
from datasets.dataset_dict import DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate


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


def_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def to_device(
    x: Tensor | Iterable[Tensor] | Mapping[str, Tensor],
    device: str = def_device,
):
    """
    Copy tensor(s) to device. If the tensor is already on the device,
    returns the tensor itself.

    Parameters
    ----------
    x : Tensor | Iterable[Tensor] | Mapping[str, Tensor]
        Tensor or collection of tensors to move to devive.
    device : str, default='cuda` if available else 'cpu'
        _description_, by default def_device

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


def collate_dict(ds: DatasetDict):
    """
    Collate inputs from HF Dataset dictionary and returns list of inputs after
    applying pytorch's defacult collate function.

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
                bs=batch_size,
                collate_fn=collate_dict(dd["train"]),
                **kwargs
            )
        )
