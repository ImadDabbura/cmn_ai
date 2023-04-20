from collections.abc import Iterable
from operator import itemgetter
from typing import Mapping

import torch
from datasets.dataset_dict import DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader, default_collate


def get_dls(train_ds, valid_ds, bs, **kwargs):
    """
    Returns two dataloaders: 1 for training and 1 for 1 for validation. The
    validation dataloader has twice the batch size and doesn't shuffle data.
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, shuffle=False, **kwargs),
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
