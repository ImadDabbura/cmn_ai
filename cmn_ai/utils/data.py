from torch.utils.data import DataLoader


def get_dls(train_ds, valid_ds, bs, **kwargs):
    """
    Returns two dataloaders: 1 for training and 1 for 1 for validation. The
    validation dataloader has twice the batch size and doesn't shuffle data.
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, shuffle=False, **kwargs),
    )
