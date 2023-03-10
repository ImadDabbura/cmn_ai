from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def na_percentages(
    df: pd.DataFrame, formatted: bool = True
) -> pd.Series | pd.DataFrame:
    """
    Compute percentage of missing values in `df` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to compute missing values.
    formatted : bool, default=True
        Whether to return styled/formatted dataframe or raw percentages.

    Returns
    -------
    res: pd.Series or pd.DataFrame
        Percentages of missing values in each column.
    """
    res = df.isna().mean()
    res = res[res > 0.0].sort_values(ascending=False)
    if not formatted:
        return res
    return (
        pd.DataFrame(res)
        .reset_index()
        .rename(columns={"index": "feature", 0: "na_percentages"})
        .style.hide_index()
        .background_gradient(cmap=sns.light_palette("red", as_cmap=True))
        .format({"na_percentages": "{:.3%}"})
    )


def get_ecdf(a: list | np.array | pd.Series) -> np.array:
    """
    Compute empirical cumulative distribution function of `a`.

    Parameters
    ----------
    a : list, Array, or pd.Series
        Array to compute ECDF on.

    Returns
    -------
    x : np.array
        Sorted version of given array in ascending order.
    y : np.array
        Cumulative probability of each value if the sorted array.
    """
    x = np.sort(a)
    y = np.arange(1, len(a) + 1) / len(a)
    return x, y
