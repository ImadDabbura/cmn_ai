from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


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


def plot_ecdf(a: list | np.array | pd.Series, xlabel: str = "X"):
    """
    Plot empirical cumulative distribution of `a`.

    Parameters
    ----------
    a : list, Array, or pd.Series
        Array to compute ECDF on.
    xlabel : str, default="X"
        XlLabel of the plot.

    Returns
    -------
    axes : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.
    """
    # Check empirical cumulative distribution
    a = np.array(a)
    x, y = get_ecdf(a)
    _, axes = plt.subplots(1)
    axes.plot(x, y, marker=".", linestyle="none")
    # Get normal distributed data
    x_norm = np.random.normal(a.mean(), a.std(), size=len(x))
    x, y = get_ecdf(x_norm)
    axes.plot(x, y, marker=".", linestyle="none")
    axes.set_xlabel(xlabel)
    axes.legend(["Empirical CDF", "Normal CDF"])
    return axes


def plot_pca_var_explained(
    pca_transformer: PCA, figsize: tuple = (12, 6)
) -> None:
    """
    Plot individual and cumulative of the variance explained by each PCA
    component.

    Parameters
    ----------
    pca_transformer : PCA
        PCA transformer.
    figsize : tuple, default=(12, 6)
        Figure size.
    """
    var_ratio = pca_transformer.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_ratio)
    _, axes = plt.subplots(1, figsize=figsize)
    axes.bar(
        range(1, len(cum_var_exp) + 1),
        var_ratio,
        align="center",
        color="red",
        label="Individual explained variance",
    )
    axes.step(
        range(1, len(cum_var_exp) + 1),
        cum_var_exp,
        where="mid",
        label="Cumulative explained variance",
    )
    axes.set_xticks(range(1, len(cum_var_exp)))
    axes.legend(loc="best")
    axes.set_xlabel("Principal component index", {"fontsize": 14})
    axes.set_ylabel("Explained variance ratio", {"fontsize": 14})
    axes.set_title("PCA on training data", {"fontsize": 18})
