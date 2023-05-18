"""
Exploratory Data Analysis (EDA) is the first thing that needs to be done
before start working on model development. This module contains common
utilities for EDA on tabular data. Most of these functions are around plotting,
data quality, and computing stats on the data.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
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


def plot_ecdf(a: list | np.array | pd.Series, xlabel: str = "X") -> None:
    """
    Plot empirical cumulative distribution of `a`.

    Parameters
    ----------
    a : list, Array, or pd.Series
        Array to compute ECDF on.
    xlabel : str, default="X"
        XlLabel of the plot.
    """
    # Check empirical cumulative distribution
    a = np.array(a)
    x, y = get_ecdf(a)
    plt.plot(x, y, marker=".", linestyle="none")
    # Get normal distributed data
    x_norm = np.random.normal(a.mean(), a.std(), size=len(x))
    x, y = get_ecdf(x_norm)
    plt.plot(x, y, marker=".", linestyle="none")
    plt.xlabel(xlabel)
    plt.legend(["Empirical CDF", "Normal CDF"])
    plt.title("Empirical vs Normal CDF")


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
    plt.figure(figsize=figsize)
    plt.bar(
        range(1, len(cum_var_exp) + 1),
        var_ratio,
        align="center",
        color="red",
        label="Individual explained variance",
    )
    plt.step(
        range(1, len(cum_var_exp) + 1),
        cum_var_exp,
        where="mid",
        label="Cumulative explained variance",
    )
    plt.xticks(range(1, len(cum_var_exp)))
    plt.legend(loc="best")
    plt.xlabel("Principal component index", {"fontsize": 14})
    plt.ylabel("Explained variance ratio", {"fontsize": 14})
    plt.title("PCA on training data", {"fontsize": 18})


def plot_corr_matrix(
    df: pd.DataFrame, method: str = "pearson", figsize: tuple = (12, 6)
) -> None:
    """
    Plot correlation matrix using `method`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to compute correlation.
    method : str, default='pearson'
        Method of correlation.
    figsize : tuple, default=(12, 6)
        Figure size.
    """
    corr_matrix = df.corr(method=method).round(2)
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Plot the heat map
    sns.set(font_scale=1.2)
    plt.style.use("seaborn-v0_8-white")
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(0, 120, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": 0.5},
        xticklabels=df.columns,
        yticklabels=df.columns,
        annot_kws={"size": 12},
    )
    plt.title(f"{method.capitalize()} Correlation", {"fontsize": 18})


def plot_featurebased_hier_clustering(
    X: np.ndarray | pd.DataFrame,
    feature_names: np.ndarray | list | None = None,
    linkage_method: str = "single",
    figsize: tuple = (16, 12),
) -> None:
    """
    Plot features-based hierarchical clustering based on spearman correlation
    matrix.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        Data to compute hierarchical clustering.
    feature_names : np.ndarray | list | None, default=None
        Feature names to use as labels with plotting.
    linkage_method : str, default="single"
        method for calculating the distance between clusters.
    figsize : tuple, optional
        Figure size.
    """
    corr = np.round(spearmanr(X).correlation, 4)
    corr_densed = hierarchy.distance.squareform(1 - corr)
    z = hierarchy.linkage(corr_densed, linkage_method)
    plt.figure(figsize=figsize)
    hierarchy.dendrogram(
        z, orientation="left", labels=feature_names, leaf_font_size=16
    )
