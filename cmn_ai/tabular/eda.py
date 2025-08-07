"""
Exploratory Data Analysis (EDA) utilities for tabular data.

This module provides comprehensive tools for exploratory data analysis on
tabular datasets. It includes functions for data quality assessment,
statistical analysis, correlation analysis, dimensionality reduction
visualization, and hierarchical clustering of features.

The module is designed to help data scientists and analysts quickly
understand their data through various visualization and statistical
techniques before proceeding with model development.

Functions
---------
na_percentages : Compute and visualize missing value percentages
get_ecdf : Calculate empirical cumulative distribution function
plot_ecdf : Plot ECDF with normal distribution comparison
plot_pca_var_explained : Visualize PCA variance explanation
plot_corr_matrix : Create correlation matrix heatmaps
plot_featurebased_hier_clustering : Hierarchical clustering of features

Examples
--------
Basic usage for data quality assessment:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from cmn_ai.tabular.eda import na_percentages, plot_corr_matrix
    >>>
    >>> # Create sample data with missing values
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4, 5],
    ...     'B': [1, np.nan, 3, 4, np.nan],
    ...     'C': [1, 2, 3, 4, 5]
    ... })
    >>>
    >>> # Check missing value percentages
    >>> na_percentages(df, formatted=False)
    B    0.4
    A    0.2
    dtype: float64
    >>>
    >>> # Plot correlation matrix
    >>> plot_corr_matrix(df, method='pearson')

PCA variance analysis:

    >>> from sklearn.decomposition import PCA
    >>> from cmn_ai.tabular.eda import plot_pca_var_explained
    >>>
    >>> # Fit PCA on data
    >>> pca = PCA().fit(df.dropna())
    >>> plot_pca_var_explained(pca)

Dependencies
------------
- numpy : For numerical operations and array handling
- pandas : For data manipulation and analysis
- matplotlib : For plotting and visualization
- seaborn : For enhanced statistical visualizations
- scipy : For statistical functions and clustering
- scikit-learn : For PCA and machine learning utilities

Notes
-----
- All plotting functions use matplotlib backend and will display plots
  when called in interactive environments
- Correlation analysis supports Pearson, Kendall, and Spearman methods
- Hierarchical clustering uses Spearman correlation for feature similarity
- ECDF functions include comparison with theoretical normal distribution
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sns.set()


def na_percentages(
    df: pd.DataFrame, formatted: bool = True
) -> pd.Series | pd.DataFrame:
    """
    Compute percentage of missing values in dataframe columns.

    This function calculates the percentage of missing values (NaN) in each
    column of the input dataframe and returns either a raw pandas Series
    or a formatted/styled dataframe for better visualization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to compute missing values percentages.
    formatted : bool, default=True
        If True, returns a styled dataframe with background gradient.
        If False, returns a raw pandas Series with percentages.

    Returns
    -------
    pd.Series | pd.DataFrame
        If formatted=True: Styled dataframe with feature names and NA percentages
        If formatted=False: Pandas Series with column names and NA percentages

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [1, np.nan, 3], 'C': [1, 2, 3]})
    >>> na_percentages(df, formatted=False)
    B    0.333333
    A    0.333333
    dtype: float64
    """
    res = df.isna().mean()
    res = res[res > 0.0].sort_values(ascending=False)
    if not formatted:
        return res
    return (
        pd.DataFrame(res)
        .reset_index()
        .rename(columns={"index": "feature", 0: "na_percentages"})
        .style.hide(axis="index")
        .background_gradient(cmap=sns.light_palette("red", as_cmap=True))
        .format({"na_percentages": "{:.3%}"})
    )


def get_ecdf(
    a: list[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical cumulative distribution function (ECDF) of input array.

    The ECDF is a step function that jumps up by 1/n at each of the n data points.
    Its value at any specified value of the measured variable is the fraction
    of observations of the measured variable that are less than or equal to
    the specified value.

    Parameters
    ----------
    a : list[float] | np.ndarray | pd.Series
        Input array to compute ECDF on. Can be a list, numpy array, or pandas Series.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x : Sorted version of input array in ascending order
        y : Cumulative probability values (0 to 1) corresponding to each sorted value

    Examples
    --------
    >>> import numpy as np
    >>> x, y = get_ecdf([1, 3, 2, 4])
    >>> print(x)
    [1 2 3 4]
    >>> print(y)
    [0.25 0.5  0.75 1.  ]
    """
    x = np.sort(a)
    y = np.arange(1, len(a) + 1) / len(a)
    return x, y


def plot_ecdf(
    a: list[float] | np.ndarray | pd.Series, xlabel: str = "X"
) -> None:
    """
    Plot empirical cumulative distribution function (ECDF) with normal comparison.

    Creates a plot showing the empirical CDF of the input data alongside
    a theoretical normal CDF with the same mean and standard deviation
    for comparison purposes.

    Parameters
    ----------
    a : list[float] | np.ndarray | pd.Series
        Input array to compute and plot ECDF.
    xlabel : str, default="X"
        Label for the x-axis of the plot.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, 1000)
    >>> plot_ecdf(data, xlabel="Values")
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
    pca_transformer: PCA, figsize: tuple[int, int] = (12, 6)
) -> None:
    """
    Plot individual and cumulative variance explained by PCA components.

    Creates a bar plot showing the individual explained variance ratio for each
    principal component, along with a step plot showing the cumulative explained
    variance. This helps visualize how much information is captured by each
    component and how many components are needed to explain a certain percentage
    of the total variance.

    Parameters
    ----------
    pca_transformer : PCA
        Fitted PCA transformer object from scikit-learn.
    figsize : tuple[int, int], default=(12, 6)
        Figure size as (width, height) in inches.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=100, n_features=10, random_state=42)
    >>> pca = PCA().fit(X)
    >>> plot_pca_var_explained(pca)
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
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Plot correlation matrix heatmap with specified correlation method.

    Creates a heatmap visualization of the correlation matrix using the specified
    correlation method. The upper triangle is masked to avoid redundancy, and
    correlation values are displayed as annotations on the heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to compute correlation matrix.
    method : str, default='pearson'
        Correlation method to use. Options: 'pearson', 'kendall', 'spearman'.
    figsize : tuple[int, int], default=(12, 6)
        Figure size as (width, height) in inches.

    Returns
    -------
    None
        Displays the correlation heatmap using matplotlib and seaborn.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    >>> plot_corr_matrix(df, method='spearman')
    """
    corr_matrix = df.corr(method=method).round(2)
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.set(font_scale=1.2)
    plt.style.use("seaborn-v0_8-white")
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(0, 120, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".1f",
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
    feature_names: np.ndarray | list[str] = None,
    linkage_method: str = "single",
    figsize: tuple[int, int] = (16, 12),
) -> None:
    """
    Plot feature-based hierarchical clustering dendrogram using Spearman correlation.

    Performs hierarchical clustering on features based on their Spearman correlation
    matrix and displays the resulting dendrogram. This helps identify groups of
    features that are highly correlated with each other.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        Input data matrix or dataframe to compute hierarchical clustering.
    feature_names : np.ndarray | list[str], default=None
        Names of features to use as labels in the dendrogram. If None, uses
        column names if X is a DataFrame, or generic labels if X is an array.
    linkage_method : str, default="single"
        Linkage method for hierarchical clustering. Options: 'single', 'complete',
        'average', 'weighted', 'centroid', 'median', 'ward'.
    figsize : tuple[int, int], default=(16, 12)
        Figure size as (width, height) in inches.

    Returns
    -------
    None
        Displays the hierarchical clustering dendrogram using matplotlib.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.random.randn(100, 5)
    >>> feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E']
    >>> plot_featurebased_hier_clustering(X, feature_names, linkage_method='ward')
    """
    corr = np.round(spearmanr(X).correlation, 4)
    corr_densed = hierarchy.distance.squareform(1 - corr)
    z = hierarchy.linkage(corr_densed, linkage_method)
    plt.figure(figsize=figsize)
    hierarchy.dendrogram(
        z, orientation="left", labels=feature_names, leaf_font_size=16
    )
