"""Tests for tabular EDA utilities."""

import importlib

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use("Agg", force=True)

import cmn_ai.tabular.eda as eda


def test_importing_eda_does_not_set_global_seaborn_style(monkeypatch):
    """Test importing EDA utilities does not mutate global seaborn style."""
    calls = []
    monkeypatch.setattr(
        eda.sns, "set", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    importlib.reload(eda)

    assert calls == []


def test_na_percentages_returns_sorted_missing_value_rates():
    """Test NA percentages excludes complete columns and sorts descending."""
    df = pd.DataFrame(
        {
            "complete": [1, 2, 3],
            "some_missing": [1, np.nan, 3],
            "mostly_missing": [np.nan, np.nan, 3],
        }
    )

    result = eda.na_percentages(df, formatted=False)

    assert result.index.tolist() == ["mostly_missing", "some_missing"]
    np.testing.assert_allclose(result.to_numpy(), [2 / 3, 1 / 3])


def test_get_ecdf_returns_sorted_values_and_cumulative_probabilities():
    """Test ECDF values are sorted with cumulative probabilities."""
    x, y = eda.get_ecdf([3.0, 1.0, 2.0])

    np.testing.assert_allclose(x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [1 / 3, 2 / 3, 1.0])


def test_eda_plot_functions_smoke():
    """Test EDA plotting helpers run on small deterministic inputs."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 1.5, 3.5, 4.5],
            "c": [4.0, 3.0, 2.0, 1.0],
        }
    )

    eda.plot_ecdf(df["a"], xlabel="a")
    eda.plot_corr_matrix(df)
    eda.plot_pca_var_explained(PCA().fit(df))
    eda.plot_featurebased_hier_clustering(df)

    assert len(matplotlib.pyplot.get_fignums()) >= 4
    matplotlib.pyplot.close("all")
