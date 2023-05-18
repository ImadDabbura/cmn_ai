"""
Most datasets need to be preprocessed/transformed before they can be passed to
the model. This module includes common transformers that are compatible with
[`sklearn`](www.scikit-learn.org) `Pipeline` or `ColumnTransformer`.
"""
from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
import pandas.api.types as types
from sklearn.base import BaseEstimator, TransformerMixin, clone


class DateTransformer(TransformerMixin, BaseEstimator):
    """
    Transform date features by deriving useful date/time attributes:

    - date attributes: `Year, Month, Week, Day, Dayofweek, Dayofyear,
    Is_month_end, Is_month_start, Is_quarter_end, Is_quarter_start,
    Is_year_end, Is_year_start`.
    - time attributes: `Hour, Minute, Second`.

    Parameters
    ----------
    date_feats : Iterable, default=None
        Date features to transform. If None, all features with `datetime64`
        data type will be used.
    time : bool, default=False
        Whether to add time-related derived features such as Hour/Minute/...
    drop : bool, default=True
        Whether to drop date features used.
    """

    def __init__(
        self,
        date_feats: Iterable[str] | None = None,
        time: bool = False,
        drop: bool = True,
    ) -> None:
        """Initialize self."""
        self.date_feats = date_feats
        self.time = time
        self.drop = drop
        self.dates_min = {}
        self.attrs = [
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ]
        if self.time:
            self.attrs += ["Hour", "Minute", "Second"]

    def fit(
        self, X: pd.DataFrame, y: np.array | pd.DataFrame | None = None
    ) -> DateTransformer:
        """
        Populate date features if not provided at initialization.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe that has the date features to transform.
        y : np.array | pd.DataFrame | None, default=None
            Included for completeness to be compatible with scikit-learn
            transformers and pipelines but will not be used.

        Returns
        -------
        self:
            Fitted date transformer.
        """
        if not self.date_feats:
            self.date_feats = [
                col for col in X.columns if types.is_datetime64_dtype(X[col])
            ]
        return self

    def transform(
        self, X: pd.DataFrame, y: np.array | pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Derive the date/time attributes for all date features.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe that has the date features to transform.
        y : np.array | pd.DataFrame | None, default=None
            Included for completeness to be compatible with scikit-learn
            transformers and pipelines but will not be used.

        Returns
        -------
        X_tr : pd.DataFrame
            Dataframe with derived date/time features and NaN indicators.
        """
        X_tr = X.copy()
        for col, attr in product(self.date_feats, self.attrs):
            if attr == "Week" and hasattr(X_tr[col].dt, "isocalendar"):
                X_tr[f"{col}_{attr}"] = (
                    X_tr[col]
                    .dt.isocalendar()
                    .week.astype(X_tr[col].dt.day.dtype)
                )
                continue
            X_tr[f"{col}_{attr}"] = getattr(X_tr[col].dt, attr.lower())
        for col in self.date_feats:
            X_tr[f"{col}_na_indicator"] = X_tr[col].isna().astype(np.int8)
        if self.drop:
            return X_tr.drop(columns=self.date_feats)
        return X_tr
