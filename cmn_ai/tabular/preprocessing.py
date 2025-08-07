"""
Data preprocessing and transformation utilities for tabular data.

This module provides transformers and preprocessing utilities that are compatible
with scikit-learn Pipeline and ColumnTransformer. It includes specialized
transformers for handling date/time features and other common preprocessing
tasks in machine learning workflows.

The transformers follow scikit-learn conventions with fit/transform methods
and can be seamlessly integrated into preprocessing pipelines.

Classes
-------
DateTransformer : Transform date features into derived attributes

Examples
--------
Basic usage with date features:

    >>> import pandas as pd
    >>> from cmn_ai.tabular.preprocessing import DateTransformer
    >>>
    >>> # Create sample data with date column
    >>> df = pd.DataFrame({
    ...     'date': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-30']),
    ...     'value': [1, 2, 3]
    ... })
    >>>
    >>> # Transform date features
    >>> transformer = DateTransformer(time=True, drop=True)
    >>> df_transformed = transformer.fit_transform(df)
    >>> print(df_transformed.columns)
    Index(['value', 'date_Year', 'date_Month', 'date_Week', 'date_Day',
           'date_Dayofweek', 'date_Dayofyear', 'date_Is_month_end',
           'date_Is_month_start', 'date_Is_quarter_end', 'date_Is_quarter_start',
           'date_Is_year_end', 'date_Is_year_start', 'date_Hour', 'date_Minute',
           'date_Second', 'date_na_indicator'], dtype='object')

Integration with scikit-learn Pipeline:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> pipeline = Pipeline([
    ...     ('date_transform', DateTransformer()),
    ...     ('scaler', StandardScaler())
    ... ])
    >>> pipeline.fit_transform(df)

Dependencies
------------
- numpy : For numerical operations and array handling
- pandas : For data manipulation and analysis
- scikit-learn : For base transformer classes and pipeline compatibility

Notes
-----
- All transformers are compatible with scikit-learn Pipeline and ColumnTransformer
- DateTransformer automatically detects datetime64 columns if date_feats is None
- Missing value indicators are automatically added for date features
- Time-related features (Hour, Minute, Second) are optional and disabled by default
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
    Transform date features by deriving useful date/time attributes.

    This transformer extracts various temporal features from datetime columns,
    including date attributes (year, month, day, etc.) and optional time
    attributes (hour, minute, second). It also adds missing value indicators
    for each date feature.

    The transformer is compatible with scikit-learn Pipeline and can be used
    in preprocessing workflows. It automatically detects datetime64 columns
    if no specific date features are provided.

    Attributes
    ----------
    date_feats : Iterable[str] | None, default=None
        list of date feature column names to transform.
    time : bool
        Whether to include time-related features (Hour, Minute, Second).
    drop : bool
        Whether to drop original date features after transformation.
    attrs : list[str]
        list of date attributes to extract from each date feature.

    Examples
    --------
    >>> import pandas as pd
    >>> from cmn_ai.tabular.preprocessing import DateTransformer
    >>>
    >>> df = pd.DataFrame({
    ...     'date': pd.to_datetime(['2023-01-01', '2023-02-15']),
    ...     'value': [1, 2]
    ... })
    >>>
    >>> transformer = DateTransformer(time=True)
    >>> df_transformed = transformer.fit_transform(df)
    >>> print(df_transformed.columns.tolist())
    ['value', 'date_Year', 'date_Month', 'date_Week', 'date_Day',
     'date_Dayofweek', 'date_Dayofyear', 'date_Is_month_end',
     'date_Is_month_start', 'date_Is_quarter_end', 'date_Is_quarter_start',
     'date_Is_year_end', 'date_Is_year_start', 'date_Hour', 'date_Minute',
     'date_Second', 'date_na_indicator']
    """

    def __init__(
        self,
        date_feats: Iterable[str] | None = None,
        time: bool = False,
        drop: bool = True,
    ) -> None:
        """
        Initialize DateTransformer.

        Parameters
        ----------
        date_feats : Iterable[str], default=None
            Date features to transform. If None, all features with `datetime64`
            data type will be automatically detected and used.
        time : bool, default=False
            Whether to add time-related derived features such as Hour, Minute, Second.
            If True, adds Hour, Minute, and Second attributes to the transformation.
        drop : bool, default=True
            Whether to drop original date features after transformation.
            If True, original date columns are removed from the output.
            If False, original date columns are preserved alongside derived features.

        Examples
        --------
        >>> transformer = DateTransformer(
        ...     date_feats=['date_col'],
        ...     time=True,
        ...     drop=False
        ... )
        """
        self.date_feats = date_feats
        self.time = time
        self.drop = drop
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
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.DataFrame | None = None,
    ) -> DateTransformer:
        """
        Fit the transformer by identifying date features.

        This method populates the date_feats attribute if not provided at
        initialization. It scans the input dataframe for columns with
        datetime64 data type and stores them for transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing the date features to transform.
            Must contain at least one datetime64 column if date_feats is None.
        y : np.ndarray | pd.DataFrame, default=None
            Target values. Included for compatibility with scikit-learn
            transformers and pipelines but not used in this transformer.

        Returns
        -------
        self : DateTransformer
            Fitted transformer instance with populated date_feats attribute.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.to_datetime(['2023-01-01', '2023-02-15']),
        ...     'value': [1, 2]
        ... })
        >>> transformer = DateTransformer()
        >>> fitted_transformer = transformer.fit(df)
        >>> print(fitted_transformer.date_feats)
        ['date']
        """
        if not self.date_feats:
            self.date_feats = X.select_dtypes("datetime64").columns.tolist()
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Transform date features by extracting derived attributes.

        For each date feature, this method creates new columns with extracted
        date/time attributes. It also adds missing value indicators for each
        date feature. The transformation preserves all non-date columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing the date features to transform.
            Must contain the same date features used during fitting.
        y : np.ndarray | pd.DataFrame, default=None
            Target values. Included for compatibility with scikit-learn
            transformers and pipelines but not used in this transformer.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe with derived date/time features.
            Contains original non-date columns plus new derived columns.
            Original date columns are dropped if drop=True.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.to_datetime(['2023-01-01', '2023-02-15']),
        ...     'value': [1, 2]
        ... })
        >>> transformer = DateTransformer(time=True)
        >>> transformer.fit(df)
        >>> df_transformed = transformer.transform(df)
        >>> print(df_transformed['date_Year'].tolist())
        [2023, 2023]
        >>> print(df_transformed['date_Month'].tolist())
        [1, 2]
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
