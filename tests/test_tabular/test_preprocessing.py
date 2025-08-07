"""
Unit tests for tabular preprocessing module.

This module contains comprehensive tests for the DateTransformer class,
covering initialization, fitting, transformation, edge cases, and
scikit-learn compatibility.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cmn_ai.tabular.preprocessing import DateTransformer


class TestDateTransformer:
    """Test cases for DateTransformer class."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.sample_dates = pd.to_datetime(
            [
                "2023-01-01 10:30:45",
                "2023-02-15 14:20:30",
                "2023-03-30 08:15:00",
                "2023-12-31 23:59:59",
            ]
        )

        self.df_with_dates = pd.DataFrame(
            {
                "date": self.sample_dates,
                "value": [1, 2, 3, 4],
                "category": ["A", "B", "A", "C"],
            }
        )

        self.df_multiple_dates = pd.DataFrame(
            {
                "start_date": pd.to_datetime(
                    ["2023-01-01", "2023-02-15", "2023-03-30"]
                ),
                "end_date": pd.to_datetime(
                    ["2023-01-31", "2023-02-28", "2023-03-31"]
                ),
                "value": [1, 2, 3],
            }
        )

    def test_init_default_parameters(self):
        """Test DateTransformer initialization with default parameters."""
        transformer = DateTransformer()

        assert transformer.date_feats is None
        assert transformer.time is False
        assert transformer.drop is True
        assert "Year" in transformer.attrs
        assert "Month" in transformer.attrs
        assert "Hour" not in transformer.attrs

    def test_init_with_parameters(self):
        """Test DateTransformer initialization with custom parameters."""
        transformer = DateTransformer(
            date_feats=["date_col"], time=True, drop=False
        )

        assert transformer.date_feats == ["date_col"]
        assert transformer.time is True
        assert transformer.drop is False
        assert "Hour" in transformer.attrs
        assert "Minute" in transformer.attrs
        assert "Second" in transformer.attrs

    def test_fit_with_auto_detection(self):
        """Test fitting with automatic date feature detection."""
        transformer = DateTransformer()
        fitted_transformer = transformer.fit(self.df_with_dates)

        assert fitted_transformer.date_feats == ["date"]
        assert fitted_transformer is transformer

    def test_fit_with_specified_features(self):
        """Test fitting with specified date features."""
        transformer = DateTransformer(date_feats=["date"])
        fitted_transformer = transformer.fit(self.df_with_dates)

        assert fitted_transformer.date_feats == ["date"]

    def test_fit_with_multiple_dates(self):
        """Test fitting with multiple date columns."""
        transformer = DateTransformer()
        fitted_transformer = transformer.fit(self.df_multiple_dates)

        assert set(fitted_transformer.date_feats) == {"start_date", "end_date"}

    def test_fit_with_no_dates(self):
        """Test fitting with dataframe containing no date columns."""
        df_no_dates = pd.DataFrame(
            {"value": [1, 2, 3], "category": ["A", "B", "C"]}
        )

        transformer = DateTransformer()
        fitted_transformer = transformer.fit(df_no_dates)

        assert fitted_transformer.date_feats == []

    def test_transform_basic(self):
        """Test basic transformation without time features."""
        transformer = DateTransformer(time=False, drop=True)
        transformer.fit(self.df_with_dates)
        result = transformer.transform(self.df_with_dates)

        # Check that original date column is dropped
        assert "date" not in result.columns

        # Check that derived features are created
        expected_features = [
            "date_Year",
            "date_Month",
            "date_Week",
            "date_Day",
            "date_Dayofweek",
            "date_Dayofyear",
            "date_Is_month_end",
            "date_Is_month_start",
            "date_Is_quarter_end",
            "date_Is_quarter_start",
            "date_Is_year_end",
            "date_Is_year_start",
            "date_na_indicator",
        ]

        for feature in expected_features:
            assert feature in result.columns

        # Check that non-date columns are preserved
        assert "value" in result.columns
        assert "category" in result.columns

        # Check specific values
        assert result["date_Year"].tolist() == [2023, 2023, 2023, 2023]
        assert result["date_Month"].tolist() == [1, 2, 3, 12]

    def test_transform_with_time_features(self):
        """Test transformation with time features enabled."""
        transformer = DateTransformer(time=True, drop=True)
        transformer.fit(self.df_with_dates)
        result = transformer.transform(self.df_with_dates)

        # Check time-related features
        time_features = ["date_Hour", "date_Minute", "date_Second"]
        for feature in time_features:
            assert feature in result.columns

        # Check specific time values
        assert result["date_Hour"].tolist() == [10, 14, 8, 23]
        assert result["date_Minute"].tolist() == [30, 20, 15, 59]

    def test_transform_without_dropping(self):
        """Test transformation without dropping original date columns."""
        transformer = DateTransformer(drop=False)
        transformer.fit(self.df_with_dates)
        result = transformer.transform(self.df_with_dates)

        # Check that original date column is preserved
        assert "date" in result.columns

        # Check that derived features are also present
        assert "date_Year" in result.columns
        assert "date_Month" in result.columns

    def test_transform_multiple_dates(self):
        """Test transformation with multiple date columns."""
        transformer = DateTransformer(time=True, drop=True)
        transformer.fit(self.df_multiple_dates)
        result = transformer.transform(self.df_multiple_dates)

        # Check features for both date columns
        for date_col in ["start_date", "end_date"]:
            assert f"{date_col}_Year" in result.columns
            assert f"{date_col}_Month" in result.columns
            assert f"{date_col}_na_indicator" in result.columns

        # Check that original date columns are dropped
        assert "start_date" not in result.columns
        assert "end_date" not in result.columns

    def test_transform_with_missing_values(self):
        """Test transformation with missing values in date columns."""
        df_with_nulls = self.df_with_dates.copy()
        df_with_nulls.loc[1, "date"] = pd.NaT

        transformer = DateTransformer()
        transformer.fit(df_with_nulls)
        result = transformer.transform(df_with_nulls)

        # Check that na_indicator is created and works correctly
        assert "date_na_indicator" in result.columns
        assert result["date_na_indicator"].tolist() == [0, 1, 0, 0]

    def test_fit_transform(self):
        """Test fit_transform method."""
        transformer = DateTransformer(time=True, drop=True)
        result = transformer.fit_transform(self.df_with_dates)

        # Check that transformation worked
        assert "date_Year" in result.columns
        assert "date_Hour" in result.columns
        assert "date" not in result.columns

    def test_get_params(self):
        """Test get_params method for scikit-learn compatibility."""
        transformer = DateTransformer(
            date_feats=["date_col"], time=True, drop=False
        )
        params = transformer.get_params()

        assert params["date_feats"] == ["date_col"]
        assert params["time"] is True
        assert params["drop"] is False

    def test_set_params(self):
        """Test set_params method for scikit-learn compatibility."""
        transformer = DateTransformer()
        transformer.set_params(time=True, drop=False)

        assert transformer.time is True
        assert transformer.drop is False

    def test_clone(self):
        """Test that transformer can be cloned."""
        transformer = DateTransformer(time=True, drop=False)
        cloned_transformer = clone(transformer)

        assert cloned_transformer.time is True
        assert cloned_transformer.drop is False
        assert cloned_transformer is not transformer

    def test_pipeline_integration(self):
        """Test integration with scikit-learn Pipeline."""
        # Create dataframe with only numeric and date columns for StandardScaler
        df_numeric = pd.DataFrame(
            {"date": self.sample_dates, "value": [1, 2, 3, 4]}
        )

        pipeline = Pipeline(
            [
                ("date_transform", DateTransformer(time=True, drop=True)),
                ("scaler", StandardScaler()),
            ]
        )

        # Test that pipeline can be fitted and transformed
        result = pipeline.fit_transform(df_numeric)

        # Result should be a numpy array after StandardScaler
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(df_numeric)

    def test_week_calculation(self):
        """Test week calculation using isocalendar."""
        transformer = DateTransformer()
        transformer.fit(self.df_with_dates)
        result = transformer.transform(self.df_with_dates)

        # Check that week column exists and has reasonable values
        assert "date_Week" in result.columns
        assert all(1 <= week <= 53 for week in result["date_Week"])

    def test_boolean_features(self):
        """Test boolean date features (is_month_end, is_year_start, etc.)."""
        transformer = DateTransformer()
        transformer.fit(self.df_with_dates)
        result = transformer.transform(self.df_with_dates)

        # Check boolean features
        boolean_features = [
            "date_Is_month_end",
            "date_Is_month_start",
            "date_Is_quarter_end",
            "date_Is_quarter_start",
            "date_Is_year_end",
            "date_Is_year_start",
        ]

        for feature in boolean_features:
            assert feature in result.columns
            # Check that values are boolean (0 or 1)
            assert all(val in [0, 1] for val in result[feature])

    def test_error_handling_missing_columns(self):
        """Test error handling when specified date columns don't exist."""
        transformer = DateTransformer(date_feats=["nonexistent_date"])

        # Fit should work, but transform should fail
        transformer.fit(self.df_with_dates)
        with pytest.raises(KeyError):
            transformer.transform(self.df_with_dates)

    def test_error_handling_non_datetime_column(self):
        """Test error handling when specified column is not datetime."""
        transformer = DateTransformer(date_feats=["value"])

        # Fit should work, but transform should fail
        transformer.fit(self.df_with_dates)
        with pytest.raises(AttributeError):
            transformer.transform(self.df_with_dates)

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame(columns=["date", "value"])
        empty_df["date"] = pd.to_datetime([])
        empty_df["value"] = []

        transformer = DateTransformer()
        transformer.fit(empty_df)
        result = transformer.transform(empty_df)

        assert len(result) == 0
        assert "date_Year" in result.columns

    def test_single_row_dataframe(self):
        """Test handling of single row dataframe."""
        single_row_df = pd.DataFrame(
            {"date": [pd.to_datetime("2023-01-01")], "value": [1]}
        )

        transformer = DateTransformer()
        transformer.fit(single_row_df)
        result = transformer.transform(single_row_df)

        assert len(result) == 1
        assert result["date_Year"].iloc[0] == 2023
        assert result["date_Month"].iloc[0] == 1

    def test_different_datetime_formats(self):
        """Test handling of different datetime formats."""
        df_mixed_formats = pd.DataFrame(
            {
                "date1": pd.to_datetime(["2023-01-01", "2023-02-15"]),
                "date2": pd.to_datetime(
                    ["2023-01-01 10:30:00", "2023-02-15 14:20:00"]
                ),
                "value": [1, 2],
            }
        )

        transformer = DateTransformer(time=True)
        transformer.fit(df_mixed_formats)
        result = transformer.transform(df_mixed_formats)

        # Check that both date columns are processed
        assert "date1_Year" in result.columns
        assert "date2_Year" in result.columns
        assert "date2_Hour" in result.columns  # Only date2 has time info
