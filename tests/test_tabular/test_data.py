"""Tests for tabular data splitting functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from cmn_ai.tabular.data import DataSplitter, get_data_splits


class TestDataSplitting:
    """Test both DataSplitter class and get_data_splits function."""

    def test_data_splitter_init_defaults(self):
        """Test DataSplitter initialization with defaults."""
        splitter = DataSplitter()
        assert splitter.train_size == 0.7
        assert splitter.val_size == 0.15
        assert splitter.test_size == 0.15
        assert splitter.stratify is True
        assert splitter.shuffle is True
        assert splitter.random_state is None
        assert splitter.min_class_count_check is True

    def test_data_splitter_init_custom(self):
        """Test DataSplitter initialization with custom parameters."""
        splitter = DataSplitter(
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_class_count_check=False,
        )
        assert splitter.train_size == 0.6
        assert splitter.val_size == 0.2
        assert splitter.test_size == 0.2
        assert splitter.stratify is False
        assert splitter.shuffle is False
        assert splitter.random_state == 42
        assert splitter.min_class_count_check is False

    def test_basic_splitting(self):
        """Test basic splitting functionality with both APIs."""
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=5, random_state=42
        )

        # Test function API
        func_result = get_data_splits(X, y, random_state=42)
        assert len(func_result) == 6
        X_train, X_val, X_test, y_train, y_val, y_test = func_result
        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 15
        assert X_test.shape[0] == 15

        # Test class API
        splitter = DataSplitter(random_state=42)
        class_result = splitter.split(X, y)
        assert len(class_result) == 6

    def test_return_indices(self):
        """Test splitting with return_indices=True."""
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=5, random_state=42
        )

        # Function API
        func_result = get_data_splits(
            X, y, return_indices=True, random_state=42
        )
        assert len(func_result) == 3
        idx_train, idx_val, idx_test = func_result
        assert len(idx_train) == 70
        assert len(idx_val) == 15
        assert len(idx_test) == 15

        # Class API
        splitter = DataSplitter(random_state=42)
        class_result = splitter.split(X, y, return_indices=True)
        assert len(class_result) == 3

    def test_no_labels(self):
        """Test splitting without y labels."""
        X = np.random.rand(100, 5)

        # Function API
        func_result = get_data_splits(X, random_state=42)
        assert len(func_result) == 3
        X_train, X_val, X_test = func_result
        assert X_train.shape[0] == 70

        # Class API
        splitter = DataSplitter(random_state=42)
        class_result = splitter.split(X)
        assert len(class_result) == 3

    def test_pandas_data(self):
        """Test splitting with pandas DataFrames and Series."""
        X = pd.DataFrame(
            np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.choice([0, 1, 2], size=100))

        result = get_data_splits(X, y, random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = result
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert list(X_train.columns) == [f"feat_{i}" for i in range(5)]

    def test_custom_split_sizes(self):
        """Test with custom split sizes."""
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=5, random_state=42
        )

        result = get_data_splits(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
        )
        X_train, X_val, X_test, y_train, y_val, y_test = result
        assert X_train.shape[0] == 60
        assert X_val.shape[0] == 20
        assert X_test.shape[0] == 20

    def test_infer_missing_sizes(self):
        """Test inference of missing split sizes."""
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=5, random_state=42
        )

        result = get_data_splits(
            X, y, train_size=0.8, val_size=None, test_size=0.1, random_state=42
        )
        X_train, X_val, X_test, y_train, y_val, y_test = result
        assert X_train.shape[0] == 80
        assert (
            X_val.shape[0] + X_test.shape[0] == 20
        )  # remaining 20% split between val and test
        assert (
            abs(X_val.shape[0] - 10) <= 1
        )  # approximately 10% (stratification may cause slight variation)
        assert abs(X_test.shape[0] - 10) <= 1  # approximately 10%

    def test_parameter_override_in_class(self):
        """Test that split parameters can override instance defaults."""
        splitter = DataSplitter(
            train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
        )
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=5, random_state=42
        )

        result = splitter.split(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2, stratify=False
        )
        X_train, X_val, X_test, y_train, y_val, y_test = result
        assert X_train.shape[0] == 60
        assert X_val.shape[0] == 20
        assert X_test.shape[0] == 20

    def test_group_splitting(self):
        """Test group-aware splitting."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.choice([0, 1, 2], size=100)
        groups = np.repeat(range(20), 5)  # 20 groups of 5 samples each

        result = get_data_splits(
            X, y, groups=groups, stratify=False, random_state=42
        )
        assert len(result) == 6
        X_train, X_val, X_test, y_train, y_val, y_test = result
        # Group splitting may not give exact proportions, just check reasonable sizes
        assert 50 <= X_train.shape[0] <= 85
        assert 5 <= X_val.shape[0] <= 35
        assert 5 <= X_test.shape[0] <= 35

    def test_reusable_splitter(self):
        """Test that splitter can be reused for multiple datasets."""
        splitter = DataSplitter(random_state=42)

        # First dataset
        X1, y1 = make_classification(
            n_samples=100, n_classes=2, random_state=1
        )
        result1 = splitter.split(X1, y1)
        assert len(result1) == 6
        assert result1[0].shape[0] == 70  # 70% of 100

        # Second dataset
        X2, y2 = make_classification(
            n_samples=200, n_classes=3, n_informative=5, random_state=2
        )
        result2 = splitter.split(X2, y2)
        assert len(result2) == 6
        assert result2[0].shape[0] == 140  # 70% of 200

    def test_stratification_error_handling(self):
        """Test error handling for insufficient class samples."""
        X = np.random.rand(6, 5)
        y = [0, 0, 1, 1, 2, 2]  # Only 2 samples per class

        with pytest.raises(
            ValueError, match="Cannot create a 3-way stratified split"
        ):
            get_data_splits(X, y, random_state=42)

    def test_size_validation(self):
        """Test validation that sizes sum to 1.0."""
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=5, random_state=42
        )

        with pytest.raises(ValueError, match="must sum to 1.0"):
            get_data_splits(X, y, train_size=0.5, val_size=0.3, test_size=0.3)

    def test_groups_with_stratify_error(self):
        """Test that groups with stratify raises an error."""
        X = np.random.rand(100, 5)
        y = np.random.choice([0, 1, 2], size=100)
        groups = np.repeat(range(20), 5)

        with pytest.raises(
            ValueError, match="Stratification with groups is not supported"
        ):
            get_data_splits(X, y, groups=groups, stratify=True)

    def test_api_equivalence(self):
        """Test that function and class APIs give same results with same parameters."""
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=5, random_state=42
        )

        # Function API
        func_result = get_data_splits(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
        )

        # Class API
        splitter = DataSplitter(
            train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
        )
        class_result = splitter.split(X, y)

        # Check shapes are identical
        for i in range(6):
            assert func_result[i].shape == class_result[i].shape

        # Check actual values are identical
        for i in range(6):
            if hasattr(func_result[i], "values"):  # pandas
                np.testing.assert_array_equal(
                    func_result[i].values, class_result[i].values
                )
            else:  # numpy
                np.testing.assert_array_equal(func_result[i], class_result[i])
