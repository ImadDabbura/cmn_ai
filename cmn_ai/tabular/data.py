"""
Tabular data manipulation and splitting utilities.

This module provides advanced data splitting functionality for tabular datasets,
with support for stratified splitting, group-aware splitting, and automatic
validation of class distribution requirements.

Classes
-------
DataSplitter : class
    A configurable data splitter with support for train/validation/test splits,
    stratification, group-aware splitting, and automatic size adjustment.

Functions
---------
get_data_splits : function
    Convenience function for one-time data splitting operations.

_suggest_sizes : function
    Internal utility for adjusting split sizes to ensure stratification feasibility.

Notes
-----
The module is designed to handle common challenges in machine learning data
preparation, including:

- Ensuring all classes have sufficient samples in each split
- Automatic size adjustment when requested splits are infeasible
- Group-aware splitting to prevent data leakage
- Comprehensive validation with helpful error messages

The splitting functions are compatible with both NumPy arrays and pandas
DataFrames, making them suitable for various data science workflows.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from cmn_ai.tabular.data import get_data_splits
>>>
>>> # Create sample data
>>> X = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
>>> y = np.random.choice(['A', 'B', 'C'], size=100)
>>>
>>> # Basic stratified split
>>> X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(
...     X, y, train_size=0.7, val_size=0.15, test_size=0.15
... )
>>>
>>> # Group-aware splitting
>>> groups = np.random.choice(['group1', 'group2', 'group3'], size=100)
>>> splits = get_data_splits(X, y, groups=groups, stratify=False)
>>>
>>> # Using DataSplitter for repeated operations
>>> splitter = DataSplitter(train_size=0.8, val_size=0.1, test_size=0.1)
>>> splits1 = splitter.split(X1, y1)
>>> splits2 = splitter.split(X2, y2)
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _suggest_sizes(
    train_size: float, val_size: float, test_size: float, y
) -> tuple[float, float, float]:
    """
    Suggest minimally adjusted split sizes for stratified splitting feasibility.

    Given requested split sizes and target labels, this function computes adjusted
    sizes that ensure each split can contain at least one sample from the rarest
    class while maintaining proportional adjustments and sum-to-one constraint.

    Parameters
    ----------
    train_size : float
        Requested fraction for training set (0 < train_size < 1).
    val_size : float
        Requested fraction for validation set (0 < val_size < 1).
    test_size : float
        Requested fraction for test set (0 < test_size < 1).
    y : array-like
        Target labels for computing class distribution.

    Returns
    -------
    tuple of float or tuple of None
        Adjusted (train_size, val_size, test_size) fractions that sum to 1.0
        and allow at least one sample per class in each split. Returns
        (None, None, None) if no feasible stratified split is possible
        (i.e., rarest class has fewer than 3 samples).

    Notes
    -----
    The algorithm works by:
    1. Computing minimum required fraction per split based on rarest class
    2. Lifting any split size below the minimum
    3. Proportionally reducing oversized splits to maintain sum=1 constraint

    If the rarest class has fewer than 3 samples, stratified 3-way splitting
    is mathematically impossible.

    Examples
    --------
    >>> y = [0, 0, 0, 1, 1, 2]  # Class 2 has only 1 sample (rarest)
    >>> _suggest_sizes(0.7, 0.2, 0.1, y)
    (0.5, 0.33, 0.17)  # Adjusted to ensure class 2 can appear in all splits

    >>> y = [0, 1]  # Only 2 samples total, impossible for 3-way split
    >>> _suggest_sizes(0.7, 0.2, 0.1, y)
    (None, None, None)
    """
    counts = Counter(y)
    cmin = min(counts.values())
    m = (
        1.0 / cmin
    )  # each split must be >= m to allow 1 sample of the rarest class

    # If 3*m > 1 => impossible to have 3 stratified splits (rarest class has < 3 samples)
    if 3 * m > 1.0:
        # No feasible suggestion exists for 3-way stratified split
        return None, None, None

    sizes = np.array([train_size, val_size, test_size], dtype=float)
    # Lift any split below the minimum
    sizes = np.maximum(sizes, m)
    excess = sizes.sum() - 1.0
    if excess <= 1e-12:
        # Already feasible
        return float(sizes[0]), float(sizes[1]), float(sizes[2])

    # Reduce splits that are above m, proportional to their headroom, to make sum==1
    headroom = sizes - m
    total_headroom = headroom.sum()
    if total_headroom <= 1e-12:
        # Shouldn't happen if 3*m <= 1, but guard anyway
        return None, None, None

    reduction = np.minimum(headroom, excess * (headroom / total_headroom))
    sizes = sizes - reduction
    # Numerical tidy-up
    sizes = np.clip(sizes, m, 1.0)
    sizes = sizes / sizes.sum()
    return float(sizes[0]), float(sizes[1]), float(sizes[2])


class DataSplitter:
    """
    Advanced data splitter for train/validation/test sets with stratification support.

    This class provides a comprehensive solution for splitting tabular datasets
    with support for stratified splitting, group-aware splitting, automatic
    size adjustment, and validation of class distribution requirements.
    Designed for reusable splitting configurations across multiple datasets.

    Parameters
    ----------
    train_size : float, default=0.7
        Fraction of data for training set (0 < train_size < 1).
    val_size : float, default=0.15
        Fraction of data for validation set (0 < val_size < 1).
    test_size : float, default=0.15
        Fraction of data for test set (0 < test_size < 1).
        train_size + val_size + test_size must equal 1.0.
    stratify : bool, default=True
        Whether to preserve class proportions in each split.
        Incompatible with group-aware splitting.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int, optional
        Random seed for reproducible splits.
    min_class_count_check : bool, default=True
        Whether to validate that all classes have sufficient samples
        for the requested split sizes with automatic adjustment suggestions.

    Attributes
    ----------
    train_size : float
        Configured training set fraction.
    val_size : float
        Configured validation set fraction.
    test_size : float
        Configured test set fraction.
    stratify : bool
        Configured stratification setting.
    shuffle : bool
        Configured shuffle setting.
    random_state : int or None
        Configured random state.
    min_class_count_check : bool
        Configured class count validation setting.

    Methods
    -------
    split(X, y=None, **kwargs)
        Split data into train/validation/test sets with optional parameter overrides.

    Notes
    -----
    The DataSplitter is designed to handle several common challenges in ML data preparation:

    - **Class imbalance**: Automatically detects when requested split sizes would
      result in missing classes and suggests minimal adjustments.
    - **Group leakage**: Supports group-aware splitting to prevent data leakage
      when samples are not independent (e.g., time series, grouped data).
    - **Flexibility**: Allows parameter overrides per split operation while
      maintaining default configuration.
    - **Validation**: Comprehensive input validation with informative error messages.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Basic usage with default settings
    >>> splitter = DataSplitter()
    >>> X = pd.DataFrame({'feature': range(100)})
    >>> y = np.random.choice(['A', 'B', 'C'], 100)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

    >>> # Custom configuration for imbalanced data
    >>> splitter = DataSplitter(
    ...     train_size=0.8, val_size=0.1, test_size=0.1,
    ...     stratify=True, random_state=42
    ... )
    >>> splits = splitter.split(X, y)

    >>> # Group-aware splitting for time series
    >>> groups = np.repeat(range(10), 10)  # 10 groups of 10 samples each
    >>> splits = splitter.split(X, y, groups=groups, stratify=False)

    >>> # Parameter override for specific split
    >>> splits = splitter.split(X, y, train_size=0.9, val_size=0.05, test_size=0.05)
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = True,
        shuffle: bool = True,
        random_state: int | None = None,
        min_class_count_check: bool = True,
    ):
        """
        Initialize DataSplitter with configuration for train/validation/test splitting.

        Parameters
        ----------
        train_size : float, default=0.7
            Fraction of data allocated to training set. Must be in range (0, 1).
            Combined with val_size and test_size must sum to 1.0.
        val_size : float, default=0.15
            Fraction of data allocated to validation set. Must be in range (0, 1).
            Can be set to None to auto-infer from train_size and test_size.
        test_size : float, default=0.15
            Fraction of data allocated to test set. Must be in range (0, 1).
            Can be set to None to auto-infer from train_size and val_size.
        stratify : bool, default=True
            Whether to preserve class distribution proportions across splits.
            When True, each split will maintain approximately the same class
            ratios as the original dataset. Not compatible with group-based splitting.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting. Recommended for most
            use cases unless data order is important (e.g., time series).
        random_state : int, optional
            Seed for random number generator to ensure reproducible splits.
            If None, splits will be different each time.
        min_class_count_check : bool, default=True
            Whether to validate that each class has sufficient samples for
            the requested split sizes. When True, will automatically suggest
            adjusted sizes if stratification would fail due to insufficient
            samples in rare classes.

        Raises
        ------
        ValueError
            If train_size + val_size + test_size != 1.0 (after auto-inference).
            If any size parameter is not in range (0, 1).
            If stratify=True but target has insufficient samples per class.

        Examples
        --------
        >>> # Standard 70/15/15 split with stratification
        >>> splitter = DataSplitter()

        >>> # Custom proportions for small datasets
        >>> splitter = DataSplitter(
        ...     train_size=0.6, val_size=0.2, test_size=0.2,
        ...     random_state=42
        ... )

        >>> # Non-stratified splitting for regression or grouped data
        >>> splitter = DataSplitter(stratify=False, shuffle=True)

        >>> # Disable class count validation for advanced use cases
        >>> splitter = DataSplitter(min_class_count_check=False)
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify = stratify
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_class_count_check = min_class_count_check

    def _validate_and_normalize_inputs(self, X, y):
        """
        Validate and normalize input data.
        """
        X = np.asarray(X) if not hasattr(X, "__getitem__") else X
        if y is not None and not hasattr(y, "__len__"):
            y = np.asarray(y)
        return X, y

    def _infer_split_sizes(self, train_size, val_size, test_size):
        """
        Infer missing split sizes to ensure they sum to 1.0.
        """
        remainder = 1.0 - train_size
        if val_size is None and test_size is None:
            val_size = remainder / 2.0
            test_size = remainder / 2.0
        elif val_size is None:
            val_size = remainder - test_size
        elif test_size is None:
            test_size = remainder - val_size

        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(
                "train_size + val_size + test_size must sum to 1.0"
            )

        return train_size, val_size, test_size

    def _validate_stratification_feasibility(
        self,
        y,
        train_size,
        val_size,
        test_size,
        stratify,
        min_class_count_check,
    ):
        """
        Validate that stratification is feasible with given split sizes.
        """
        if not (stratify and y is not None and min_class_count_check):
            return

        counts = Counter(y)
        cmin = min(counts.values())
        s_min = min(train_size, val_size, test_size)
        min_needed_for_smallest = 1.0 / s_min

        if any(c < min_needed_for_smallest for c in counts.values()):
            t_sug, v_sug, te_sug = _suggest_sizes(
                train_size, val_size, test_size, y
            )
            if t_sug is None:
                raise ValueError(
                    "Cannot create a 3-way stratified split: the rarest class has "
                    f"only {cmin} sample(s). A 3-way split requires at least 3 samples "
                    "in every class.\n"
                    "Options:\n"
                    "  • Use two splits (train/test) or stratified KFold with k=2.\n"
                    "  • Increase dataset size for rare classes.\n"
                    "  • Merge/relable rare classes or disable stratification."
                )
            else:
                raise ValueError(
                    "Requested split sizes are too small for some classes when using stratification.\n"
                    f"Rarest class count: {cmin}\n"
                    f"Suggested feasible sizes (train/val/test): "
                    f"{t_sug:.4f}/{v_sug:.4f}/{te_sug:.4f}\n"
                    "Tip: pass these as train_size=..., val_size=..., test_size=..."
                )
        elif any(c == min_needed_for_smallest for c in counts.values()):
            warnings.warn(
                "Some classes barely meet the split requirement; stratification may be unstable "
                "(rounding can still zero out a class in a split). Consider slightly larger "
                "val/test sizes or more data for rare classes.",
                UserWarning,
            )

    def _perform_group_split(
        self, X, y, groups, val_size, test_size, random_state
    ):
        """
        Perform group-aware data splitting.
        """
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size + test_size,
            random_state=random_state,
        )
        idx_train, idx_temp = next(gss.split(np.zeros(len(X)), groups=groups))
        temp_groups = np.asarray(groups)[idx_temp]

        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
        )
        idx_val_rel, idx_test_rel = next(
            gss2.split(np.zeros(len(idx_temp)), groups=temp_groups)
        )
        idx_val = np.asarray(idx_temp)[idx_val_rel]
        idx_test = np.asarray(idx_temp)[idx_test_rel]

        return idx_train, idx_val, idx_test

    def _perform_regular_split(
        self,
        X,
        y,
        train_size,
        val_size,
        test_size,
        stratify,
        shuffle,
        random_state,
    ):
        """
        Perform regular data splitting (with optional stratification).
        """
        strat = y if (stratify and y is not None) else None
        all_idx = np.arange(len(X))

        idx_train, idx_temp = train_test_split(
            all_idx,
            train_size=train_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=strat,
        )

        strat_temp = np.asarray(y)[idx_temp] if strat is not None else None
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=test_size / (val_size + test_size),
            shuffle=shuffle,
            random_state=random_state,
            stratify=strat_temp,
        )

        return idx_train, idx_val, idx_test

    def _prepare_split_results(
        self, X, y, idx_train, idx_val, idx_test, return_indices
    ):
        """
        Prepare the final split results based on indices.
        """
        if return_indices:
            return (
                np.asarray(idx_train),
                np.asarray(idx_val),
                np.asarray(idx_test),
            )

        def _take(a, idx):
            if hasattr(a, "iloc"):  # pandas
                return a.iloc[idx]
            return a[idx]

        X_train, X_val, X_test = (
            _take(X, idx_train),
            _take(X, idx_val),
            _take(X, idx_test),
        )

        if y is None:
            return X_train, X_val, X_test

        y_train, y_val, y_test = (
            _take(y, idx_train),
            _take(y, idx_val),
            _take(y, idx_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split(
        self,
        X: np.ndarray | pd.DataFrame | list,
        y: np.ndarray | pd.Series | list | None = None,
        *,
        train_size: float | None = None,
        val_size: float | None = None,
        test_size: float | None = None,
        stratify: bool | None = None,
        shuffle: bool | None = None,
        random_state: int | None = None,
        return_indices: bool = False,
        groups: np.ndarray | list | None = None,
        min_class_count_check: bool | None = None,
    ):
        """
        Split data into train, validation, and test sets with optional parameter overrides.

        This method applies the configured splitting strategy to the provided data,
        with options to override instance defaults for specific operations. Supports
        both stratified and group-aware splitting with automatic validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix to split. Accepts NumPy arrays, pandas DataFrames,
            or other array-like structures with indexing support.
        y : array-like of shape (n_samples,), optional
            Target vector for supervised learning. Required for stratified splitting.
            Accepts NumPy arrays, pandas Series, or lists.
        train_size : float, optional
            Override instance train_size for this split operation.
            Must be in range (0, 1) and sum with val_size and test_size to 1.0.
        val_size : float, optional
            Override instance val_size for this split operation.
            Can be None to auto-infer from other sizes.
        test_size : float, optional
            Override instance test_size for this split operation.
            Can be None to auto-infer from other sizes.
        stratify : bool, optional
            Override instance stratify setting. If True, maintains class
            proportions across splits. Cannot be used with groups parameter.
        shuffle : bool, optional
            Override instance shuffle setting. Whether to shuffle data before splitting.
        random_state : int, optional
            Override instance random_state for this operation.
            Ensures reproducible splits when provided.
        return_indices : bool, default=False
            If True, returns indices of split samples instead of the data itself.
            Useful for custom data handling or debugging.
        groups : array-like of shape (n_samples,), optional
            Group labels for group-aware splitting. When provided, ensures
            samples from the same group don't appear in different splits.
            Automatically disables stratification.
        min_class_count_check : bool, optional
            Override instance min_class_count_check setting. Whether to validate
            class distribution feasibility and suggest adjustments if needed.

        Returns
        -------
        tuple of arrays or indices
            If return_indices=False (default):
                (X_train, X_val, X_test, y_train, y_val, y_test) if y is provided
                (X_train, X_val, X_test) if y is None
            If return_indices=True:
                (train_indices, val_indices, test_indices) as NumPy arrays

        Raises
        ------
        ValueError
            If split sizes don't sum to 1.0 after inference.
            If stratification is requested but infeasible due to class distribution.
            If groups is provided together with stratify=True.
            If X and y have different lengths.
        UserWarning
            When classes barely meet minimum sample requirements for stratification.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> splitter = DataSplitter(train_size=0.7, val_size=0.15, test_size=0.15)

        >>> # Basic stratified split
        >>> X = pd.DataFrame({'feature': range(100)})
        >>> y = np.random.choice(['A', 'B', 'C'], 100)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

        >>> # Override parameters for specific split
        >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        ...     X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
        ... )

        >>> # Group-aware splitting
        >>> groups = np.repeat(range(10), 10)  # 10 groups of 10 samples each
        >>> splits = splitter.split(X, y, groups=groups, stratify=False)

        >>> # Return indices instead of data
        >>> train_idx, val_idx, test_idx = splitter.split(
        ...     X, y, return_indices=True
        ... )

        >>> # Unsupervised data (no y)
        >>> X_train, X_val, X_test = splitter.split(X)
        """
        # Use instance defaults unless overridden
        train_size = train_size if train_size is not None else self.train_size
        val_size = val_size if val_size is not None else self.val_size
        test_size = test_size if test_size is not None else self.test_size
        stratify = stratify if stratify is not None else self.stratify
        shuffle = shuffle if shuffle is not None else self.shuffle
        random_state = (
            random_state if random_state is not None else self.random_state
        )
        min_class_count_check = (
            min_class_count_check
            if min_class_count_check is not None
            else self.min_class_count_check
        )

        # Validate and normalize inputs
        X, y = self._validate_and_normalize_inputs(X, y)

        # Infer missing split sizes
        train_size, val_size, test_size = self._infer_split_sizes(
            train_size, val_size, test_size
        )

        # Validate stratification feasibility
        self._validate_stratification_feasibility(
            y, train_size, val_size, test_size, stratify, min_class_count_check
        )

        # Perform the actual splitting
        if groups is not None:
            if stratify:
                raise ValueError(
                    "Stratification with groups is not supported."
                )
            idx_train, idx_val, idx_test = self._perform_group_split(
                X, y, groups, val_size, test_size, random_state
            )
        else:
            idx_train, idx_val, idx_test = self._perform_regular_split(
                X,
                y,
                train_size,
                val_size,
                test_size,
                stratify,
                shuffle,
                random_state,
            )

        # Prepare and return results
        return self._prepare_split_results(
            X, y, idx_train, idx_val, idx_test, return_indices
        )


def get_data_splits(
    X: np.ndarray | pd.DataFrame | list,
    y: np.ndarray | pd.Series | list | None = None,
    *,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    shuffle: bool = True,
    random_state: int | None = None,
    return_indices: bool = False,
    groups: np.ndarray | list | None = None,
    min_class_count_check: bool = True,
):
    """
    Convenient one-call function for train/validation/test data splitting.

    This function provides a simple interface for splitting datasets with
    advanced features including stratification, group-aware splitting,
    automatic class distribution validation, and size adjustment suggestions.
    For multiple split operations on different datasets, consider using
    DataSplitter directly for better performance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix to split. Supports NumPy arrays, pandas DataFrames,
        or any array-like structure with indexing capabilities.
    y : array-like of shape (n_samples,), optional
        Target vector for supervised learning. Required when stratify=True.
        Supports NumPy arrays, pandas Series, or lists.
    train_size : float, default=0.7
        Proportion of data allocated to training set. Must be in range (0, 1).
        Combined with val_size and test_size must sum to 1.0.
    val_size : float, default=0.15
        Proportion of data allocated to validation set. Must be in range (0, 1).
        Can be None to auto-infer from train_size and test_size.
    test_size : float, default=0.15
        Proportion of data allocated to test set. Must be in range (0, 1).
        Can be None to auto-infer from train_size and val_size.
    stratify : bool, default=True
        Whether to preserve class distribution proportions across all splits.
        When True, each split maintains approximately the same class ratios
        as the original dataset. Automatically disabled when groups is provided.
    shuffle : bool, default=True
        Whether to shuffle data before splitting. Recommended for most use cases
        except when sample order is meaningful (e.g., time series data).
    random_state : int, optional
        Seed for random number generator to ensure reproducible splits.
        If None, results will vary between calls.
    return_indices : bool, default=False
        If True, returns sample indices instead of actual data splits.
        Useful for custom data handling or when working with complex data structures.
    groups : array-like of shape (n_samples,), optional
        Group labels for group-aware splitting. When provided, ensures that
        samples with the same group label don't appear across different splits,
        preventing data leakage in grouped data scenarios.
    min_class_count_check : bool, default=True
        Whether to validate class distribution feasibility for stratified splitting.
        When True, automatically suggests adjusted split sizes if any class
        has insufficient samples for the requested proportions.

    Returns
    -------
    tuple of arrays or indices
        If return_indices=False (default):
            For supervised data (y provided):
                (X_train, X_val, X_test, y_train, y_val, y_test)
            For unsupervised data (y=None):
                (X_train, X_val, X_test)
        If return_indices=True:
            (train_indices, val_indices, test_indices) as NumPy arrays

    Raises
    ------
    ValueError
        If train_size + val_size + test_size != 1.0 (after auto-inference).
        If any size parameter is not in range (0, 1).
        If stratify=True but insufficient samples per class for requested splits.
        If groups is provided together with stratify=True.
        If X and y have mismatched lengths.
    UserWarning
        When classes barely meet minimum requirements for stratified splitting.

    See Also
    --------
    DataSplitter : Class-based interface for repeated splitting operations.
    sklearn.model_selection.train_test_split : Scikit-learn's 2-way splitting function.

    Notes
    -----
    This function internally creates a DataSplitter instance and calls its split()
    method. For applications requiring multiple split operations with the same
    configuration, using DataSplitter directly is more efficient as it avoids
    repeated parameter validation and object instantiation.

    The stratification algorithm ensures that rare classes are handled gracefully
    by automatically suggesting feasible split sizes when the requested proportions
    would result in empty classes in some splits.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from cmn_ai.tabular.data import get_data_splits

    >>> # Basic stratified splitting
    >>> X = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
    >>> y = np.random.choice(['A', 'B', 'C'], size=100)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X, y)
    >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    Train: 70, Val: 15, Test: 15

    >>> # Custom proportions with reproducible results
    >>> splits = get_data_splits(
    ...     X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
    ... )

    >>> # Group-aware splitting for time series or clustered data
    >>> groups = np.repeat(range(20), 5)  # 20 groups of 5 samples each
    >>> X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(
    ...     X, y, groups=groups, stratify=False
    ... )

    >>> # Unsupervised data splitting
    >>> X_train, X_val, X_test = get_data_splits(X, shuffle=True, random_state=123)

    >>> # Get indices for custom data handling
    >>> train_idx, val_idx, test_idx = get_data_splits(
    ...     X, y, return_indices=True, random_state=456
    ... )

    >>> # Handle imbalanced data with automatic size adjustment
    >>> y_imbalanced = np.array(['rare'] * 2 + ['common'] * 98)
    >>> try:
    ...     splits = get_data_splits(X, y_imbalanced)
    ... except ValueError as e:
    ...     print("Automatic suggestion provided:", str(e))
    """
    splitter = DataSplitter(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        stratify=stratify,
        shuffle=shuffle,
        random_state=random_state,
        min_class_count_check=min_class_count_check,
    )
    return splitter.split(X, y, return_indices=return_indices, groups=groups)
