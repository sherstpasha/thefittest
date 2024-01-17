import random
from typing import List
from typing import Tuple
from typing import Union

from numba import boolean
from numba import float64
from numba import int64
from numba import njit

import numpy as np
from numpy.typing import NDArray

from ..utils import binary_search_interval
from ..utils import check_for_value
from ..utils import numpy_group_by


def sattolo_shuffle(items: Union[List, NDArray]) -> None:
    """
    Perform an in-place Sattolo's algorithm shuffle on the input array.

    Sattolo's algorithm shuffles the elements of the array in such a way that
    no element ends up in its original position. This is achieved by iteratively
    selecting a random element and swapping it with the element at a randomly chosen
    position ahead of it.

    Parameters
    ----------
    items : Union[List, NDArray]
        The input array or list to be shuffled. The function shuffles the elements
        in place.

    Returns
    -------
    None
        The function operates in-place, and the input array is shuffled without
        returning a new array.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.random import sattolo_shuffle
    >>>
    >>> # Example with a list
    >>> my_list = [1, 2, 3, 4, 5]
    >>> sattolo_shuffle(my_list)
    >>> print("Shuffled List:", my_list)
    Shuffled List: ...
    >>>
    >>> # Example with a NumPy array
    >>> my_array = np.array([1, 2, 3, 4, 5])
    >>> sattolo_shuffle(my_array)
    >>> print("Shuffled NumPy Array:", my_array)
    Shuffled NumPy Array: ...
    """

    i = len(items)
    while i > 1:
        i = i - 1
        j = random.randrange(i)
        items[j], items[i] = items[i], items[j]


@njit(float64[:](float64, float64, int64))
def cauchy_distribution(loc: np.float64, scale: np.float64, size: np.int64) -> NDArray[np.float64]:
    """
    Generate an array of random numbers from a Cauchy distribution.

    Parameters
    ----------
    loc : np.float64
        The location parameter of the Cauchy distribution.
    scale : np.float64
        The scale parameter of the Cauchy distribution.
    size : np.int64
        The size of the array to generate.

    Returns
    -------
    NDArray[np.float64]
        An array of random numbers drawn from a Cauchy distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.random import cauchy_distribution
    >>>
    >>> # Generate an array of 10 random numbers from a Cauchy distribution
    >>> loc_value = 0.0
    >>> scale_value = 1.0
    >>> size_value = 10
    >>> cauchy_result = cauchy_distribution(loc_value, scale_value, size_value)
    >>>
    >>> print("Cauchy Distribution Result:", cauchy_result)
    Cauchy Distribution Result: ...
    """

    x_ = np.random.standard_cauchy(size=size).astype(np.float64)
    return loc + scale * x_


@njit(int64[:](float64[:], int64, boolean))
def random_weighted_sample(
    weights: NDArray[np.float64], quantity: Union[np.int64, int] = 1, replace: bool = True
) -> NDArray[np.int64]:
    """
    Generate a random weighted sample.

    Parameters
    ----------
    weights : NDArray[np.float64]
        1D array of weights representing the probability of each element being selected.
    quantity : Union[np.int64, int]
        The number of elements to sample. Default is 1.
    replace : bool
        Whether sampling is done with replacement. Default is True.

    Returns
    -------
    NDArray[np.int64]
        An array of sampled indices.

    Examples
    --------
    >>> from thefittest.utils.random import random_weighted_sample
    >>> import numpy as np
    >>>
    >>> # Example with replacement
    >>> weights = np.array([0.3, 0.2, 0.5])
    >>> sampled_indices = random_weighted_sample(weights, quantity=2, replace=True)
    >>> print("Sampled Indices:", sampled_indices)
    Sampled Indices: ...
    >>>
    >>> # Example without replacement
    >>> sampled_indices_no_replace = random_weighted_sample(weights, quantity=2, replace=False)
    >>> print("Sampled Indices (No Replace):", sampled_indices_no_replace)
    Sampled Indices (No Replace): ...
    """

    if not replace:
        assert len(weights) >= quantity
    sample = np.empty(quantity, dtype=np.int64)

    cumsumweights = np.cumsum(weights)
    sumweights = cumsumweights[-1]

    i = 0
    while i < quantity:
        roll = sumweights * np.random.rand()
        ind = binary_search_interval(roll, cumsumweights)
        if not replace:
            if check_for_value(ind, sample, i):
                continue

        sample[i] = ind
        i += 1
    return sample


@njit(int64[:](int64, int64, boolean))
def random_sample(
    range_size: np.int64, quantity: np.int64, replace: bool = True
) -> NDArray[np.int64]:
    """
    Generate a random sample from a range.

    Parameters
    ----------
    range_size : np.int64
        The size of the range to sample from.
    quantity : np.int64
        The number of elements to sample.
    replace : bool
        Whether sampling is done with replacement. Default is True.

    Returns
    -------
    NDArray[np.int64]
        An array of sampled indices.

    Examples
    --------
    >>> from thefittest.utils.random import random_sample
    >>>
    >>> # Example with replacement
    >>> sampled_indices = random_sample(range_size=10, quantity=3, replace=True)
    >>> print("Sampled Indices:", sampled_indices)
    Sampled Indices: ...
    >>>
    >>> # Example without replacement
    >>> sampled_indices_no_replace = random_sample(range_size=10, quantity=3, replace=False)
    >>> print("Sampled Indices (No Replace):", sampled_indices_no_replace)
    Sampled Indices (No Replace): ...
    """

    if not replace:
        assert range_size >= quantity
    sample = np.empty(quantity, dtype=np.int64)
    i = 0
    while i < quantity:
        ind = random.randrange(int(range_size))

        if not replace:
            if check_for_value(ind, sample, i):
                continue

        sample[i] = ind
        i += 1
    return sample


def stratified_sample(data: NDArray[np.int64], sample_ratio: float) -> NDArray[np.int64]:
    """
    Generate a stratified random sample.

    Parameters
    ----------
    data : NDArray[np.int64]
        1D array of integer values representing the strata or groups.
    sample_ratio : float
        The ratio of the sample size to the total data size.

    Returns
    -------
    NDArray[np.int64]
        An array of sampled indices.

    Examples
    --------
    >>> from thefittest.utils.random import stratified_sample
    >>> import numpy as np
    >>>
    >>> # Example
    >>> data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    >>> sample_ratio = 0.8
    >>> sampled_indices = stratified_sample(data, sample_ratio)
    >>> print("Stratified Sampled Indices:", sampled_indices)
    Stratified Sampled Indices: ...
    >>> print("Stratified Sampled Values:", data[sampled_indices])
    Stratified Sampled Values: ...
    """

    sample = []
    data_size = len(data)
    sample_size = int(sample_ratio * data_size)
    indexes = np.arange(len(data), dtype=np.int64)
    keys, groups = numpy_group_by(indexes, data)
    assert sample_size >= len(keys)

    for group in groups:
        group_size = len(group)
        sample_size_i = max(1, int((group_size / data_size) * sample_size))
        sample_i_id = random_sample(group_size, sample_size_i, False)
        sample_i = group[sample_i_id]
        sample.extend(sample_i)

    sample_numpy = np.array(sample, dtype=np.int64)
    return sample_numpy


def train_test_split_stratified(
    X: NDArray[Union[np.float64, np.int64]],
    y: NDArray[np.int64],
    test_size: float,
) -> Tuple:
    """
    Split the dataset into training and testing sets while preserving the class distribution.

    Parameters
    ----------
    X : NDArray[Union[np.float64, np.int64]]
        The input features.
    y : NDArray[np.int64]
        The target labels.
    test_size : float
        The proportion of the dataset to include in the test split.

    Returns
    -------
    Tuple
        A tuple containing the following four elements:
        - X_train: Training data features.
        - X_test: Testing data features.
        - y_train: Training data labels.
        - y_test: Testing data labels.

    Examples
    --------
    >>> from thefittest.utils.random import train_test_split_stratified
    >>> import numpy as np
    >>>
    >>> # Example
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    >>> test_size = 0.25
    >>> X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size)
    >>> print("X_train:", X_train)
    X_train: ...
    >>> print("X_test:", X_test)
    X_test: ...
    >>> print("y_train:", y_train)
    y_train: ...
    >>> print("y_test:", y_test)
    y_test: ...
    """

    indexes = np.arange(len(y), dtype=np.int64)
    sample_id = stratified_sample(y, test_size)
    test_id = sample_id
    train_id = np.setdiff1d(indexes, test_id)
    return (
        X[train_id].astype(np.float64),
        X[test_id].astype(np.float64),
        y[train_id].astype(np.int64),
        y[test_id].astype(np.int64),
    )


def train_test_split(
    X: NDArray[Union[np.float64, np.int64]],
    y: NDArray[Union[np.float64, np.int64]],
    test_size: float,
) -> Tuple:
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    X : NDArray[Union[np.float64, np.int64]]
        The input features.
    y : NDArray[Union[np.float64, np.int64]]
        The target labels.
    test_size : float
        The proportion of the dataset to include in the test split.

    Returns
    -------
    Tuple
        A tuple containing the following four elements:
        - X_train: Training data features.
        - X_test: Testing data features.
        - y_train: Training data labels.
        - y_test: Testing data labels.

    Examples
    --------
    >>> from thefittest.utils.random import train_test_split
    >>> import numpy as np
    >>>
    >>> # Example
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 1, 1])
    >>> test_size = 0.25
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
    >>> print("X_train:", X_train)
    X_train: ...
    >>> print("X_test:", X_test)
    X_test: ...
    >>> print("y_train:", y_train)
    y_train: ...
    >>> print("y_test:", y_test)
    y_test: ...
    """

    data_size = len(X)
    sample_size = int(test_size * data_size)
    indexes = np.arange(data_size, dtype=np.int64)
    sample_id = random_sample(range_size=data_size, quantity=sample_size, replace=False)
    test_id = sample_id
    train_id = np.setdiff1d(indexes, test_id)
    return (
        X[train_id].astype(np.float64),
        X[test_id].astype(np.float64),
        y[train_id].astype(np.float64),
        y[test_id].astype(np.float64),
    )
