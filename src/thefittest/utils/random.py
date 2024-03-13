import random
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
import numbers

from numba import boolean
from numba import float64
from numba import int64
from numba import njit

import numpy as np
from numpy.typing import NDArray

from ..utils import binary_search_interval
from ..utils import check_for_value


@njit(int64[:](int64[:]))
def sattolo_shuffle(arr):
    """
    Perform Sattolo's algorithm for in-place array shuffling.

    Parameters
    ----------
    arr : int64[:]
        Input array to be shuffled.

    Returns
    -------
    int64[:]
        Shuffled array.

    Examples
    --------
    >>> from numba import jit
    >>> import numpy as np
    >>>
    >>> # Example of using Sattolo's shuffle
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> shuffled_arr = sattolo_shuffle(arr)
    >>> print("Shuffled Array:", shuffled_arr)
    Shuffled Array: ...

    Notes
    -----
    Sattolo's algorithm generates a random cyclic permutation of a given array.
    """
    n = len(arr)
    shuffled_arr = arr.copy()

    for i in range(n - 1, 0, -1):
        j = np.int64(np.floor(random.random() * i))
        shuffled_arr[i], shuffled_arr[j] = shuffled_arr[j], shuffled_arr[i]

    return shuffled_arr


@njit
def sattolo_shuffle_2d(arr):
    """
    Perform Sattolo's algorithm for in-place shuffling of rows in a 2D array.

    Parameters
    ----------
    arr : int64[:, :]
        Input 2D array to be shuffled.

    Returns
    -------
    int64[:, :]
        Shuffled 2D array.

    Examples
    --------
    >>> from numba import jit
    >>> import numpy as np
    >>>
    >>> # Example of using Sattolo's shuffle for a 2D array
    >>> arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> shuffled_arr_2d = sattolo_shuffle_2d(arr_2d)
    >>> print("Shuffled 2D Array:", shuffled_arr_2d)
    Shuffled 2D Array: ...

    Notes
    -----
    Sattolo's algorithm generates a random cyclic permutation of rows in a 2D array.
    """
    n_rows, _ = arr.shape
    shuffled_arr = arr.copy()

    for i in range(n_rows - 1, 0, -1):
        j = np.int64(np.floor(random.random() * i))
        shuffled_arr[i], shuffled_arr[j] = shuffled_arr[j].copy(), shuffled_arr[i].copy()

    return shuffled_arr


@njit
def numba_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)


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
        roll = sumweights * random.random()
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


@njit(float64[:](float64, float64, int64))
def uniform(low: np.float64, high: np.float64, size: np.int64):
    """
    Generate an array of random samples from a uniform distribution.

    Parameters
    ----------
    low : np.float64
        The lower boundary of the output interval. All values generated will be
        greater than or equal to `low`.
    high : np.float64
        The upper boundary of the output interval. All values generated will be
        less or equal to `high`.
    size : np.int64
        The number of elements to generate.

    Returns
    -------
    NDArray[np.float64]
        An 1D array of random samples drawn from the uniform distribution.

    Examples
    --------
    >>> from thefittest.utils.random import uniform
    >>>
    >>> # Example of generating random samples
    >>> result = uniform(low=0.0, high=1.0, size=5)
    >>> print("Random Samples:", result)
    Random Samples: ...

    Notes
    -----
    The generated samples follow a uniform distribution, where each value within
    the specified range has an equal probability of being selected.

    """
    return np.random.uniform(low=low, high=high, size=size)


@njit(int64[:](int64, int64, int64))
def randint(low, high, size):
    """
    Generate an array of random integers from a discrete uniform distribution.

    Parameters
    ----------
    low : int
        The lowest integer to be drawn from the distribution.
    high : int
        The highest integer to be drawn from the distribution.
    size : int
        The number of integers to generate.

    Returns
    -------
    NDArray[int64]
        An array of random integers.

    Examples
    --------
    >>> from numba import jit
    >>> import numpy as np
    >>>
    >>> # Example of generating random integers
    >>> result = numba_randint(low=1, high=10, size=5)
    >>> print("Random Integers:", result)
    Random Integers: ...

    Notes
    -----
    The generated integers follow a discrete uniform distribution.
    """
    result = np.empty(size, dtype=np.int64)

    for i in range(size):
        result[i] = low + np.int64(np.floor((high - low) * random.random()))

    return result


def check_random_state(seed: Optional[Union[int, np.random.RandomState]] = None):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None:
        random_state = np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer)):
        random_state = np.random.RandomState(seed)
        seed = random_state.get_state()[1][0]
        numba_seed(seed)
    elif isinstance(seed, np.random.RandomState):
        random_state = seed
        seed = random_state.get_state()[1][0]
        numba_seed(seed)
    else:
        raise ValueError("%r cannot be used to seed a numpy.random.RandomState" " instance" % seed)

    return random_state
