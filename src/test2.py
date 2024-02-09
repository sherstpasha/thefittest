# from thefittest.utils.random import random_sample
import numpy as np
import numbers
from sklearn.utils._testing import assert_raises
import random 
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

from thefittest.utils import binary_search_interval
from thefittest.utils import check_for_value
from thefittest.utils import numpy_group_by
from thefittest.utils.random import random_sample, random_weighted_sample, seed


def test_check_random_state():
    """Check the check_random_state utility function behavior"""

    assert(check_random_state(None) is np.random.mtrand._rand)
    assert(check_random_state(np.random) is np.random.mtrand._rand)

    rng_42 = np.random.RandomState(42)
    assert(check_random_state(42).randint(100) == rng_42.randint(100))

    rng_42 = np.random.RandomState(42)
    assert(check_random_state(rng_42) is rng_42)

    rng_42 = np.random.RandomState(42)
    assert(check_random_state(43).randint(100) != rng_42.randint(100))

    assert_raises(ValueError, check_random_state, "some invalid seed")

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

np.random.seed(10)
weights = np.random.random(size = 100).astype(np.float64)

seed(10)

print(random_sample(100, 5, False))
print(random_sample(100, 5, False))

print(random_weighted_sample(weights, 5, False))
print(random_weighted_sample(weights, 5, False))
# # import time
# import time
# weights = np.random.random(size = 1000).astype(np.float64)

# n = 10000
# begin = time.time()
# for i in range(n):
#     random_weighted_sample2(weights, 500, False)
# print(time.time() - begin)

# begin = time.time()
# for i in range(n):
#     random_weighted_sample(weights, 500, False)
# print(time.time() - begin)