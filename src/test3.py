import numpy as np
from typing import Tuple
from typing import List
from typing import Union

from numba import boolean
from numba import float64
from numba import int64
from numba import njit
from numba.typed import Dict as numbaDict
from numba.types import List as numbaListType

import numpy as np
from numpy.typing import NDArray


from scipy.special import softmax


np.random.seed(42)

input_data = np.random.randn(10, 10)


@njit(float64[:](float64[:, :]))
def max_axis(array: NDArray[np.float64]) -> NDArray[np.float64]:
    res = np.zeros((array.shape[0]), dtype=np.float64)
    for i in range(array.shape[0]):
        res[i] = np.max(array[i])
    return res


@njit(float64[:, :](float64[:, :]))
def softmax_numba1(X: NDArray[np.float64]) -> NDArray[np.float64]:
    exps = np.exp(X - max_axis(X))
    # exps = np.exp(X)
    print("exps", exps)
    print("max_axis(X)", max_axis(X))

    sum_ = np.sum(exps, axis=1)
    for j in range(sum_.shape[0]):
        if sum_[j] == 0:
            sum_[j] = 1

    result = exps / sum_

    return result


print(input_data)
result_1 = softmax_numba1(input_data)
result_2 = softmax_numba1(input_data[0].reshape(1, -1))
