from numba import njit
from numba import int64
from numba import float64
from numba import boolean
import numpy as np
from numpy.typing import NDArray


@njit(int64(int64, int64[:]))
def find_end_subtree_from_i(index: np.int64, n_args_array: NDArray[np.int64]) -> np.int64:
    n_index = index + 1
    possible_steps = n_args_array[index]
    while possible_steps:
        possible_steps += n_args_array[n_index] - 1
        n_index += 1
    return n_index


@njit(int64[:](int64, int64[:]))
def find_id_args_from_i(index: np.int64, n_args_array: NDArray[np.int64]) -> NDArray[np.int64]:
    out = np.empty(n_args_array[index], dtype=np.int64)
    out[0] = index + 1
    for i in range(1, len(out)):
        next_stop = find_end_subtree_from_i(out[i - 1], n_args_array)
        out[i] = next_stop
    return out


@njit(int64[:](int64, int64[:]))
def get_levels_tree_from_i(origin: np.int64, n_args_array: NDArray[np.int64]) -> NDArray[np.int64]:
    d_i = -1
    s = [1]
    d = [-1]
    result_list = []
    for n_arg in n_args_array[origin:]:
        s[-1] = s[-1] - 1
        if s[-1] == 0:
            s.pop()
            d_i = d.pop() + 1
        else:
            d_i = d[-1] + 1
        result_list.append(d_i)
        if n_arg > 0:
            s.append(n_arg)
            d.append(d_i)
        if len(s) == 0:
            break
    return np.array(result_list, dtype=np.int64)


@njit(int64(int64[:], int64[:]))
def find_first_difference_between_two(
    array_1: NDArray[np.int64], array_2: NDArray[np.int64]
) -> np.int64:
    for i in np.arange(min(len(array_1), len(array_2)), dtype=np.int64):
        if array_1[i] != array_2[i]:
            break
    return i


@njit(int64(float64, float64[:]))
def binary_search_interval(value: np.float64, intervals: NDArray[np.float64]) -> np.int64:
    if value <= intervals[0]:
        ind = 0
    else:
        left = 0
        right = len(intervals) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if value <= intervals[mid]:
                right = mid
            else:
                left = mid
        ind = right
    return ind


@njit(boolean(int64, int64[:], int64))
def check_for_value(value: np.int64, index_array: NDArray[np.int64], end: np.int64) -> bool:
    found = False
    for i in range(end):
        if value == index_array[i]:
            found = True
            break
    return found


@njit(int64[:](float64[:], int64))
def argsort_k(array: NDArray[np.float64], k: np.int64) -> NDArray[np.int64]:
    size = len(array)
    array_copy = array.copy()
    to_return = np.arange(size, dtype=np.int64)
    for i in range(k):
        max_ = array_copy[i]
        for j in range(i, size):
            if array_copy[j] > max_:
                max_ = array_copy[j]
                max_id = j
        array_copy[i], array_copy[max_id] = array_copy[max_id], array_copy[i]
        to_return[i], to_return[max_id] = to_return[max_id], to_return[i]
    return to_return


@njit(int64[:](float64[:], float64))
def find_pbest_id(array: NDArray[np.float64], p: np.float64) -> NDArray[np.int64]:
    size = len(array)
    count = max(np.int64(1), np.int64(p * size))
    argsort = argsort_k(array, count)
    to_return = argsort[:count]
    return to_return


@njit(float64[:](float64[:, :]))
def max_axis(array):
    res = np.zeros((array.shape[1]), dtype=np.float64)
    for i in range(array.shape[1]):
        res[i] = np.max(array[:, i])
    return res
