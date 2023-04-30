from typing import List
from typing import Tuple
from typing import Any
from typing import Optional
import numpy as np
from .numba_funcs import find_first_difference_between_two
from .numba_funcs import find_end_subtree_from_i
from numba import njit

def root_mean_square_error(y_true: np.ndarray,
                           y_predict: np.ndarray) -> float:
    error = y_true - y_predict
    return np.sqrt(np.mean((error)**2))


def coefficient_determination(y_true: np.ndarray,
                              y_predict: np.ndarray) -> float:
    error = y_true - y_predict
    mean_y_true = np.mean(y_true)
    residual_sum = np.sum((error)**2)
    total_sum = np.sum((y_true - mean_y_true)**2)
    return 1 - residual_sum/total_sum


def categorical_crossentropy(target: np.ndarray,
                             output: np.ndarray):
    output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.mean(np.sum(target * -np.log(output),
                          axis=-1, keepdims=False))


def common_region(trees: List) -> Tuple:
    terminate = False
    indexes = []
    common_indexes = []
    border_indexes = []
    for tree in trees:
        indexes.append(list(range(len(tree._nodes))))
        common_indexes.append([])
        border_indexes.append([])

    while not terminate:
        inner_break = False
        iters = min(list(map(len, indexes)))

        for i in range(iters):
            first_n_args = trees[0]._nodes[indexes[0][i]]._n_args
            common_indexes[0].append(indexes[0][i])
            for j in range(1, len(indexes)):
                common_indexes[j].append(indexes[j][i])
                if first_n_args != trees[j]._nodes[indexes[j][i]]._n_args:
                    inner_break = True

            if inner_break:
                for j in range(0, len(indexes)):
                    border_indexes[j].append(indexes[j][i])
                break

        for j in range(len(indexes)):
            _, right = trees[j].subtree(common_indexes[j][-1])
            delete_to = indexes[j].index(right-1) + 1
            indexes[j] = indexes[j][delete_to:]

            if len(indexes[j]) < 1:
                terminate = True
                break

    return common_indexes, border_indexes


def common_region_two_trees(n_args_array_1: np.ndarray,
                            n_args_array_2: np.ndarray) -> Tuple:
    index_list_1 = range(len(n_args_array_1))
    index_list_2 = range(len(n_args_array_2))
    index_1 = 0
    index_2 = 0

    common_1 = []
    common_2 = []
    border_1 = []
    border_2 = []

    while True:
        if index_1 < len(n_args_array_1) and index_2 < len(n_args_array_2):
            id_1 = index_1
            id_2 = index_2
            end = find_first_difference_between_two(n_args_array_1[id_1:],
                                                    n_args_array_2[id_2:])
            index_1 = index_1 + end
            index_2 = index_2 + end
            common_1.extend(index_list_1[id_1:index_1+1])
            common_2.extend(index_list_2[id_2:index_2+1])

        if len(n_args_array_1)-1 > index_1 or len(n_args_array_2)-1 > index_2:
            border_1.append(index_1)
            border_2.append(index_2)
            index_1 = find_end_subtree_from_i(index_1, n_args_array_1)
            index_2 = find_end_subtree_from_i(index_2, n_args_array_2)
        else:
            break

    return [common_1, common_2], [border_1, border_2]


def common_region_(trees: List) -> Tuple:
    if len(trees) == 2:
        to_return = common_region_two_trees(trees[0]._n_args, trees[1]._n_args)
    else:
        to_return = common_region_(trees)

    return to_return


def donothing(x: Any) -> Any:
    return x


def numpy_group_by(group: np.ndarray,
                   by: np.ndarray) -> Tuple:

    argsort = np.argsort(by)
    group = group[argsort]
    by = by[argsort]

    keys, cut_index = np.unique(by, return_index=True)
    groups = np.split(group, cut_index)[1:]
    return keys, groups


def lehmer_mean(x: np.ndarray,
                power: int = 2,
                weight: Optional[np.ndarray] = None) -> float:
    if weight is None:
        weight = 1
    x_up = weight*np.power(x, power)
    x_down = weight*np.power(x, power-1)
    return np.sum(x_up)/np.sum(x_down)


def rank_data(arr: np.ndarray) -> np.ndarray:
    arange = np.arange(len(arr), dtype=int)

    argsort = np.argsort(arr)
    arr_sorted = arr.copy()[argsort]

    cond = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
    raw_ranks = np.r_[arange[cond == True], len(arange)]
    ranks = (raw_ranks[1:] + raw_ranks[:-1] + 1)/2
    count = raw_ranks[1:] - raw_ranks[:-1]

    retults = np.empty_like(arr, dtype=np.float64)
    retults[argsort] = ranks.repeat(count)
    return retults


@njit
def protect_norm(x: np.ndarray) -> np.ndarray:
    result = np.empty(len(x), dtype=np.float64)
    sum_ = x.sum()
    if sum_ > 0:
        for i in range(result.size):
            result[i] = x[i]/sum_
    else:
        value = 1/len(x)
        for i in range(result.size):
            result[i] = value
    return result


def scale_data(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    max_ = arr.max()
    min_ = arr.min()
    if max_ == min_:
        to_return = np.ones_like(arr, dtype = np.float64)
    else:
        to_return = ((arr - min_)/(max_ - min_)).astype(np.float64)
    return to_return


def numpy_bit_to_int(bit_array: np.ndarray,
                     powers: np.ndarray = None) -> np.ndarray:
    if powers is None:
        powers = 2**np.arange(bit_array.shape[1], dtype=np.int64)
    arange_ = powers[:bit_array.shape[1]][::-1]
    int_array = np.sum(bit_array*arange_, axis=1)
    return int_array


def numpy_int_to_bit(int_array: np.ndarray) -> np.ndarray:
    result = []
    bit = int_array % 2
    remains = int_array//2
    result.append(bit)
    while np.sum(remains) > 0:
        bit = remains % 2
        remains = remains//2
        result.append(bit)
    bit_array = np.array(result)[::-1].T
    return bit_array


def numpy_gray_to_bit(gray_array: np.ndarray) -> np.ndarray:
    bit_array = np.logical_xor.accumulate(
        gray_array, axis=-1).astype(np.byte)
    return bit_array


def numpy_bit_to_gray(bit_array: np.ndarray) -> np.ndarray:
    cut_gray = np.logical_xor(bit_array[:, :-1],
                              bit_array[:, 1:])
    gray_array = np.hstack(
        [bit_array[:, 0].reshape(-1, 1), cut_gray])
    return gray_array


class SamplingGrid:
    def __init__(self,
                 fit_by: str = 'h') -> None:
        self._fit_by = fit_by
        self.left: np.ndarray
        self.right: np.ndarray
        self.parts: np.ndarray
        self.h: np.ndarray
        self._power_arange: np.ndarray

    def _culc_h_from_parts(self,
                           left: np.ndarray,
                           right: np.ndarray,
                           parts: np.ndarray) -> np.ndarray:
        h = (right - left)/(2.0**parts - 1)
        return h

    def _culc_parts_from_h(self,
                           left: np.ndarray,
                           right: np.ndarray,
                           h: np.ndarray) -> np.ndarray:
        parts = np.ceil(np.log2((right - left)/h + 1)).astype(int)
        return parts

    def _decode(self,
                bit_array_i: np.ndarray) -> np.ndarray:
        int_convert = numpy_bit_to_int(bit_array_i, self._power_arange)
        return int_convert

    def fit(self,
            left: np.ndarray,
            right: np.ndarray,
            arg: np.ndarray):
        self.left = left
        self.right = right

        assert self._fit_by in [
            'h', 'parts'], f"incorrect option {self._fit_by} for fit_by."
        "The available ones are 'h' and 'parts'"
        if self._fit_by == 'h':
            min_h = arg
            self.parts = self._culc_parts_from_h(left, right, min_h)
            self.h = self._culc_h_from_parts(left, right, self.parts)
        else:
            self.parts = arg
            self.h = self._culc_h_from_parts(left, right, self.parts)

        self._power_arange = 2**np.arange(self.parts.max(), dtype=np.int64)
        return self

    def transform(self,
                  population: np.ndarray) -> np.ndarray:
        splits = np.add.accumulate(self.parts)
        p_parts = np.split(population, splits[:-1], axis=1)

        int_array = np.array(list(map(self._decode, p_parts))).T
        float_array = self.left[np.newaxis, :] +\
            self.h[np.newaxis, :]*int_array
        return float_array

    def _float_to_bit(self,
                      float_array: np.ndarray,
                      left: np.ndarray,
                      h: np.ndarray) -> np.ndarray:
        grid_number = (float_array - left)/h
        int_array = np.rint(grid_number)
        bit_array = numpy_int_to_bit(int_array)
        return bit_array

    def inverse_transform(self,
                          population: np.ndarray) -> np.ndarray:
        map_ = map(self._float_to_bit, population.T, self.left, self.h)
        bit_array = np.hstack(list(map_))
        return bit_array


class GrayCode(SamplingGrid):

    def __init__(self,
                 fit_by: str = 'h') -> None:
        SamplingGrid.__init__(self, fit_by)

    def _decode(self,
                gray_array_i: np.ndarray) -> np.ndarray:
        bit_array_i = numpy_gray_to_bit(gray_array_i)
        int_convert = numpy_bit_to_int(bit_array_i, self._power_arange)
        return int_convert

    def _float_to_bit(self,
                      float_array: np.ndarray,
                      left: np.ndarray,
                      h: np.ndarray) -> np.ndarray:
        grid_number = (float_array - left)/h
        int_array = np.rint(grid_number)
        bit_array = numpy_int_to_bit(int_array)
        gray_array = numpy_bit_to_gray(bit_array)
        return gray_array
