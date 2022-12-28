import numpy as np


def rank_data(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    raw_ranks = np.zeros(shape=(arr.shape[0]))
    argsort = np.argsort(arr)

    s = (arr[:, np.newaxis] == arr).astype(int)
    raw_ranks[argsort] = np.arange(arr.shape[0]) + 1
    ranks = np.sum(raw_ranks*s, axis=1)/s.sum(axis=0)
    return ranks


def protect_norm(x: np.ndarray) -> np.ndarray:
    sum_ = x.sum()
    if sum_ > 0:
        return x/sum_
    else:
        len_ = len(x)
        return np.full(len_, 1/len_)


def scale_data(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    max_ = arr.max()
    min_ = arr.min()
    if max_ == min_:
        arr_n = np.ones_like(arr)
    else:
        arr_n = (arr - min_)/(max_ - min_)
    return arr_n


def numpy_bit_to_int(bit_array, powers=None):
    if powers is None:
        powers = 2**np.arange(bit_array.shape[1], dtype=np.int64)
    arange_ = powers[:bit_array.shape[1]][::-1]
    int_array = np.sum(bit_array*arange_, axis=1)
    return int_array


def numpy_int_to_bit(int_array):
    result = []
    bit = int_array % 2
    remains = int_array//2
    result.append(bit)
    while np.sum(remains) > 0:
        bit = remains % 2
        remains = remains//2
        result.append(bit)

    return np.array(result)[::-1].T


def numpy_gray_to_bit(gray_array):
    bit_array = np.logical_xor.accumulate(
        gray_array, axis=-1).astype(np.byte)
    return bit_array


def numpy_bit_to_gray(bit_array):
    cut_gray = np.logical_xor(bit_array[:, :-1],
                              bit_array[:, 1:])
    gray_array = np.hstack(
        [bit_array[:, 0].reshape(-1, 1), cut_gray])
    return gray_array


class SamplingGrid:

    def __init__(self, fit_by: str = 'h') -> None:
        self.fit_by = fit_by
        self.left: np.ndarray
        self.right: np.ndarray
        self.parts: np.ndarray
        self.h: np.ndarray
        self.power_arange: np.ndarray

    def culc_h_from_parts(self, left: np.ndarray, right: np.ndarray,
                          parts: np.ndarray) -> np.ndarray:
        return (right - left)/(2.0**parts - 1)

    def culc_parts_from_h(self, left: np.ndarray, right: np.ndarray,
                          h: np.ndarray) -> np.ndarray:
        return np.ceil(np.log2((right - left)/h + 1)).astype(int)

    def decode(self, bit_array_i):
        int_convert = numpy_bit_to_int(bit_array_i, self.power_arange)
        return int_convert

    def fit(self, left: np.ndarray, right: np.ndarray,
            arg: np.ndarray):
        self.left = left
        self.right = right

        assert self.fit_by in [
            'h', 'parts'], f"incorrect option {self.fit_by} for fit_by. The available ones are 'h' and 'parts'"
        if self.fit_by == 'h':
            min_h = arg
            self.parts = self.culc_parts_from_h(left, right, min_h)
            self.h = self.culc_h_from_parts(left, right, self.parts)
        else:
            self.parts = arg
            self.h = self.culc_h_from_parts(left, right, self.parts)

        self.power_arange = 2**np.arange(self.parts.max(), dtype=np.int64)
        return self

    def transform(self, population: np.ndarray) -> np.ndarray:
        splits = np.add.accumulate(self.parts)
        p_parts = np.split(population, splits[:-1], axis=1)

        int_array = np.array(list(map(self.decode, p_parts))).T
        float_array = self.left[np.newaxis, :] +\
            self.h[np.newaxis, :]*int_array

        return float_array

    def float_to_bit(self, float_array, left, h):
        grid_number = (float_array - left)/h
        int_array = np.rint(grid_number)
        return numpy_int_to_bit(int_array)

    def inverse_transform(self, population: np.ndarray) -> np.ndarray:
        map_ = map(self.float_to_bit, population.T, self.left, self.h)
        return np.hstack(list(map_))


class GrayCode(SamplingGrid):

    def __init__(self, fit_by: str = 'h') -> None:
        SamplingGrid.__init__(self, fit_by)

    def decode(self, gray_array_i):
        bit_array_i = numpy_gray_to_bit(gray_array_i)
        int_convert = numpy_bit_to_int(bit_array_i, self.power_arange)
        return int_convert

    def float_to_bit(self, float_array, left, h):
        grid_number = (float_array - left)/h
        int_array = np.rint(grid_number)
        bit_array = numpy_int_to_bit(int_array)
        return numpy_bit_to_gray(bit_array)
