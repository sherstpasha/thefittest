from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from numba import float64
from numba import njit

import numpy as np
from numpy.typing import NDArray

from thefittest.optimizers import GeneticAlgorithm


def numpy_bit_to_int(
    bit_array: NDArray[np.int64], powers: Optional[NDArray[np.int64]] = None
) -> NDArray[np.int64]:
    if powers is None:
        powers = 2 ** np.arange(bit_array.shape[1], dtype=np.byte)
    arange_ = powers[: bit_array.shape[1]][::-1]
    int_array = np.sum(bit_array * arange_, axis=1)
    return int_array


def numpy_int_to_bit(int_array: NDArray[np.int64]) -> NDArray[np.byte]:
    result = []
    bit = int_array % 2
    remains = int_array // 2
    result.append(bit)
    while np.sum(remains) > 0:
        bit = remains % 2
        remains = remains // 2
        result.append(bit)
    bit_array = np.array(result, dtype=np.int8)[::-1].T
    return bit_array


def numpy_gray_to_bit(gray_array: NDArray[np.byte]) -> NDArray[np.byte]:
    bit_array = np.logical_xor.accumulate(gray_array, axis=-1).astype(np.byte)
    return bit_array


def numpy_bit_to_gray(bit_array: NDArray[np.byte]) -> NDArray[np.byte]:
    cut_gray = np.logical_xor(bit_array[:, :-1], bit_array[:, 1:])
    gray_array = np.hstack([bit_array[:, 0].reshape(-1, 1), cut_gray])
    return gray_array


class SamplingGrid:
    def __init__(self, fit_by: str = "h") -> None:
        self._fit_by = fit_by
        self.left: NDArray[np.float64]
        self.right: NDArray[np.float64]
        self.parts: NDArray[np.int64]
        self.h: NDArray[np.float64]
        self._power_arange: NDArray[np.int64]

    def _culc_h_from_parts(
        self, left: NDArray[np.float64], right: NDArray[np.float64], parts: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        h = (right - left) / (2.0**parts - 1)
        return h

    def _culc_parts_from_h(
        self, left: NDArray[np.float64], right: NDArray[np.float64], h: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        parts = np.ceil(np.log2((right - left) / h + 1)).astype(int)
        return parts

    def _decode(self, bit_array_i: NDArray[np.byte]) -> NDArray[np.int64]:
        int_convert = numpy_bit_to_int(bit_array_i, self._power_arange)
        return int_convert

    def fit(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        arg: Union[NDArray[np.float64], NDArray[np.int64]],
    ):
        self.left = left
        self.right = right

        assert self._fit_by in ["h", "parts"], f"incorrect option {self._fit_by} for fit_by."
        "The available ones are 'h' and 'parts'"
        if self._fit_by == "h":
            min_h = arg
            self.parts = self._culc_parts_from_h(left, right, min_h)
            self.h = self._culc_h_from_parts(left, right, self.parts)
        else:
            self.parts = arg
            self.h = self._culc_h_from_parts(left, right, self.parts)

        self._power_arange = 2 ** np.arange(self.parts.max(), dtype=np.int64)
        return self

    def transform(self, population: np.ndarray) -> np.ndarray:
        splits = np.add.accumulate(self.parts)
        p_parts = np.split(population, splits[:-1], axis=1)

        int_array = np.array(list(map(self._decode, p_parts))).T
        float_array = self.left[np.newaxis, :] + self.h[np.newaxis, :] * int_array
        return float_array

    def _float_to_bit(self, float_array: np.ndarray, left: np.ndarray, h: np.ndarray) -> np.ndarray:
        grid_number = (float_array - left) / h
        int_array = np.rint(grid_number)
        bit_array = numpy_int_to_bit(int_array)
        return bit_array

    def inverse_transform(self, population: np.ndarray) -> np.ndarray:
        map_ = map(self._float_to_bit, population.T, self.left, self.h)
        bit_array = np.hstack(list(map_))
        return bit_array


class GrayCode(SamplingGrid):
    def __init__(self, fit_by: str = "h") -> None:
        SamplingGrid.__init__(self, fit_by)

    def _decode(self, gray_array_i: np.ndarray) -> np.ndarray:
        bit_array_i = numpy_gray_to_bit(gray_array_i)
        int_convert = numpy_bit_to_int(bit_array_i, self._power_arange)
        return int_convert

    def _float_to_bit(self, float_array: np.ndarray, left: np.ndarray, h: np.ndarray) -> np.ndarray:
        grid_number = (float_array - left) / h
        int_array = np.rint(grid_number)
        bit_array = numpy_int_to_bit(int_array)
        gray_array = numpy_bit_to_gray(bit_array)
        return gray_array


import numpy as np
from thefittest.benchmarks import Sphere


n_dimension = 10
left_border = -5.0
right_border = 5.0
n_bits_per_variable = 32

number_of_iterations = 300
population_size = 500

left_border_array = np.full(shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(shape=n_dimension, fill_value=right_border, dtype=np.float64)
parts = np.full(shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

genotype_to_phenotype = GrayCode(fit_by="parts").fit(
    left=left_border_array, right=right_border_array, arg=parts
)
optimizer = GeneticAlgorithm(
    fitness_function=Sphere(),
    genotype_to_phenotype=genotype_to_phenotype.transform,
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=sum(parts),
    show_progress_each=30,
    minimization=True,
    selection="rank",
    crossover="two_point",
    mutation="strong",
    tour_size=6,
    optimal_value=0.0,
)

optimizer.fit()

fittest = optimizer.get_fittest()

print("The fittest individ:", fittest["genotype"])
print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])
