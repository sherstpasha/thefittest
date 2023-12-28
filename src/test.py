import numpy as np
from typing import Optional
from typing import Union
from numpy.typing import NDArray

from thefittest.optimizers import GeneticAlgorithm


def numpy_bit_to_int(
    bit_array: NDArray[np.int64], powers: Optional[NDArray[np.int64]] = None
) -> NDArray[np.int64]:
    if powers is None:
        powers = 2 ** np.arange(bit_array.shape[1], dtype=np.int64)

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


class SamplingGrid_old:
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


class SamplingGrid:
    def __init__(
        self,
    ) -> None:
        self._left_border: NDArray[np.float64]
        self._right_border: NDArray[np.float64]
        self._num_variables: int
        self._h_per_variable: NDArray[np.float64]
        self._bits_per_variable: NDArray[np.int64]
        self._reversed_powers: NDArray[np.int64]

    def fit(
        self,
        left_border: Union[float, NDArray[np.float64]],
        right_border: Union[float, NDArray[np.float64]],
        num_variables: int,
        h_per_variable: Optional[Union[float, NDArray[np.float64]]] = None,
        bits_per_variable: Optional[Union[int, NDArray[np.int64]]] = None,
    ):
        assert (bits_per_variable is None) != (
            h_per_variable is None
        ), "Either bits_per_variable or h_per_variable must be defined, but not both."

        self._num_variables = num_variables
        self._left_border = self._if_single_or_array_to_float_array(left_border)
        self._right_border = self._if_single_or_array_to_float_array(right_border)

        if bits_per_variable is None:
            self._h_per_variable = self._if_single_or_array_to_float_array(h_per_variable)
            self._culc_num_bits_from_h()
            self._culc_h_from_num_bits()
        else:
            self._bits_per_variable = self._if_single_or_array_to_int_array(bits_per_variable)
            self._culc_h_from_num_bits()

        powers = 2 ** np.arange(self._bits_per_variable.max(), dtype=np.int64)
        self._reversed_powers = np.flip(powers)

        return self

    def _if_single_or_array_to_float_array(
        self, single_or_array: Union[float, int, np.number, NDArray[np.number]]
    ) -> NDArray[np.float64]:
        if isinstance(single_or_array, (int, float, np.number)):
            array = np.full(shape=self._num_variables, fill_value=single_or_array, dtype=np.float64)
        else:
            assert len(single_or_array) == self._num_variables
            array = single_or_array.astype(np.float64)

        return array

    def _if_single_or_array_to_int_array(
        self, single_or_array: Union[float, int, np.number, NDArray[np.number]]
    ) -> NDArray[np.float64]:
        if isinstance(single_or_array, (int, float, np.number)):
            array = np.full(shape=self._num_variables, fill_value=single_or_array, dtype=np.int64)
        else:
            assert len(single_or_array) == self._num_variables
            array = single_or_array.astype(np.int64)

        return array

    def _culc_h_from_num_bits(self) -> None:
        self._h_per_variable = (self._right_border - self._left_border) / (
            2.0**self._bits_per_variable - 1
        )

    def _culc_num_bits_from_h(self) -> None:
        self._bits_per_variable = np.ceil(
            np.log2((self._right_border - self._left_border) / self._h_per_variable + 1)
        ).astype(int)

    @staticmethod
    def bit_to_int(
        bit_array: NDArray[np.int64], reversed_powers: Optional[NDArray[np.int64]] = None
    ) -> NDArray[np.int64]:
        if reversed_powers is None:
            num_bits = bit_array.shape[1]
            powers = 2 ** np.arange(num_bits, dtype=np.int64)
            reversed_powers = np.flip(powers)
        int_array = np.dot(bit_array, reversed_powers)
        return int_array

    @staticmethod
    def int_to_bit(
        int_array: NDArray[np.int64], reversed_powers: Optional[NDArray[np.int64]] = None
    ) -> NDArray[np.byte]:
        num_bits = int(np.ceil(np.log2(np.max(int_array) + 1)))
        bit_array = np.empty(shape=(int_array.shape[0], num_bits), dtype=np.int8)

        if reversed_powers is None:
            powers = 2 ** np.arange(num_bits, dtype=np.int64)
            reversed_powers = np.flip(powers)

        int_array = int_array.astype(np.int64)

        for i, reversed_power_i in enumerate(reversed_powers):
            bit_array[:, i] = np.int8((int_array & reversed_power_i) > 0)
        return bit_array

    def _float_to_bit(
        self, float_array: NDArray[np.float64], left: NDArray[np.float64], h: NDArray[np.float64]
    ) -> np.int8:
        grid_number = (float_array - left) / h
        int_array = np.rint(grid_number)
        bit_array = self.int_to_bit(int_array)
        return bit_array

    def _decode(self, bit_array_i: NDArray[np.byte]) -> NDArray[np.int64]:
        int_convert = self.bit_to_int(bit_array_i, self._reversed_powers)
        return int_convert

    def transform(self, population: NDArray[np.int8]) -> NDArray[np.float64]:
        splits = np.add.accumulate(self._bits_per_variable)
        p_parts = np.split(population, splits[:-1], axis=1)

        int_array = np.array(list(map(self._decode, p_parts))).T
        float_array = (
            self._left_border[np.newaxis, :] + self._h_per_variable[np.newaxis, :] * int_array
        )
        return float_array

    def inverse_transform(self, population: NDArray[np.float64]) -> NDArray[np.int8]:
        map_ = map(self._float_to_bit, population.T, self._left_border, self._h_per_variable)
        bit_array = np.hstack(list(map_))
        return bit_array


bit_array = np.random.randint(0, 2, size=(100, 63), dtype=np.int8)


def bit_to_int(
    bit_array: NDArray[np.int64], reversed_powers: Optional[NDArray[np.int64]] = None
) -> NDArray[np.int64]:
    if reversed_powers is None:
        num_bits = bit_array.shape[1]
        powers = 2 ** np.arange(num_bits, dtype=np.int64)
        reversed_powers = np.flip(powers)

    int_array = np.dot(bit_array, reversed_powers)
    return int_array


# res1 = numpy_bit_to_int(bit_array)
# res2 = bit_to_int(bit_array)

# # print(res1)

# # print(res2)

# print(np.allclose(res1, res2))

# тестирование
# 1 каждая переменная равна и задана вручную и количество бит
# 2 каждая переменная равна и задана вручную и погрешность
# 3 каждая переменная равна и задана коротко и количество бит
# 4 каждая переменная равна и задана коротко и погрешность
# каждая переменная разная и задана вручную


# 1
n_dimension = 5
left_border_array = np.array([-5, -5, -5, -5, -5])
right_border_array = np.array([10, 10, 10, 10, 10])
parts_array = np.array([8, 8, 8, 8, 8])

sg_old = SamplingGrid_old(fit_by="parts").fit(
    left=left_border_array,
    right=right_border_array,
    arg=parts_array,
)

sg_new = SamplingGrid().fit(
    left_border=left_border_array,
    right_border=right_border_array,
    num_variables=n_dimension,
    bits_per_variable=parts_array,
)

print("left", np.allclose(sg_old.left, sg_new._left_border))
print("right", np.allclose(sg_old.right, sg_new._right_border))
print("parts", np.allclose(sg_old.parts, sg_new._bits_per_variable))
print("h", np.allclose(sg_old.h, sg_new._h_per_variable))

population = GeneticAlgorithm.binary_string_population(10, np.sum(parts_array))

population_float_old = sg_old.transform(population)
population_float_new = sg_new.transform(population)

population_bit_old = sg_old.inverse_transform(population_float_old)
population_bit_new = sg_new.inverse_transform(population_float_new)

print(np.allclose(population_float_old, population_float_new))
print(np.allclose(population_bit_old, population_bit_new))

# 2
n_dimension = 5
left_border_array = np.array([-5, -5, -5, -5, -5])
right_border_array = np.array([10, 10, 10, 10, 10])
h_array = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

sg_old = SamplingGrid_old(fit_by="h").fit(
    left=left_border_array,
    right=right_border_array,
    arg=h_array,
)

sg_new = SamplingGrid().fit(
    left_border=left_border_array,
    right_border=right_border_array,
    num_variables=n_dimension,
    h_per_variable=h_array,
)

print("left", np.allclose(sg_old.left, sg_new._left_border))
print("right", np.allclose(sg_old.right, sg_new._right_border))
print("parts", np.allclose(sg_old.parts, sg_new._bits_per_variable))
print("h", np.allclose(sg_old.h, sg_new._h_per_variable))

population = GeneticAlgorithm.binary_string_population(10, np.sum(parts_array))

population_float_old = sg_old.transform(population)
population_float_new = sg_new.transform(population)

population_bit_old = sg_old.inverse_transform(population_float_old)
population_bit_new = sg_new.inverse_transform(population_float_new)

print(np.allclose(population_float_old, population_float_new))
print(np.allclose(population_bit_old, population_bit_new))
