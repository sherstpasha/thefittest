import numpy as np
import pytest

from ..utils.transformations import GrayCode
from ..utils.transformations import SamplingGrid
from ..utils.transformations import minmax_scale


def test_minmax_scale_contract():
    values = np.array([2, 5, 10, 8, 3], dtype=np.int64)
    scaled = minmax_scale(values)

    assert np.allclose(scaled, np.array([0.0, 0.375, 1.0, 0.75, 0.125]))
    assert scaled.dtype == np.float64
    assert np.array_equal(values, np.array([2, 5, 10, 8, 3], dtype=np.int64))
    assert np.array_equal(minmax_scale(np.array([3, 3], dtype=np.int64)), np.ones(2))


def test_sampling_grid_fit_and_accessors():
    grid = SamplingGrid().fit(
        left_border=np.array([-1.0, 0.0], dtype=np.float64),
        right_border=np.array([1.0, 3.0], dtype=np.float64),
        num_variables=2,
        bits_per_variable=np.array([2, 3], dtype=np.int64),
    )

    assert np.array_equal(grid.get_left_border(), np.array([-1.0, 0.0]))
    assert np.array_equal(grid.get_right_border(), np.array([1.0, 3.0]))
    assert grid.get_num_variables() == 2
    assert np.array_equal(grid.get_bits_per_variable(), np.array([2, 3]))
    assert grid.get_str_len() == 5
    assert np.allclose(grid.get_h_per_variable(), np.array([2 / 3, 3 / 7]))


def test_sampling_grid_fit_requires_one_resolution_argument():
    grid = SamplingGrid()

    with pytest.raises(ValueError):
        grid.fit(0.0, 1.0, num_variables=1)
    with pytest.raises(ValueError):
        grid.fit(0.0, 1.0, num_variables=1, h_per_variable=0.1, bits_per_variable=4)


def test_sampling_grid_bit_integer_conversions_keep_requested_width():
    integers = np.array([0, 1, 5], dtype=np.int64)
    powers = np.array([1, 2, 4, 8], dtype=np.int64)

    bits = SamplingGrid.int_to_bit(integers, powers)
    converted = SamplingGrid.bit_to_int(bits, powers)

    assert bits.shape == (3, 4)
    assert np.array_equal(bits[0], np.array([0, 0, 0, 0], dtype=np.int8))
    assert np.array_equal(converted, integers)


def test_sampling_grid_inverse_transform_uses_fixed_bit_width_for_low_values():
    grid = SamplingGrid().fit(0.0, 1.0, num_variables=2, bits_per_variable=4)
    population = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)

    encoded = grid.inverse_transform(population)
    decoded = grid.transform(encoded)

    assert encoded.shape == (2, 8)
    assert np.array_equal(encoded[0], np.zeros(8, dtype=np.int8))
    assert np.allclose(decoded, population)


def test_gray_code_conversions_and_inverse_transform_width():
    binary = np.array([[0, 0, 0, 0], [1, 0, 1, 1]], dtype=np.byte)
    gray = GrayCode.bit_to_gray(binary)

    assert gray.dtype == np.byte
    assert np.array_equal(GrayCode.gray_to_bit(gray), binary)

    grid = GrayCode().fit(0.0, 1.0, num_variables=2, bits_per_variable=4)
    population = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    encoded = grid.inverse_transform(population)
    decoded = grid.transform(encoded)

    assert encoded.shape == (2, 8)
    assert np.allclose(decoded, population)
