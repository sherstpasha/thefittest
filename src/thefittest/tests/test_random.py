import numpy as np
import pytest

from ..utils import random as random_utils


def test_numba_seed_makes_random_sample_reproducible():
    random_utils.numba_seed(123)
    first = random_utils.random_sample(10, 5, True)

    random_utils.numba_seed(123)
    second = random_utils.random_sample(10, 5, True)

    assert np.array_equal(first, second)


def test_random_sample_without_replacement_returns_unique_indexes():
    sample = random_utils.random_sample(7, 7, False)

    assert sample.dtype == np.int64
    assert sample.shape == (7,)
    assert np.array_equal(np.sort(sample), np.arange(7, dtype=np.int64))


def test_random_weighted_sample_respects_zero_weights():
    weights = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sample = random_utils.random_weighted_sample(weights, 5, True)

    assert np.array_equal(sample, np.full(5, 2, dtype=np.int64))


def test_sattolo_shuffle_returns_copy_and_permutation():
    arr = np.arange(1, 6, dtype=np.int64)

    shuffled = random_utils.sattolo_shuffle(arr)

    assert np.array_equal(arr, np.arange(1, 6, dtype=np.int64))
    assert np.array_equal(np.sort(shuffled), arr)
    assert np.all(shuffled != arr)


def test_sattolo_shuffle_2d_returns_row_permutation_copy():
    arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)

    shuffled = random_utils.sattolo_shuffle_2d(arr)

    assert np.array_equal(arr, np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64))
    assert sorted(map(tuple, shuffled.tolist())) == sorted(map(tuple, arr.tolist()))
    assert all(not np.array_equal(shuffled[i], arr[i]) for i in range(len(arr)))


def test_uniform_and_randint_ranges():
    floats = random_utils.uniform(2.0, 5.0, 100)
    ints = random_utils.randint(2, 5, 100)

    assert np.all(2.0 <= floats)
    assert np.all(floats < 5.0)
    assert np.all(2 <= ints)
    assert np.all(ints < 5)


def test_check_random_state_contract():
    state = random_utils.check_random_state(1)
    same_state = random_utils.check_random_state(state)

    assert isinstance(state, np.random.RandomState)
    assert same_state is state
    with pytest.raises(ValueError):
        random_utils.check_random_state("bad-seed")


def test_default_generators_return_expected_ranges():
    value_1 = random_utils.generator1()
    value_2 = random_utils.generator2()

    assert isinstance(value_1, float)
    assert 0 <= value_1 < 10
    assert isinstance(value_2, np.integer)
    assert 0 <= value_2 < 10
