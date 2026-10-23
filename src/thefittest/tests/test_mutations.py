import inspect

import numpy as np

from ..utils import mutations


def test_flip_mutation_probability_extremes():
    individual = np.array([0, 1, 1, 0], dtype=np.byte)

    unchanged = mutations.flip_mutation(individual, 0.0)
    flipped = mutations.flip_mutation(individual, 1.0)

    assert np.array_equal(unchanged, individual)
    assert unchanged is not individual
    assert np.array_equal(flipped, np.array([1, 0, 0, 1], dtype=np.byte))


def test_differential_evolution_mutation_formulas(monkeypatch):
    current = np.array([1.0, 2.0], dtype=np.float64)
    best = np.array([10.0, 20.0], dtype=np.float64)
    population = np.array(
        [
            [1.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [8.0, 13.0],
            [21.0, 34.0],
        ],
        dtype=np.float64,
    )
    scale = 0.5

    def fixed_random_sample(range_size, quantity, replace):
        assert range_size == len(population)
        assert replace is False
        return np.arange(quantity, dtype=np.int64)

    monkeypatch.setattr(mutations, "random_sample", fixed_random_sample)

    cases = (
        (mutations.best_1, best + scale * (population[0] - population[1])),
        (mutations.rand_1, population[0] + scale * (population[1] - population[2])),
        (
            mutations.rand_to_best1,
            population[0] + scale * (best - population[0]) + scale * (population[1] - population[2]),
        ),
        (
            mutations.current_to_best_1,
            current + scale * (best - current) + scale * (population[0] - population[1]),
        ),
        (
            mutations.best_2,
            best + scale * (population[0] - population[1]) + scale * (population[2] - population[3]),
        ),
        (
            mutations.rand_2,
            population[0] + scale * (population[1] - population[2]) + scale * (population[3] - population[4]),
        ),
        (
            mutations.current_to_rand_1,
            current + scale * (population[0] - current) + scale * (population[1] - population[2]),
        ),
    )

    for mutation, expected in cases:
        mutated = mutation.py_func(current, best, population, scale)
        assert np.allclose(mutated, expected)


def test_current_to_pbest_1_archive_formula(monkeypatch):
    current = np.array([1.0, 2.0], dtype=np.float64)
    population = np.array(
        [
            [1.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float64,
    )
    pbest = np.array([2], dtype=np.int64)
    archive = np.array([[8.0, 13.0]], dtype=np.float64)
    scale = 0.5
    sample_calls = []

    def fixed_randint(low, high, size):
        assert (low, high, size) == (0, len(pbest), 1)
        return np.array([0], dtype=np.int64)

    def fixed_random_sample(range_size, quantity, replace):
        sample_calls.append((range_size, quantity, replace))
        if range_size == len(population):
            return np.array([1], dtype=np.int64)
        return np.array([0], dtype=np.int64)

    monkeypatch.setattr(mutations, "randint", fixed_randint)
    monkeypatch.setattr(mutations, "random_sample", fixed_random_sample)

    mutated = mutations.current_to_pbest_1_archive.py_func(current, population, pbest, scale, archive)
    expected = current + scale * (population[2] - current) + scale * (population[1] - archive[0])

    assert np.allclose(mutated, expected)
    assert sample_calls == [(len(population), 1, True), (len(archive), 1, True)]


def test_swap_mutation_uses_documented_max_level_parameter_name():
    assert "max_level" in inspect.signature(mutations.swap_mutation).parameters
