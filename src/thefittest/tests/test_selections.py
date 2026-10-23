import numpy as np

from ..utils import selections


def test_selection_dispatchers_return_index_arrays():
    fitness = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
    rank = np.array([3.0, 2.0, 4.0, 1.0], dtype=np.float64)

    cases = (
        (selections.proportional_selection, 0),
        (selections.rank_selection, 0),
        (selections.tournament_selection, 2),
    )

    for selection, tour_size in cases:
        selected = selection(fitness, rank, tour_size, 3)

        assert selected.shape == (3,)
        assert selected.dtype == np.int64
        assert np.all(0 <= selected)
        assert np.all(selected < fitness.size)


def test_proportional_selection_uses_fitness_weights(monkeypatch):
    fitness = np.array([0.8, 0.5, 0.9], dtype=np.float64)
    rank = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    calls = []

    def fixed_random_weighted_sample(weights, quantity, replace):
        calls.append((weights, quantity, replace))
        return np.array([2, 0], dtype=np.int64)

    monkeypatch.setattr(selections, "random_weighted_sample", fixed_random_weighted_sample)

    selected = selections.proportional_selection.py_func(fitness, rank, 7, 2)

    assert np.array_equal(selected, np.array([2, 0], dtype=np.int64))
    assert calls[0][0] is fitness
    assert calls[0][1:] == (2, True)


def test_rank_selection_uses_rank_weights(monkeypatch):
    fitness = np.array([0.8, 0.5, 0.9], dtype=np.float64)
    rank = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    calls = []

    def fixed_random_weighted_sample(weights, quantity, replace):
        calls.append((weights, quantity, replace))
        return np.array([1, 2], dtype=np.int64)

    monkeypatch.setattr(selections, "random_weighted_sample", fixed_random_weighted_sample)

    selected = selections.rank_selection.py_func(fitness, rank, 7, 2)

    assert np.array_equal(selected, np.array([1, 2], dtype=np.int64))
    assert calls[0][0] is rank
    assert calls[0][1:] == (2, True)


def test_tournament_selection_uses_highest_fitness(monkeypatch):
    fitness = np.array([0.1, 0.9, 0.4], dtype=np.float64)
    rank = np.array([100.0, 1.0, 50.0], dtype=np.float64)
    tournaments = [
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]
    calls = []

    def fixed_random_sample(range_size, quantity, replace):
        calls.append((range_size, quantity, replace))
        return tournaments[len(calls) - 1]

    monkeypatch.setattr(selections, "random_sample", fixed_random_sample)

    selected = selections.tournament_selection.py_func(fitness, rank, 2, 2)

    assert np.array_equal(selected, np.array([2, 1], dtype=np.int64))
    assert calls == [(len(fitness), 2, False), (len(fitness), 2, False)]
