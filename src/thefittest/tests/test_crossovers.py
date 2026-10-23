import numpy as np

from ..utils import crossovers


def test_uniform_tournament_crossover_uses_winning_parent_indices(monkeypatch):
    individs = np.array(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=np.byte,
    )
    fitness = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rank = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def fixed_tournaments(range_size, quantity, replace):
        assert range_size == len(individs)
        assert quantity == 2 * individs.shape[1]
        assert replace is True
        return np.array([0, 2, 1, 2, 2, 0, 1, 0], dtype=np.int64)

    monkeypatch.setattr(crossovers, "random_sample", fixed_tournaments)

    offspring = crossovers.uniform_tournament_crossover(individs, fitness, rank)

    assert np.array_equal(offspring, np.array([30, 31, 32, 23], dtype=np.byte))
