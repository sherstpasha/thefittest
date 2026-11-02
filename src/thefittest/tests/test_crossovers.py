import numpy as np
import pytest

from ..utils import crossovers


@pytest.mark.parametrize(
    "ranks, distances",
    [([1, 2], [0.0, np.inf]), ([1, 1], [np.inf, 0.0])],
)
def test_rank_crowding_crossover_prefers_rank_then_distance(
    ranks: list[int], distances: list[float]
) -> None:
    parents = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.byte)
    original = parents.copy()
    offspring = crossovers.uniform_rank_crowding_crossover(
        parents, np.array(distances), np.array(ranks, dtype=np.int64)
    )
    np.testing.assert_array_equal(offspring, parents[0])
    np.testing.assert_array_equal(parents, original)
    assert offspring.dtype == np.byte
    assert not np.shares_memory(offspring, parents)


def test_rank_crowding_crossover_resolves_ties_randomly() -> None:
    parents = np.vstack((np.zeros(256), np.ones(256))).astype(np.byte)
    offspring = crossovers.uniform_rank_crowding_crossover(
        parents, np.array([np.inf, np.inf]), np.ones(2, dtype=np.int64)
    )
    assert set(offspring) == {0, 1}


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
