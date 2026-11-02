import numpy as np
from numpy.typing import NDArray
import pytest

from ..optimizers import NSGAII


@pytest.mark.parametrize("crossover_rate", [0.0, 1.0])
def test_crossover_probability_can_skip_operator(crossover_rate: float) -> None:
    optimizer = NSGAII(
        lambda population: population.astype(float),
        1,
        2,
        2,
        (False, False),
        init_population=np.array([[1, 1], [0, 0]], dtype=np.byte),
        crossover="one_point",
        crossover_rate=crossover_rate,
        mutation="custom_rate",
        mutation_rate=0.0,
    )
    optimizer._get_init_population()
    optimizer._from_population_g_to_fitness()
    calls = []

    def crossover(parents: NDArray, fitness: NDArray, rank: NDArray) -> NDArray[np.byte]:
        calls.append(True)
        return np.zeros(2, dtype=np.byte)

    optimizer._crossover_pool["one_point"] = (crossover, 2)
    result = optimizer._get_new_individ_g("dominance_crowding", "one_point", "custom_rate")
    np.testing.assert_array_equal(result, [0, 0] if crossover_rate == 1.0 else [1, 1])
    assert len(calls) == int(crossover_rate)


@pytest.mark.parametrize("selection", ["rank_crowding", "dominance_crowding"])
def test_new_individual_uses_selected_parents_and_scaled_mutation(selection: str) -> None:
    def forbidden_fitness(population: NDArray[np.byte]) -> NDArray[np.float64]:
        raise AssertionError("offspring generation must not evaluate fitness")

    optimizer = NSGAII(forbidden_fitness, 10, 3, 4, (False, False), selection=selection)
    population = np.array([[0, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.byte)
    optimizer._population_g_i = population.copy()
    optimizer._fitness_i = np.array([[1.0, 3.0], [0.0, 0.0], [3.0, 1.0]])
    optimizer._pareto_rank_i = np.array([1, 2, 1], dtype=np.int64)
    optimizer._crowding_distance_i = np.array([np.inf, np.inf, np.inf])
    calls = []

    def select(
        metadata: NDArray, distances: NDArray[np.float64], tour_size: int, quantity: int
    ) -> NDArray[np.int64]:
        expected = (
            optimizer._pareto_rank_i if selection == "rank_crowding" else optimizer._fitness_i
        )
        assert metadata is expected
        assert distances is optimizer._crowding_distance_i
        assert tour_size == quantity == 2
        calls.append("selection")
        return np.array([0, 2], dtype=np.int64)

    def crossover(
        parents: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
    ) -> NDArray[np.byte]:
        np.testing.assert_array_equal(parents, population[[0, 2]])
        calls.append("crossover")
        return parents[0].copy()

    def mutate(individual: NDArray[np.byte], proba: float) -> NDArray[np.byte]:
        assert proba == 0.25
        calls.append("mutation")
        return 1 - individual

    optimizer._selection_pool[selection] = (select, 2)
    optimizer._crossover_pool["one_point"] = (crossover, 2)
    optimizer._mutation_pool["average"] = (mutate, 1, False)
    result = optimizer._get_new_individ_g(selection, "one_point", "average")
    np.testing.assert_array_equal(result, 1 - population[0])
    np.testing.assert_array_equal(optimizer._population_g_i, population)
    assert calls == ["selection", "crossover", "mutation"]
    assert optimizer.get_calls() == 0


@pytest.mark.parametrize(
    "values, parent_ids, child_ids, expected_ids",
    [
        ([[0, 10], [2, 8], [5, 5], [9, 1], [10, 0], [3, 7]], [0, 2, 5], [1, 3, 4], {0, 2, 4}),
        ([[5, 5], [2, 2], [0, 0], [4, 1], [1, 4], [-1, -1]], [0, 1, 2], [3, 4, 5], {0, 3, 4}),
    ],
)
def test_new_population_selects_from_union_and_preserves_evaluation_and_distances(
    values: list,
    parent_ids: list,
    child_ids: list,
    expected_ids: set,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup = np.array(values, dtype=np.float64)
    bits = np.array([[int(bit) for bit in f"{i:03b}"] for i in range(6)], dtype=np.byte)
    evaluated = []

    def decode(population: NDArray[np.byte]) -> NDArray[np.int64]:
        return (population @ np.array([4, 2, 1]))[:, None]

    def objectives(phenotypes: NDArray[np.int64]) -> NDArray[np.float64]:
        evaluated.append(phenotypes[:, 0].copy())
        return lookup[phenotypes[:, 0]]

    optimizer = NSGAII(
        objectives,
        2,
        3,
        3,
        (False, False),
        init_population=bits[parent_ids],
        genotype_to_phenotype=decode,
        keep_history=True,
    )
    optimizer._get_init_population()
    optimizer._from_population_g_to_fitness()
    original_parents = optimizer._population_g_i.copy()
    monkeypatch.setattr(optimizer, "_get_offspring_population", lambda: bits[child_ids])
    optimizer._get_new_population()
    ids = optimizer._population_ph_i[:, 0]
    assert set(ids) == expected_ids
    np.testing.assert_array_equal(optimizer._population_g_i, bits[ids])
    np.testing.assert_array_equal(optimizer._fitness_i, lookup[ids])
    np.testing.assert_array_equal(evaluated[0], parent_ids)
    np.testing.assert_array_equal(evaluated[1], child_ids)
    assert len(evaluated) == 2
    assert optimizer.get_calls() == 6

    selected_distances = optimizer._crowding_distance_i.copy()
    if expected_ids == {0, 2, 4}:
        # The retained interior point has distance 1.2 in the union, not 2.0.
        np.testing.assert_allclose(selected_distances[ids == 2], [1.2])
    else:
        np.testing.assert_array_equal(optimizer._pareto_rank_i, [1, 2, 2])
    optimizer._from_population_g_to_fitness()
    np.testing.assert_array_equal(optimizer._crowding_distance_i, selected_distances)
    assert len(optimizer.get_stats()["fitness"]) == 2
    np.testing.assert_array_equal(optimizer.get_stats()["population_g"][0], original_parents)
    assert optimizer.get_calls() == 6


@pytest.mark.parametrize("selection", ["rank_crowding", "dominance_crowding"])
def test_fit_runs_generations_with_only_offspring_evaluations(selection: str) -> None:
    def objectives(population: NDArray[np.byte]) -> NDArray[np.float64]:
        total = population.sum(axis=1).astype(float)
        return np.column_stack((total, -total))

    optimizer = NSGAII(
        objectives,
        3,
        4,
        6,
        (False, False),
        selection=selection,
        keep_history=True,
        random_state=42,
    )
    assert optimizer.fit() is optimizer
    assert optimizer.get_calls() == 12
    assert optimizer._population_g_i.shape == (4, 6)
    assert optimizer._population_g_i.dtype == np.byte
    assert len(optimizer.get_stats()["fitness"]) == 3
    np.testing.assert_array_equal(optimizer._pareto_rank_i, np.ones(4))


@pytest.mark.parametrize("crossover, parents_num", [("uniform_tour_3", 3), ("uniform_tour_7", 7)])
def test_rank_crowding_crossover_presets(crossover: str, parents_num: int) -> None:
    def objectives(population: NDArray[np.byte]) -> NDArray[np.float64]:
        total = population.sum(axis=1).astype(float)
        return np.column_stack((total, -total))

    optimizer = NSGAII(
        objectives,
        3,
        8,
        12,
        (False, False),
        crossover=crossover,
        mutation="average",
        random_state=42,
    )
    assert optimizer._crossover_pool[crossover][1] == parents_num
    optimizer.fit()
    assert optimizer.get_calls() == 24
    assert optimizer._population_g_i.shape == (8, 12)
    assert set(np.unique(optimizer._population_g_i)) <= {0, 1}


@pytest.mark.parametrize("selection", ["rank_crowding", "dominance_crowding"])
def test_new_individual_real_operators_and_custom_mutation(selection: str) -> None:
    optimizer = NSGAII(
        lambda population: population.astype(float),
        10,
        2,
        2,
        (False, False),
        selection=selection,
        crossover="empty",
        mutation="custom_rate",
        mutation_rate=1.0,
        init_population=np.array([[1, 1], [0, 0]], dtype=np.byte),
    )
    optimizer._get_init_population()
    optimizer._from_population_g_to_fitness()
    original = optimizer._population_g_i.copy()
    calls = optimizer.get_calls()
    result = optimizer._get_new_individ_g(selection, "empty", "custom_rate")
    np.testing.assert_array_equal(result, [0, 0])
    assert result.dtype == np.byte
    np.testing.assert_array_equal(optimizer._population_g_i, original)
    assert optimizer.get_calls() == calls


def test_offspring_population_preserves_parents_and_does_not_evaluate_fitness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = NSGAII(
        lambda population: population.astype(float),
        10,
        3,
        4,
        (False, False),
        selection="rank_crowding",
        crossover="one_point",
        mutation="average",
    )
    parents = np.zeros((3, 4), dtype=np.byte)
    optimizer._population_g_i = parents
    children = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.byte)
    calls = []

    def make_child(selection: str, crossover: str, mutation: str) -> NDArray[np.byte]:
        calls.append((selection, crossover, mutation))
        return children[len(calls) - 1]

    monkeypatch.setattr(optimizer, "_get_new_individ_g", make_child)
    offspring = optimizer._get_offspring_population()
    np.testing.assert_array_equal(offspring, children)
    assert offspring.shape == (3, 4)
    assert offspring.dtype == np.byte
    assert calls == [("rank_crowding", "one_point", "average")] * 3
    assert optimizer._population_g_i is parents
    np.testing.assert_array_equal(parents, np.zeros((3, 4)))
    assert not np.shares_memory(offspring, parents)
    assert optimizer.get_calls() == 0
