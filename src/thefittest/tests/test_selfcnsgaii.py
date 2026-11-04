import numpy as np
from numpy.typing import NDArray
import pytest

from ..optimizers import SelfCNSGAII
from ..base import TerminalNode, Tree


@pytest.mark.parametrize("empty_name", ["empty", "gp_empty"])
@pytest.mark.parametrize("only_empty", [False, True])
def test_initial_empty_crossover_probability(empty_name: str, only_empty: bool) -> None:
    class NamedEmptyCrossover(SelfCNSGAII):
        def _init_variation_pools(self) -> None:
            super()._init_variation_pools()
            self._crossover_pool[empty_name] = self._crossover_pool.pop("empty")

    crossovers = (empty_name,) if only_empty else (empty_name, "one_point", "uniform_2")
    optimizer = NamedEmptyCrossover(
        lambda population: np.zeros((len(population), 2)),
        10,
        4,
        8,
        (False, False),
        crossovers=crossovers,
    )
    expected = (
        {empty_name: 1.0} if only_empty else {empty_name: 0.1, "one_point": 0.45, "uniform_2": 0.45}
    )
    assert optimizer._crossover_proba == expected


def test_constructor_uses_operators_from_supplied_sets() -> None:
    class RestrictedOperators(SelfCNSGAII):
        def _init_variation_pools(self) -> None:
            super()._init_variation_pools()
            self._crossover_pool = {"one_point": self._crossover_pool["one_point"]}
            self._mutation_pool = {"average": self._mutation_pool["average"]}

    optimizer = RestrictedOperators(
        lambda population: np.zeros((len(population), 2)),
        10,
        4,
        8,
        (False, False),
        selections=("rank_crowding",),
        crossovers=("one_point",),
        mutations=("average",),
    )
    assert optimizer._specified_selection == "rank_crowding"
    assert optimizer._specified_crossover == "one_point"
    assert optimizer._specified_mutation == "average"


@pytest.mark.parametrize("parameter", ["selections", "crossovers", "mutations"])
def test_constructor_rejects_empty_operator_sets(parameter: str) -> None:
    with pytest.raises(ValueError, match="operators must be non-empty and unique"):
        SelfCNSGAII(
            lambda population: np.zeros((len(population), 2)),
            10,
            4,
            8,
            (False, False),
            **{parameter: ()},
        )


def test_offspring_metadata_matches_operator_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_fitness(population: NDArray[np.byte]) -> NDArray[np.float64]:
        raise AssertionError("Creating offspring must not evaluate fitness")

    optimizer = SelfCNSGAII(forbidden_fitness, 10, 3, 4, (False, False))
    parents = np.zeros((3, 4), dtype=np.byte)
    optimizer._population_g_i = parents
    selections = np.array(["rank_crowding", "dominance_crowding", "rank_crowding"])
    crossovers = np.array(["one_point", "uniform_tour_7", "empty"])
    mutations = np.array(["weak", "average", "strong"])
    choices = iter((selections, crossovers, mutations))
    calls = []

    def choose(probabilities: dict[str, float]) -> NDArray[np.str_]:
        result = next(choices)
        assert set(result) <= probabilities.keys()
        return result

    def make_child(selection: str, crossover: str, mutation: str) -> NDArray[np.byte]:
        calls.append((selection, crossover, mutation))
        return np.eye(4, dtype=np.byte)[len(calls) - 1]

    monkeypatch.setattr(optimizer, "_choice_operators", choose)
    monkeypatch.setattr(optimizer, "_get_new_individ_g", make_child)
    offspring = optimizer._get_offspring_population()
    assert calls == list(zip(selections, crossovers, mutations))
    np.testing.assert_array_equal(optimizer._selection_operators, selections)
    np.testing.assert_array_equal(optimizer._crossover_operators, crossovers)
    np.testing.assert_array_equal(optimizer._mutation_operators, mutations)
    np.testing.assert_array_equal(offspring, np.eye(4, dtype=np.byte)[:3])
    np.testing.assert_array_equal(parents, np.zeros((3, 4)))
    assert optimizer._population_g_i is parents
    assert not np.shares_memory(offspring, parents)
    assert optimizer.get_calls() == 0


def test_choice_operators_respects_zero_probability() -> None:
    optimizer = SelfCNSGAII(lambda population: population.astype(float), 10, 8, 2, (False, False))
    probabilities = {"weak": 0.0, "average": 1.0, "strong": 0.0}
    chosen = optimizer._choice_operators(probabilities)
    np.testing.assert_array_equal(chosen, np.full(8, "average"))
    assert probabilities == {"weak": 0.0, "average": 1.0, "strong": 0.0}


def test_offspring_population_preserves_tree_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = SelfCNSGAII(
        lambda population: np.zeros((len(population), 2)), 10, 2, 1, (True, True)
    )
    trees = [Tree([TerminalNode(float(i), str(i))]) for i in range(2)]
    optimizer._population_g_i = np.array(trees, dtype=object)
    children = iter(trees)

    def make_child(selection: str, crossover: str, mutation: str) -> Tree:
        return next(children).copy()

    monkeypatch.setattr(optimizer, "_get_new_individ_g", make_child)
    offspring = optimizer._get_offspring_population()

    assert offspring.dtype == object
    assert offspring.shape == (2,)
    assert all(isinstance(tree, Tree) for tree in offspring)
    assert [tree.signature() for tree in offspring] == [tree.signature() for tree in trees]


def test_success_rates_use_fraction_of_all_offspring() -> None:
    operators = np.array(["a"] * 6 + ["b"] * 2 + ["c"] * 2)
    ranks = np.array([1, 1, 2, 2, 3, 3, 1, 2, 2, 3], dtype=np.int64)
    rates = SelfCNSGAII._get_operator_success_rates(operators, ranks)
    assert rates == {"a": 2 / 6, "b": 1 / 2, "c": 0.0}
    assert rates["b"] > rates["a"]


def test_success_rates_when_no_offspring_reaches_first_front() -> None:
    operators = np.array(["a", "b", "a"])
    ranks = np.array([2, 3, 4], dtype=np.int64)
    assert SelfCNSGAII._get_operator_success_rates(operators, ranks) == {"a": 0.0, "b": 0.0}


def test_success_rates_reject_misaligned_metadata() -> None:
    with pytest.raises(ValueError, match="same length"):
        SelfCNSGAII._get_operator_success_rates(np.array(["a", "b"]), np.array([1]))


def test_replacement_scores_all_offspring_before_survival() -> None:
    optimizer = SelfCNSGAII(lambda population: population.astype(float), 10, 3, 2, (False, False))
    optimizer._selection_operators = np.array(
        ["rank_crowding", "dominance_crowding", "rank_crowding"]
    )
    optimizer._crossover_operators = np.array(["one_point", "one_point", "uniform_2"])
    optimizer._mutation_operators = np.array(["weak", "average", "weak"])
    probabilities = [
        optimizer._selection_proba.copy(),
        optimizer._crossover_proba.copy(),
        optimizer._mutation_proba.copy(),
    ]
    population = np.arange(12, dtype=np.byte).reshape(6, 2)
    fitness = population.astype(float)
    ranks = np.array([1, 1, 2, 1, 1, 2], dtype=np.int64)
    distances = np.array([np.inf, np.inf, np.inf, 0.5, 0.5, np.inf])
    survivors = np.array([1, 0, 3], dtype=np.int64)

    optimizer._replace_population(population, population, fitness, ranks, distances, survivors)

    assert optimizer._selection_success_rates == {"rank_crowding": 0.5, "dominance_crowding": 1.0}
    assert optimizer._crossover_success_rates == {"one_point": 1.0, "uniform_2": 0.0}
    assert optimizer._mutation_success_rates == {"weak": 0.5, "average": 1.0}
    np.testing.assert_array_equal(optimizer._population_g_i, population[survivors])
    np.testing.assert_array_equal(optimizer._population_ph_i, population[survivors])
    np.testing.assert_array_equal(optimizer._fitness_i, fitness[survivors])
    np.testing.assert_array_equal(optimizer._pareto_rank_i, ranks[survivors])
    np.testing.assert_array_equal(optimizer._crowding_distance_i, distances[survivors])
    np.testing.assert_array_equal(optimizer._mutation_operators, ["weak", "average", "weak"])
    assert optimizer._selection_proba["dominance_crowding"] > probabilities[0]["dominance_crowding"]
    assert optimizer._crossover_proba["one_point"] > probabilities[1]["one_point"]
    assert optimizer._mutation_proba["average"] > probabilities[2]["average"]
    assert optimizer.get_calls() == 0


def test_probability_update_uses_selfc_step_and_preserves_lower_bound() -> None:
    optimizer = SelfCNSGAII(lambda population: population.astype(float), 10, 8, 2, (False, False))
    initial = {"a": 0.5, "b": 0.5}
    updated = optimizer._get_new_proba(initial, {"a": 1.0, "b": 0.0}, 0.05)
    assert updated == pytest.approx({"a": 0.6, "b": 0.4})
    assert initial == {"a": 0.5, "b": 0.5}
    for _ in range(30):
        updated = optimizer._get_new_proba(updated, {"a": 1.0, "b": 0.0}, 0.05)
        assert sum(updated.values()) == pytest.approx(1.0)
        assert min(updated.values()) >= 0.05
    assert updated == pytest.approx({"a": 0.95, "b": 0.05})


def test_probability_update_without_success_and_with_one_operator() -> None:
    optimizer = SelfCNSGAII(lambda population: population.astype(float), 10, 8, 2, (False, False))
    probabilities = {"a": 0.4, "b": 0.6}
    assert optimizer._get_new_proba(probabilities, {"a": 0.0, "b": 0.0}, 0.05) == probabilities
    assert optimizer._get_new_proba({"a": 1.0}, {"a": 1.0}, 0.05) == {"a": 1.0}
    assert optimizer._get_new_proba(probabilities, {"a": 1.0}, 0.5) == {"a": 0.5, "b": 0.5}


def test_fit_adapts_probabilities_reproducibly_without_extra_evaluations() -> None:
    def objectives(population: NDArray[np.byte]) -> NDArray[np.float64]:
        total = population.sum(axis=1).astype(float)
        return np.column_stack((total, -total))

    def run() -> SelfCNSGAII:
        optimizer = SelfCNSGAII(objectives, 3, 8, 12, (False, False), random_state=42)
        optimizer.fit()
        assert optimizer.get_calls() == 24
        assert optimizer._population_g_i.shape == (8, 12)
        for probabilities in (
            optimizer._selection_proba,
            optimizer._crossover_proba,
            optimizer._mutation_proba,
        ):
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert min(probabilities.values()) >= 0.05
        return optimizer

    first, second = run(), run()
    np.testing.assert_array_equal(first._population_g_i, second._population_g_i)
    assert first._selection_proba == second._selection_proba
    assert first._crossover_proba == second._crossover_proba
    assert first._mutation_proba == second._mutation_proba


@pytest.mark.parametrize("keep_history", [True, False])
def test_probability_history_tracks_generations_without_aliasing(keep_history: bool) -> None:
    def objectives(population: NDArray[np.byte]) -> NDArray[np.float64]:
        total = population.sum(axis=1).astype(float)
        return np.column_stack((total, -total))

    optimizer = SelfCNSGAII(
        objectives, 3, 8, 12, (False, False), keep_history=keep_history, random_state=42
    )
    initial = {
        "s_proba": optimizer._selection_proba.copy(),
        "c_proba": optimizer._crossover_proba.copy(),
        "m_proba": optimizer._mutation_proba.copy(),
    }
    optimizer.fit()
    stats = optimizer.get_stats()
    if not keep_history:
        assert not stats
        return

    assert len(stats["fitness"]) == 3
    for key, current in (
        ("s_proba", optimizer._selection_proba),
        ("c_proba", optimizer._crossover_proba),
        ("m_proba", optimizer._mutation_proba),
    ):
        history = stats[key]
        assert len(history) == 3
        assert history[0] == initial[key]
        assert history[-1] == current
        assert history[0] is not history[-1]
        last_snapshot = history[-1].copy()
        current[next(iter(current))] = -1.0
        assert history[0] == initial[key]
        assert history[-1] == last_snapshot
