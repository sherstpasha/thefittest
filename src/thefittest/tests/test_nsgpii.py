import numpy as np
from numpy.typing import NDArray
import pytest

from ..base import Tree
from ..base import UniversalSet
from ..base import TerminalNode
from ..base import init_symbolic_regression_uniset
from ..base._gp import fitness_function_gp_multiobjective
from ..optimizers import NSGAII
from ..optimizers._nsgpii import NSGPII
from ..utils.random import check_random_state


def forbidden_fitness(population: NDArray) -> NDArray[np.float64]:
    raise AssertionError("Generating individuals must not evaluate fitness")


@pytest.mark.parametrize("use_gp", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_new_individual_rejects_unsupported_selection(use_gp: bool, registered: bool) -> None:
    if use_gp:
        uniset = init_symbolic_regression_uniset(np.zeros((3, 1)), functional_set_names=("add",))
        optimizer = NSGPII(forbidden_fitness, uniset, 10, 2, (True, True))
    else:
        optimizer = NSGAII(forbidden_fitness, 10, 2, 4, (True, True))
    if registered:
        optimizer._selection_pool["third_selection"] = optimizer._selection_pool["rank_crowding"]
    with pytest.raises(ValueError, match="Unsupported selection: third_selection"):
        optimizer._get_new_individ_g(
            "third_selection", optimizer._specified_crossover, optimizer._specified_mutation
        )


def test_first_generation_creates_tree_population() -> None:
    x = np.arange(6, dtype=float).reshape(3, 2)
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add", "mul"))
    optimizer = NSGPII(forbidden_fitness, uniset, 10, 8, (True, True), init_level=4)
    check_random_state(42)
    optimizer._first_generation()
    population = optimizer._population_g_i
    assert population.shape == (8,)
    assert population.dtype == object
    assert all(isinstance(tree, Tree) for tree in population)
    assert all(len(tree) > 0 and tree.get_max_level() <= 4 for tree in population)
    assert len({id(tree) for tree in population}) == 8
    assert optimizer.get_calls() == 0


@pytest.mark.parametrize("selection", ["rank_crowding", "dominance_crowding"])
@pytest.mark.parametrize("constant_rate", [False, True])
def test_new_tree_uses_selected_parents_and_gp_mutation_contract(
    selection: str, constant_rate: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = np.arange(6, dtype=float).reshape(3, 2)
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add",))
    parents = np.array([Tree([TerminalNode(x[:, i], f"x{i}")]) for i in range(2)], dtype=object)
    child = Tree([uniset._functional_set[2][0], *[tree._nodes[0] for tree in parents]])
    mutated = parents[1].copy()
    optimizer = NSGPII(forbidden_fitness, uniset, 10, 2, (True, True), max_level=7)
    optimizer._population_g_i = parents
    optimizer._fitness_i = np.array([[1.0, 2.0], [3.0, 0.0]])
    optimizer._pareto_rank_i = np.array([1, 2], dtype=np.int64)
    optimizer._crowding_distance_i = np.array([np.inf, 0.5])
    calls = []

    def select(metadata: NDArray, distances: NDArray, tour_size: int, quantity: int) -> NDArray:
        expected = (
            optimizer._pareto_rank_i if selection == "rank_crowding" else optimizer._fitness_i
        )
        assert metadata is expected
        assert distances is optimizer._crowding_distance_i
        assert tour_size == quantity == 2
        calls.append("selection")
        return np.array([1, 0], dtype=np.int64)

    def crossover(selected: NDArray, distances: NDArray, ranks: NDArray, max_level: int) -> Tree:
        assert selected[0] is parents[1] and selected[1] is parents[0]
        np.testing.assert_array_equal(distances, [0.5, np.inf])
        np.testing.assert_array_equal(ranks, [2.0, 1.0])
        assert max_level == 7
        calls.append("crossover")
        return child

    def mutate(tree: Tree, actual_uniset: UniversalSet, proba: float, max_level: int) -> Tree:
        assert tree is child and actual_uniset is uniset
        assert proba == (0.25 if constant_rate else 0.25 / len(child))
        assert max_level == 7
        calls.append("mutation")
        return mutated

    monkeypatch.setitem(optimizer._selection_pool, selection, (select, 2))
    monkeypatch.setitem(optimizer._crossover_pool, "gp_standard", (crossover, 2))
    monkeypatch.setitem(optimizer._mutation_pool, "gp_weak_grow", (mutate, 0.25, constant_rate))
    assert optimizer._get_new_individ_g(selection, "gp_standard", "gp_weak_grow") is mutated
    assert calls == ["selection", "crossover", "mutation"]
    assert [str(tree) for tree in parents] == ["x0", "x1"]
    assert optimizer.get_calls() == 0


@pytest.mark.parametrize("crossover_rate", [0.0, 1.0])
def test_new_tree_real_operators_and_crossover_skip(crossover_rate: float) -> None:
    x = np.arange(6, dtype=float).reshape(3, 2)
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add",))
    parents = np.array([Tree([TerminalNode(x[:, i], f"x{i}")]) for i in range(2)], dtype=object)
    optimizer = NSGPII(
        forbidden_fitness,
        uniset,
        10,
        2,
        (False, False),
        mutation="gp_custom_rate_point",
        mutation_rate=0.0,
        crossover_rate=crossover_rate,
    )
    optimizer._population_g_i = parents
    optimizer._fitness_i = np.array([[1.0, 1.0], [0.0, 0.0]])
    optimizer._pareto_rank_i = np.array([1, 2], dtype=np.int64)
    optimizer._crowding_distance_i = np.array([np.inf, np.inf])
    check_random_state(42)
    child = optimizer._get_new_individ_g(
        "dominance_crowding", "gp_standard", "gp_custom_rate_point"
    )
    assert isinstance(child, Tree)
    assert child is not parents[0] and child is not parents[1]
    assert str(child) == "x0"
    assert [str(tree) for tree in parents] == ["x0", "x1"]
    assert child.get_max_level() <= optimizer._max_level
    assert optimizer.get_calls() == 0


def test_first_generation_copies_supplied_population_array() -> None:
    x = np.arange(6, dtype=float).reshape(3, 2)
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add",))
    trees = [Tree([TerminalNode(x[:, i], f"x{i}")]) for i in range(2)]
    initial = np.array(trees, dtype=object)
    optimizer = NSGPII(forbidden_fitness, uniset, 10, 2, (True, True), init_population=initial)
    optimizer._first_generation()
    population = optimizer._population_g_i
    assert population.shape == (2,)
    assert population.dtype == object
    assert not np.shares_memory(population, initial)
    assert all(actual is expected for actual, expected in zip(population, trees))
    population[0] = trees[1]
    assert initial[0] is trees[0]
    assert optimizer.get_calls() == 0


@pytest.mark.parametrize("penalize", [True, False])
def test_duplicate_trees_are_ranked_after_distinct_structures(penalize: bool) -> None:
    x = np.arange(3, dtype=float)[:, None]
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add",))
    terminal = TerminalNode(x[:, 0], "x0")
    trees = np.array(
        [
            Tree([uniset._functional_set[2][0], terminal, TerminalNode(0.0, "zero")]),
            Tree([terminal]),
            Tree([terminal]),
            Tree([TerminalNode(0.0, "zero")]),
        ],
        dtype=object,
    )
    fitness = -fitness_function_gp_multiobjective(trees, x[:, 0])
    optimizer = NSGPII(forbidden_fitness, uniset, 10, 3, (True, True), penalize_duplicates=penalize)
    ranks, distances = optimizer._rank_and_crowding(fitness, trees)
    if penalize:
        np.testing.assert_array_equal(optimizer._duplicate_mask_i, [False, False, True, False])
        assert ranks[2] > max(ranks[[0, 1, 3]])
        survivors = optimizer._get_survivor_indices(ranks, distances)
        assert set(survivors) == {0, 1, 3}
    else:
        expected_ranks, expected_distances = NSGAII._rank_and_crowding(fitness)
        np.testing.assert_array_equal(ranks, expected_ranks)
        np.testing.assert_array_equal(distances, expected_distances)
        assert not optimizer._duplicate_mask_i.any()
    assert optimizer.get_calls() == 0


def test_duplicate_penalty_uses_cache_keys_without_executing_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = TerminalNode({"action": "left"}, "left")
    right = TerminalNode({"action": "right"}, "right")
    uniset = UniversalSet(functional_set=(), terminal_set=(left, right))
    tree = Tree([left])
    trees = np.array([tree, tree.copy(), Tree([right])], dtype=object)
    fitness = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])
    optimizer = NSGPII(forbidden_fitness, uniset, 10, 3, (True, True))
    cache_keys = []
    original_key = optimizer._get_genotype_cache_key

    def evaluate(tree: Tree) -> None:
        raise AssertionError("Duplicate detection must not execute trees")

    def get_key(tree: Tree) -> tuple[str, str]:
        key = original_key(tree)
        cache_keys.append(key)
        return key

    monkeypatch.setattr(Tree, "__call__", evaluate)
    monkeypatch.setattr(optimizer, "_get_genotype_cache_key", get_key)
    optimizer._rank_and_crowding(fitness, trees)
    assert len(cache_keys) == 3
    assert cache_keys[0] == cache_keys[1]
    np.testing.assert_array_equal(optimizer._duplicate_mask_i, [False, True, False])
    optimizer._rank_and_crowding(fitness, trees)
    np.testing.assert_array_equal(optimizer._duplicate_mask_i, [False, True, False])


def test_duplicates_fill_remaining_slots_and_mask_follows_survivor_order() -> None:
    x = np.arange(3, dtype=float)[:, None]
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add",))
    trees = np.array(
        [
            *[Tree([TerminalNode(0.0, "zero")]) for _ in range(4)],
            *[Tree([TerminalNode(1.0, "one")]) for _ in range(2)],
            *[Tree([TerminalNode(x[:, 0], "x0")]) for _ in range(2)],
        ],
        dtype=object,
    )
    fitness = -fitness_function_gp_multiobjective(trees, x[:, 0])
    optimizer = NSGPII(forbidden_fitness, uniset, 10, 4, (True, True))
    ranks, distances = optimizer._rank_and_crowding(fitness, trees)
    original_mask = optimizer._duplicate_mask_i.copy()
    survivors = optimizer._get_survivor_indices(ranks, distances)
    optimizer._replace_population(trees, trees, fitness, ranks, distances, survivors)
    assert len(optimizer._population_g_i) == 4
    assert {str(tree) for tree in optimizer._population_g_i} == {"zero", "one", "x0"}
    assert optimizer._duplicate_mask_i.sum() == 1
    np.testing.assert_array_equal(optimizer._duplicate_mask_i, original_mask[survivors])
    optimizer._update_data()
    assert len(optimizer.get_pareto_front()["fitness"]) == 1


@pytest.mark.parametrize("selection", ["rank_crowding", "dominance_crowding"])
def test_penalized_copy_loses_parent_selection(selection: str) -> None:
    x = np.arange(3, dtype=float)[:, None]
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add",))
    optimizer = NSGPII(
        forbidden_fitness,
        uniset,
        10,
        3,
        (True, True),
        crossover="gp_empty",
        mutation="gp_custom_rate_point",
        mutation_rate=0,
    )
    optimizer._population_g_i = np.array(
        [
            Tree([TerminalNode(x[:, 0], "x0")]),
            Tree([TerminalNode(2.0, "two")]),
            Tree([TerminalNode(x[:, 0], "x0")]),
        ],
        dtype=object,
    )
    optimizer._fitness_i = np.array([[10.0, 10.0], [0.0, 0.0], [10.0, 10.0]])
    original_fitness = optimizer._fitness_i.copy()
    optimizer._pareto_rank_i = np.array([4, 2, 1], dtype=np.int64)
    optimizer._crowding_distance_i = np.full(3, np.inf)
    optimizer._duplicate_mask_i = np.array([True, False, False])
    selector, tour_size = optimizer._selection_pool[selection]

    def first_pair(metadata: NDArray, distances: NDArray, size: int, quantity: int) -> NDArray:
        return selector(metadata[:2], distances[:2], size, quantity)

    optimizer._selection_pool[selection] = (first_pair, tour_size)
    child = optimizer._get_new_individ_g(selection, "gp_empty", "gp_custom_rate_point")
    assert str(child) == "two"
    np.testing.assert_array_equal(optimizer._fitness_i, original_fitness)
