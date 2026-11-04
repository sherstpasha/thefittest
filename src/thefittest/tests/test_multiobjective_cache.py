from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from ..base import TerminalNode, Tree, init_symbolic_regression_uniset
from ..base._ea import MultiObjectiveEvolutionaryAlgorithm
from ..optimizers import NSGAII, NSGPII, SelfCNSGAII


def objectives(population: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((population.sum(axis=1), population[:, 0]))


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_vector_cache_reuses_results_preserves_directions_and_copies(n_jobs: int) -> None:
    population = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        objectives, 1, 3, (True, False), use_fitness_cache=True, n_jobs=n_jobs
    )
    expected = np.array([[-3.0, 1.0], [-7.0, 3.0], [-3.0, 1.0]])
    phenotype, fitness = optimizer._evaluate_population(population)
    np.testing.assert_array_equal(fitness, expected)
    assert optimizer.get_calls() == 2
    phenotype[:] = 999
    fitness[:] = 999

    phenotype, fitness = optimizer._evaluate_population(population[::-1])
    np.testing.assert_array_equal(phenotype, population[::-1])
    np.testing.assert_array_equal(fitness, expected[::-1])
    assert optimizer.get_calls() == 2


@pytest.mark.parametrize(
    "enabled,limit,expected_calls", [(False, 1000, 6), (True, 0, 4), (True, None, 2)]
)
def test_vector_cache_enablement_and_zero_or_unlimited_capacity(
    enabled: bool, limit: int | None, expected_calls: int
) -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        objectives,
        1,
        3,
        (False, False),
        use_fitness_cache=enabled,
        fitness_cache_size=limit,
    )
    population = np.array([[0.0], [0.0], [1.0]])
    optimizer._evaluate_population(population)
    optimizer._evaluate_population(population)
    assert optimizer.get_calls() == expected_calls


def test_vector_cache_evicts_least_recently_used_entry() -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        objectives, 1, 2, (False, False), use_fitness_cache=True, fitness_cache_size=2
    )
    optimizer._evaluate_population(np.array([[0.0], [1.0]]))
    optimizer._evaluate_population(np.array([[0.0]]))
    optimizer._evaluate_population(np.array([[2.0]]))
    keys = [optimizer._get_genotype_cache_key(np.array([value])) for value in (0.0, 1.0, 2.0)]
    assert keys[0] in optimizer._fitness_cache
    assert keys[1] not in optimizer._fitness_cache
    assert keys[2] in optimizer._fitness_cache
    optimizer._evaluate_population(np.array([[1.0]]))
    assert optimizer.get_calls() == 4
    assert len(optimizer._fitness_cache) == 2


@pytest.mark.parametrize("optimizer_type", [NSGAII, SelfCNSGAII, NSGPII])
def test_optimizers_reuse_cached_fitness_across_generations(optimizer_type: type) -> None:
    arguments: dict[str, Any] = dict(
        iters=3,
        pop_size=4,
        minimization=(True, False),
        random_state=42,
    )
    if optimizer_type is NSGPII:
        uniset = init_symbolic_regression_uniset(np.ones((3, 1)), ("sin",))
        first, second = Tree([TerminalNode(0.0, "0")]), Tree([TerminalNode(1.0, "1")])
        population = np.array([first, first.copy(), second, second.copy()], dtype=object)

        def tree_objectives(trees: NDArray[Any]) -> NDArray[np.float64]:
            return np.array([[float(tree()), len(tree)] for tree in trees], dtype=float)

        arguments.update(
            fitness_function=tree_objectives,
            uniset=uniset,
            init_population=population,
            crossover="gp_empty",
            mutation="gp_custom_rate_point",
            mutation_rate=0.0,
        )
    else:
        arguments.update(
            fitness_function=objectives,
            str_len=2,
            init_population=np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.byte),
            mutation_rate=0.0,
        )
        if optimizer_type is SelfCNSGAII:
            arguments.update(crossovers=("empty",), mutations=("custom_rate",))
        else:
            arguments.update(crossover="empty", mutation="custom_rate")
    cached = optimizer_type(**arguments, use_fitness_cache=True, fitness_cache_size=10)
    uncached = optimizer_type(**arguments, use_fitness_cache=False)
    cached.fit()
    uncached.fit()
    assert cached.get_calls() == 2
    assert uncached.get_calls() == 12
    np.testing.assert_array_equal(
        cached.get_pareto_front()["fitness"], uncached.get_pareto_front()["fitness"]
    )
