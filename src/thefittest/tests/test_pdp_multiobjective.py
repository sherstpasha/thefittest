from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from ..base import init_symbolic_regression_uniset
from ..base._gp import fitness_function_gp_multiobjective
from ..optimizers import PDPNSGAII, PDPNSGPII, SelfCNSGPII


def test_pdp_probabilities_match_known_operator_scores() -> None:
    model = PDPNSGAII(lambda population: np.zeros((len(population), 2)), 10, 4, 8, (True, True))
    model._success_i = np.array([True, False, True, True])
    operators = np.array(["a", "a", "b", "b"])
    probabilities = model._get_new_proba_pdp({"a": 0.4, "b": 0.4, "unused": 0.2}, operators, 0.05)
    np.testing.assert_allclose(
        [probabilities["a"], probabilities["b"], probabilities["unused"]],
        [0.05 + 0.85 * 2 / 7, 0.05 + 0.85 * 5 / 7, 0.05],
    )


def test_pdp_success_uses_dominance_for_all_offspring() -> None:
    model = PDPNSGAII(lambda population: np.zeros((len(population), 2)), 10, 4, 3, (True, False))
    parents_fitness = np.array([[2.0, 1.0]] * 4) * model._objective_signs
    offspring_fitness = (
        np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [3.0, 1.0]]) * model._objective_signs
    )
    model._previous_fitness_i = [row.copy() for row in parents_fitness]
    model._selection_operators = np.array(["rank_crowding"] * 4)
    model._crossover_operators = np.array(["empty"] * 4)
    model._mutation_operators = np.array(["weak"] * 4)
    population = np.zeros((8, 3), dtype=np.byte)
    model._replace_population(
        population,
        population,
        np.concatenate((parents_fitness, offspring_fitness)),
        np.ones(8, dtype=np.int64),
        np.ones(8),
        np.arange(4, dtype=np.int64),
    )
    np.testing.assert_array_equal(model._success_i, [True, False, False, False])
    np.testing.assert_array_equal(model._fitness_i, parents_fitness)
    assert model._previous_fitness_i == []


@pytest.mark.parametrize("optimizer_type", [PDPNSGAII, PDPNSGPII, SelfCNSGPII])
def test_adaptive_optimizers_preserve_evolution_with_cache(optimizer_type: type) -> None:
    arguments: dict[str, Any] = dict(
        iters=8,
        pop_size=20,
        minimization=(True, True),
        random_state=42,
        keep_history=True,
    )
    if optimizer_type is PDPNSGAII:

        def objectives(population: NDArray) -> NDArray[np.float64]:
            return np.column_stack((population.sum(axis=1), population[:, 0])).astype(float)

        arguments.update(fitness_function=objectives, str_len=8)
    else:
        x = np.linspace(-2, 2, 30)[:, None]
        arguments.update(
            fitness_function=fitness_function_gp_multiobjective,
            fitness_function_args={"y": x[:, 0] ** 2 + np.sin(x[:, 0]), "fitness_cache": None},
            uniset=init_symbolic_regression_uniset(x, ("add", "mul", "sin")),
            max_level=5,
            init_level=3,
        )
    uncached = optimizer_type(**arguments, use_fitness_cache=False)
    cached = optimizer_type(**arguments, use_fitness_cache=True)
    uncached.fit()
    cached.fit()
    assert uncached.get_calls() == 160
    assert cached.get_calls() < uncached.get_calls()
    first, second = uncached.get_stats(), cached.get_stats()
    assert first.keys() == second.keys()
    for key in first:
        assert len(first[key]) == len(second[key]) == 8
        for a, b in zip(first[key], second[key]):
            if key in ("population_g", "population_ph") and optimizer_type is not PDPNSGAII:
                assert [tree.signature() for tree in a] == [tree.signature() for tree in b]
            elif isinstance(a, dict):
                assert a == b
                np.testing.assert_allclose(sum(a.values()), 1.0)
            else:
                np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        uncached.get_pareto_front()["fitness"], cached.get_pareto_front()["fitness"]
    )
