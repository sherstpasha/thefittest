from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pytest

from ..base._ea import MultiObjectiveEvolutionaryAlgorithm


def objective_values(population: NDArray[np.float64], offset: float = 0.0) -> NDArray[np.float64]:
    return np.column_stack((population.sum(axis=1), population[:, 0] + offset))


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_get_fitness_converts_directions_preserves_order_and_counts_individuals(
    n_jobs: int,
) -> None:
    population = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    original = population.copy()
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        fitness_function=objective_values,
        iters=10,
        pop_size=5,
        minimization=(True, False),
        fitness_function_args={"offset": 10.0},
        n_jobs=n_jobs,
    )
    expected = np.array([[-3.0, 11.0], [-7.0, 13.0], [-11.0, 15.0]])
    np.testing.assert_array_equal(optimizer._get_fitness(population), expected)
    assert optimizer.get_calls() == 3
    np.testing.assert_array_equal(optimizer._get_fitness(population[:1]), expected[:1])
    assert optimizer.get_calls() == 4
    np.testing.assert_array_equal(population, original)


@pytest.mark.parametrize("values", [np.ones(3), np.ones((3, 1)), np.ones((2, 2))])
def test_get_fitness_rejects_wrong_shape(values: NDArray[np.float64]) -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        fitness_function=lambda population: values,
        iters=10,
        pop_size=3,
        minimization=(False, True),
    )
    with pytest.raises(ValueError, match="fitness_function must return shape"):
        optimizer._get_fitness(np.ones((3, 2)))
    assert optimizer.get_calls() == 0


def test_get_fitness_does_not_modify_objective_function_result() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        fitness_function=lambda population: values,
        iters=10,
        pop_size=2,
        minimization=(True, True),
    )
    result = optimizer._get_fitness(np.ones((2, 1)))
    np.testing.assert_array_equal(result, -values)
    np.testing.assert_array_equal(values, [[1.0, 2.0], [3.0, 4.0]])
    assert result.dtype == np.float64
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize("keep_history", [False, True])
def test_update_data_calculates_each_front_and_records_population(keep_history: bool) -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        fitness_function=objective_values,
        iters=10,
        pop_size=7,
        minimization=(False, False),
        keep_history=keep_history,
    )
    optimizer._population_g_i = np.arange(14).reshape(7, 2)
    optimizer._population_ph_i = optimizer._population_g_i + 100
    optimizer._fitness_i = np.array(
        [[1.0, 3.0], [0.0, 2.0], [3.0, 1.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [-1.0, -1.0]]
    )
    original_fitness = optimizer._fitness_i.copy()

    optimizer._update_data()

    np.testing.assert_array_equal(optimizer._pareto_rank_i, [1, 2, 1, 2, 1, 2, 3])
    np.testing.assert_allclose(
        optimizer._crowding_distance_i, [np.inf, np.inf, np.inf, np.inf, 2.0, 2.0, np.inf]
    )
    front = optimizer._pareto_front.get()
    np.testing.assert_array_equal(front["indices"], [0, 2, 4])
    np.testing.assert_array_equal(front["fitness"], original_fitness[[0, 2, 4]])
    np.testing.assert_array_equal(front["genotype"], optimizer._population_g_i[[0, 2, 4]])
    np.testing.assert_array_equal(front["phenotype"], optimizer._population_ph_i[[0, 2, 4]])
    assert optimizer.get_calls() == 0

    stats = optimizer.get_stats()
    if keep_history:
        assert set(stats) == {
            "fitness",
            "population_g",
            "population_ph",
            "pareto_rank",
            "crowding_distance",
        }
        np.testing.assert_array_equal(stats["pareto_rank"][0], optimizer._pareto_rank_i)
        np.testing.assert_array_equal(stats["crowding_distance"][0], optimizer._crowding_distance_i)
    else:
        assert stats == {}

    optimizer._fitness_i[:] = 0.0
    optimizer._update_data()
    np.testing.assert_array_equal(optimizer._pareto_rank_i, np.ones(7))
    np.testing.assert_array_equal(optimizer._crowding_distance_i, np.zeros(7))
    np.testing.assert_array_equal(optimizer._pareto_front.get()["indices"], np.arange(7))
    if keep_history:
        assert len(stats["fitness"]) == 2
        np.testing.assert_array_equal(stats["fitness"][0], original_fitness)
        np.testing.assert_array_equal(stats["pareto_rank"][0], [1, 2, 1, 2, 1, 2, 3])


@pytest.mark.parametrize("show_progress_each", [None, 2])
def test_inherited_fit_uses_multiobjective_progress_and_iteration_limit(
    show_progress_each: Optional[int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    population = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        fitness_function=objective_values,
        iters=4,
        pop_size=3,
        minimization=(True, False),
        keep_history=True,
        show_progress_each=show_progress_each,
    )
    monkeypatch.setattr(
        optimizer,
        "_get_init_population",
        lambda: setattr(optimizer, "_population_g_i", population.copy()),
    )
    # With unchanged genotypes, every generation has the same fitness.
    assert optimizer.fit() is optimizer
    assert optimizer.get_calls() == 12
    assert len(optimizer.get_stats()["fitness"]) == 4
    output = capsys.readouterr().out
    if show_progress_each is None:
        assert output == ""
    else:
        assert output.splitlines() == [
            "0-th iteration with Pareto front size = 3; best objectives = [f1=3, f2=5]",
            "2-th iteration with Pareto front size = 3; best objectives = [f1=3, f2=5]",
        ]


@pytest.mark.parametrize("no_increase_num", [1, 2])
def test_fit_stops_after_unchanged_front_updates(
    no_increase_num: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    population = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        fitness_function=objective_values,
        iters=10,
        pop_size=3,
        minimization=(True, False),
        no_increase_num=no_increase_num,
        keep_history=True,
    )
    monkeypatch.setattr(
        optimizer,
        "_get_init_population",
        lambda: setattr(optimizer, "_population_g_i", population.copy()),
    )
    optimizer.fit()
    assert optimizer._pareto_front._no_update_counter == no_increase_num
    assert len(optimizer.get_stats()["fitness"]) == no_increase_num + 1
    assert optimizer.get_calls() == 3 * (no_increase_num + 1)


@pytest.mark.parametrize("no_increase_num", [0, -1, 1.5])
def test_constructor_rejects_invalid_stability_limit(no_increase_num: float) -> None:
    with pytest.raises(ValueError, match="no_increase_num"):
        MultiObjectiveEvolutionaryAlgorithm(
            fitness_function=objective_values,
            iters=10,
            pop_size=3,
            minimization=(True, False),
            no_increase_num=no_increase_num,
        )


def test_update_data_uses_fitness_update_eps_for_stopping() -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        objective_values,
        10,
        2,
        minimization=(False, False),
        no_increase_num=1,
        fitness_update_eps=0.1,
    )
    optimizer._population_g_i = np.ones((2, 2))
    optimizer._population_ph_i = np.ones((2, 2))
    optimizer._fitness_i = np.array([[1.0, 3.0], [3.0, 1.0]])
    optimizer._update_data()
    assert not optimizer._termitation_check()
    optimizer._fitness_i += 0.05
    optimizer._update_data()
    assert optimizer._termitation_check()


@pytest.mark.parametrize("tolerance, reached", [(0.0, False), (0.1, True)])
def test_objective_targets_require_one_solution_to_reach_all_criteria(
    tolerance: float, reached: bool
) -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        objective_values,
        10,
        2,
        minimization=(True, False),
        optimal_value=(1.0, 5.0),
        termination_error_value=tolerance,
    )
    # Each target is met by a different individual; this must not stop evolution.
    optimizer._fitness_i = np.array([[0.5, 4.0], [2.0, 6.0]]) * optimizer._objective_signs
    assert not optimizer._termitation_check()
    optimizer._fitness_i = np.array([[1.05, 4.95]]) * optimizer._objective_signs
    assert optimizer._termitation_check() is reached


@pytest.mark.parametrize("parameter", ["fitness_update_eps", "termination_error_value"])
def test_constructor_rejects_negative_tolerances(parameter: str) -> None:
    with pytest.raises(ValueError, match=parameter):
        MultiObjectiveEvolutionaryAlgorithm(
            objective_values, 10, 2, minimization=(False, False), **{parameter: -0.1}
        )


def test_get_pareto_front_restores_objectives_and_returns_independent_arrays() -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        objective_values, 10, 3, minimization=(True, False)
    )
    optimizer._population_g_i = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.byte)
    optimizer._population_ph_i = optimizer._population_g_i.astype(float)
    optimizer._fitness_i = np.array([[-1.0, 3.0], [-3.0, 5.0], [-4.0, 2.0]])
    optimizer._update_data()
    front = optimizer.get_pareto_front()
    np.testing.assert_array_equal(front["fitness"], [[1.0, 3.0], [3.0, 5.0]])
    np.testing.assert_array_equal(front["indices"], [0, 1])
    np.testing.assert_array_equal(front["genotype"], optimizer._population_g_i[:2])
    np.testing.assert_array_equal(front["phenotype"], optimizer._population_ph_i[:2])
    for value in front.values():
        value[:] = 99
    np.testing.assert_array_equal(optimizer.get_pareto_front()["fitness"], [[1.0, 3.0], [3.0, 5.0]])
    np.testing.assert_array_equal(optimizer._fitness_i, [[-1.0, 3.0], [-3.0, 5.0], [-4.0, 2.0]])


def test_get_pareto_front_before_evaluation_and_get_fittest_contract() -> None:
    optimizer = MultiObjectiveEvolutionaryAlgorithm(
        objective_values, 10, 3, minimization=(True, False)
    )
    assert optimizer.get_pareto_front()["fitness"].shape == (0, 2)
    with pytest.raises(NotImplementedError, match="get_pareto_front"):
        optimizer.get_fittest()
