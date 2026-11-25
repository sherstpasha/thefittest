import numpy as np
import pytest

from ..benchmarks import SCH, FON, POL, KUR, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from ..utils.multiobjective import convergence_metric, diversity_metric


@pytest.mark.parametrize("problem_type", [SCH, FON, POL, KUR, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6])
def test_problem_population_contract(problem_type: type) -> None:
    problem = problem_type()
    population = np.random.RandomState(42).uniform(
        problem.left_border, problem.right_border, size=(5, problem.n_variables)
    )
    result = problem(population)
    assert result.shape == (5, 2)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))
    for index in range(5):
        np.testing.assert_allclose(result[index], problem(population[index : index + 1])[0])


def test_classical_problems_at_known_points() -> None:
    np.testing.assert_allclose(
        SCH()(np.array([[0.0], [1.0], [2.0]])), [[0.0, 4.0], [1.0, 1.0], [4.0, 0.0]]
    )
    np.testing.assert_allclose(FON()(np.zeros((1, 3))), [[1.0 - np.exp(-1.0)] * 2])
    np.testing.assert_allclose(POL()(np.array([[1.0, 2.0]])), [[1.0, 25.0]])
    np.testing.assert_allclose(KUR()(np.zeros((1, 3))), [[-20.0, 0.0]])


@pytest.mark.parametrize("problem_type", [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6])
def test_zdt_origin_and_pareto_boundary(problem_type: type) -> None:
    problem = problem_type()
    x = np.zeros((2, problem.n_variables))
    x[1, 0] = 1.0
    expected = [[1.0, 0.0], [1.0, 0.0]] if problem_type is ZDT6 else [[0.0, 1.0], [1.0, 0.0]]
    np.testing.assert_allclose(problem(x), expected, atol=1e-14)


def test_convergence_direction_is_obtained_to_reference() -> None:
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    front = reference[:1]
    assert convergence_metric(front, reference) == 0.0
    assert diversity_metric(front, reference) == 1.0


def test_delta_for_uniform_and_irregular_fronts() -> None:
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert diversity_metric(reference[::-1], reference) == 0.0
    assert diversity_metric(np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), reference) == 1.0
