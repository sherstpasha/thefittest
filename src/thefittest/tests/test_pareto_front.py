import numpy as np
import pytest

from ..base import _ea
from ..base._ea import ParetoFront
from ..utils.multiobjective import fast_non_dominated_sort


def test_pareto_front_preserves_alignment_order_and_duplicates():
    genotype = np.arange(10).reshape(5, 2)
    phenotype = genotype.astype(np.float64) / 10
    fitness = np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 3.0], [1.0, 1.0], [3.0, 1.0]])
    front = ParetoFront()
    front._update(genotype, phenotype, fitness)

    result = front.get()
    assert set(result) == {"genotype", "phenotype", "fitness", "indices"}
    assert np.array_equal(result["indices"], [1, 2, 4])
    for name, population in (
        ("genotype", genotype),
        ("phenotype", phenotype),
        ("fitness", fitness),
    ):
        assert np.array_equal(result[name], population[[1, 2, 4]])
    assert result["indices"].dtype == np.int64


def test_pareto_front_precomputed_ranks_skip_sorting(monkeypatch):
    fitness = np.array([[1.0, 3.0], [3.0, 1.0], [0.0, 0.0]])
    ranks = fast_non_dominated_sort(fitness)
    expected = ParetoFront()
    expected._update(fitness, fitness, fitness)

    def forbidden_sort(_):
        raise AssertionError("precomputed ranks must avoid another sort")

    monkeypatch.setattr(_ea, "fast_non_dominated_sort", forbidden_sort)
    actual = ParetoFront()
    actual._update(fitness, fitness, fitness, ranks=ranks)
    for name, values in expected.get().items():
        assert np.array_equal(actual.get()[name], values)


def test_pareto_front_replaces_previous_generation():
    front = ParetoFront()
    first = np.array([[10.0, 10.0], [1.0, 1.0]])
    front._update(first, first, first)
    second = np.array([[0.0, 0.0], [-1.0, 1.0]])
    front._update(second, second, second)
    assert np.array_equal(front.get()["fitness"], second)
    assert np.array_equal(front.get()["indices"], [0, 1])


def test_pareto_front_numeric_inputs_and_outputs_are_independent():
    genotype = np.array([[1, 0], [0, 1], [0, 0]])
    phenotype = genotype.astype(float)
    fitness = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 0.0]])
    front = ParetoFront()
    front._update(genotype, phenotype, fitness)
    expected = front.get()
    genotype[:] = 9
    phenotype[:] = 9
    fitness[:] = 9
    returned = front.get()
    for name, values in expected.items():
        assert np.array_equal(returned[name], values)
        returned[name][:] = 99
    for name, values in expected.items():
        assert np.array_equal(front.get()[name], values)


def test_pareto_front_object_individuals_are_independent():
    genotype = np.empty(2, dtype=object)
    phenotype = np.empty(2, dtype=object)
    genotype[:] = [{"genes": [1, 0]}, {"genes": [0, 1]}]
    phenotype[:] = [{"decoded": [0.5]}, {"decoded": [1.5]}]
    fitness = np.array([[2.0, 1.0], [1.0, 2.0]])
    front = ParetoFront()
    front._update(genotype, phenotype, fitness)
    genotype[0]["genes"][0] = 999
    phenotype[1]["decoded"][0] = 999
    result = front.get()
    assert result["genotype"][0]["genes"] == [1, 0]
    assert result["phenotype"][1]["decoded"] == [1.5]
    result["genotype"][0]["genes"][0] = 888
    result["phenotype"][1]["decoded"][0] = 888
    assert front.get()["genotype"][0]["genes"] == [1, 0]
    assert front.get()["phenotype"][1]["decoded"] == [1.5]


@pytest.mark.parametrize("precomputed", [False, True])
def test_pareto_front_empty_update_clears_previous_front(precomputed):
    front = ParetoFront()
    front._update(np.ones((1, 3)), np.ones((1, 4)), np.ones((1, 2)))
    ranks = np.empty(0, dtype=np.int64) if precomputed else None
    front._update(np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 2)), ranks=ranks)
    result = front.get()
    assert result["genotype"].shape == (0, 3)
    assert result["phenotype"].shape == (0, 4)
    assert result["fitness"].shape == (0, 2)
    assert result["indices"].shape == (0,)


def test_pareto_front_before_first_update():
    result = ParetoFront().get()
    assert result["genotype"].size == result["phenotype"].size == 0
    assert result["fitness"].shape == (0, 0)
    assert result["indices"].shape == (0,)


@pytest.mark.parametrize(
    "fitness, ranks",
    [
        (np.ones(2), None),
        (np.empty((2, 0)), None),
        (np.array([[np.nan, 0.0], [1.0, 2.0]]), None),
        (np.ones((3, 2)), None),
        (np.ones((2, 2)), np.array([1])),
        (np.ones((2, 2)), np.array([[1], [2]])),
        (np.ones((2, 2)), np.array([1.0, 2.0])),
        (np.ones((2, 2)), np.array([0, 1])),
        (np.ones((2, 2)), np.array([2, 3])),
    ],
)
def test_pareto_front_invalid_update_keeps_previous_snapshot(fitness, ranks):
    front = ParetoFront()
    population = np.ones((2, 3))
    front._update(population, population, np.array([[2.0, 2.0], [1.0, 1.0]]))
    expected = front.get()
    expected_counter = front._no_update_counter
    with pytest.raises(ValueError):
        front._update(population, population, fitness, ranks=ranks)
    for name, values in expected.items():
        assert np.array_equal(front.get()[name], values)
    assert front._no_update_counter == expected_counter


def test_pareto_front_stability_ignores_order_duplicates_and_genotypes() -> None:
    front = ParetoFront()
    fitness = np.array([[1.0, 3.0], [3.0, 1.0]])
    front._update(fitness, fitness, fitness)
    assert front._no_update_counter == 0
    front._update(fitness, fitness, fitness)
    assert front._no_update_counter == 1

    repeated = fitness[[1, 0, 1]]
    genotype = np.arange(6).reshape(3, 2)
    front._update(genotype, genotype + 10, repeated)
    assert front._no_update_counter == 2
    np.testing.assert_array_equal(front.get()["genotype"], genotype)
    np.testing.assert_array_equal(front.get()["fitness"], repeated)


@pytest.mark.parametrize("shift", [-1.0, 1.0, 1e-12])
def test_pareto_front_any_fitness_change_resets_stability_counter(shift: float) -> None:
    front = ParetoFront()
    fitness = np.array([[1.0, 3.0], [3.0, 1.0]])
    front._update(fitness, fitness, fitness)
    front._update(fitness, fitness, fitness)
    assert front._no_update_counter == 1

    changed = fitness + shift
    front._update(changed, changed, changed)
    assert front._no_update_counter == 0
    front._update(changed, changed, changed)
    assert front._no_update_counter == 1


def test_front_tolerance_accumulates_small_shifts_from_reference() -> None:
    front = ParetoFront()
    fitness = np.array([[1.0, 3.0], [3.0, 1.0]])
    front._update(fitness, fitness, fitness, eps=0.1)
    for shift, expected_counter in ((0.04, 1), (0.08, 2), (0.12, 0)):
        changed = (fitness + shift)[::-1]
        front._update(changed, changed, changed, eps=0.1)
        assert front._no_update_counter == expected_counter
        np.testing.assert_array_equal(front.get()["fitness"], changed)


def test_front_tolerance_requires_coverage_in_both_directions() -> None:
    front = ParetoFront()
    fitness = np.array([[1.0, 3.0], [3.0, 1.0]])
    front._update(fitness, fitness, fitness, eps=0.1)
    nearby = np.array([[3.04, 0.96], [1.04, 2.96], [3.06, 0.94]])
    front._update(nearby, nearby, nearby, eps=0.1)
    assert front._no_update_counter == 1
    expanded = np.vstack((fitness, [5.0, 0.0]))
    front._update(expanded, expanded, expanded, eps=0.1)
    assert front._no_update_counter == 0
