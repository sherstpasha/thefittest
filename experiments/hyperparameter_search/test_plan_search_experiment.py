from collections import Counter
from importlib import import_module
from itertools import islice

import numpy as np
import pytest

from experiments.hyperparameter_search.plan_search_experiment import SearchPlan, optimizer_catalog


def test_all_optimizer_configurations_follow_budget() -> None:
    optimizers = import_module("thefittest.optimizers")
    catalog = optimizer_catalog()
    assert Counter(c["method"] for c in catalog) == {
        "GA": 165,
        "SHAGA": 60,
        "PDPGA": 1,
        "SelfCGA": 1,
        "SelfCSHAGA": 1,
        "PDPSHAGA": 1,
    }
    assert len({c["id"] for c in catalog}) == 229

    def fitness(population: np.ndarray) -> np.ndarray:
        return population.sum(axis=1).astype(float)

    for configuration in catalog:
        optimizer = getattr(optimizers, configuration["optimizer"])
        search = optimizer(fitness, 3, 10, 24, random_state=42, **configuration["args"])
        search.fit()
        assert search.get_calls() == 30, configuration["id"]
        assert np.isfinite(search.get_fittest()["fitness"]), configuration["id"]


def test_task_sources_and_complete_job_counts() -> None:
    plan = SearchPlan((42,))
    for task in plan.tasks:
        module, name = task["source"].rsplit(".", 1)
        assert callable(getattr(import_module(module), name))
    assert plan.summary()["classification_tasks"] == 11
    assert plan.summary()["regression_tasks"] == 19
    assert plan.budgets == tuple(range(10, 101, 10))
    counts = Counter(job["method"] for job in plan.jobs())
    assert counts == {
        "GA": 247500,
        "SHAGA": 90000,
        "PDPGA": 1500,
        "SelfCGA": 1500,
        "SelfCSHAGA": 1500,
        "PDPSHAGA": 1500,
        "Random": 1500,
        "Grid": 150,
    }
    assert sum(counts.values()) == plan.summary()["total_searches"] == 345150
    assert next(plan.jobs())["method"] == "Grid"
    jobs = list(islice((job for job in plan.jobs() if job["method"] != "Grid"), 230))
    assert len({job["id"] for job in jobs}) == 230
    assert all(job["nominal_budget"] == 100 for job in jobs)


def test_seed_validation() -> None:
    for seeds in [(), (42, 42), (-1,), (2**32,)]:
        with pytest.raises(ValueError):
            SearchPlan(seeds)
    assert SearchPlan((42, 43, 44)).summary()["total_searches"] == 1035150


@pytest.mark.parametrize("n_jobs", [1, 24, -1])
def test_parallelism_settings(n_jobs: int) -> None:
    plan = SearchPlan((42,), n_jobs=n_jobs)
    summary = plan.summary()
    assert summary["search_n_jobs"] == n_jobs
    assert summary["total_searches"] == 345150


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_rejects_invalid_parallelism(n_jobs: int) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        SearchPlan((42,), n_jobs=n_jobs)
