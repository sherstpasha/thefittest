from collections import Counter
import json
from importlib import import_module
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from experiments.hyperparameter_search.plan_search_experiment import SearchPlan, optimizer_catalog


def test_all_optimizer_configurations_follow_budget() -> None:
    optimizers = import_module("thefittest.optimizers")
    catalog = optimizer_catalog()
    assert Counter(c["method"] for c in catalog) == {
        "GA": 165, "SHAGA": 60, "PDPGA": 1, "SelfCGA": 1, "SelfCSHAGA": 1, "PDPSHAGA": 1,
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
    assert plan.budgets == (25, 50, 75, 100, 125, 150, 175, 200)
    counts = Counter(job["method"] for job in plan.jobs())
    assert counts == {
        "GA": 198000, "SHAGA": 72000, "PDPGA": 1200, "SelfCGA": 1200,
        "SelfCSHAGA": 1200, "PDPSHAGA": 1200, "Random": 1200, "Grid": 150,
    }
    assert sum(counts.values()) == plan.summary()["total_searches"] == 276150
    jobs = list(islice(plan.jobs(), 230))
    assert len({job["id"] for job in jobs}) == 230
    assert all(job["nominal_budget"] == 625 for job in jobs)


def test_manifest_reuse_and_seed_validation(tmp_path: Path) -> None:
    plan = SearchPlan((42,))
    assert plan.write(tmp_path, False) == plan.write(tmp_path, False)
    with pytest.raises(ValueError, match="different plan"):
        SearchPlan((43,)).write(tmp_path, False)
    for seeds in [(), (42, 42), (-1,), (2**32,)]:
        with pytest.raises(ValueError):
            SearchPlan(seeds)
    assert SearchPlan((42, 43, 44)).summary()["total_searches"] == 828150


@pytest.mark.parametrize("n_jobs", [1, 24, -1])
def test_parallelism_settings(tmp_path: Path, n_jobs: int) -> None:
    plan = SearchPlan((42,), n_jobs=n_jobs)
    summary = plan.write(tmp_path, False)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert summary["search_n_jobs"] == n_jobs
    assert summary["total_searches"] == 276150
    assert manifest["parallelism"] == {
        "search_n_jobs": n_jobs,
        "optimizer_n_jobs": 1,
        "estimator_n_jobs": 1,
        "stacking_n_jobs": 1,
        "inner_max_num_threads": 1,
        "backend": "loky",
        "pre_dispatch": "n_jobs",
    }
    with pytest.raises(ValueError, match="different plan"):
        SearchPlan((42,), n_jobs=2).write(tmp_path, False)


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_rejects_invalid_parallelism(n_jobs: int) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        SearchPlan((42,), n_jobs=n_jobs)
