from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any, Iterator

from thefittest.optimizers import (
    GeneticAlgorithm,
    PDPGA,
    PDPSHAGA,
    SelfCGA,
    SelfCSHAGA,
    SHAGA,
)


CLASSIFICATION_TASKS = (
    "BanknoteDataset",
    "BreastCancerDataset",
    "CreditRiskDataset",
    "DigitsDataset",
    "IrisDataset",
    "UserKnowladgeDataset",
    "WineDataset",
    "TwoNormDataset",
    "RingNormDataset",
    "TextureDataset",
    "AdultCensusDataset",
)
FAMILIES = {
    "svm": ("SVC", "SVR"),
    "forest": ("RandomForestClassifier", "RandomForestRegressor"),
    "boosting": ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"),
    "mlp": ("MLPClassifier", "MLPRegressor"),
    "stacking": ("StackingClassifier", "StackingRegressor"),
}
SELECTIONS = ("proportional", "rank", "tournament_3", "tournament_5", "tournament_7")
GA_CROSSOVERS = (
    "empty",
    "one_point",
    "two_point",
    "uniform_2",
    "uniform_7",
    "uniform_prop_2",
    "uniform_prop_7",
    "uniform_rank_2",
    "uniform_rank_7",
    "uniform_tour_3",
    "uniform_tour_7",
)
SHAGA_CROSSOVERS = (
    "empty",
    "one_point_1",
    "two_point_1",
    "uniform_1",
    "uniform_2",
    "uniform_7",
    "uniform_prop_2",
    "uniform_prop_7",
    "uniform_rank_2",
    "uniform_rank_7",
    "uniform_tour_3",
    "uniform_tour_7",
)
MUTATIONS = ("weak", "average", "strong")
ADAPTIVE_OPTIMIZERS = (PDPGA, SelfCGA, SelfCSHAGA, PDPSHAGA)
OPERATOR_PARAMETERS = {
    "tour_size",
    "parents_num",
    "mutation_rate",
    "elitism",
    "selection",
    "crossover",
    "mutation",
    "selections",
    "crossovers",
    "mutations",
    "K",
    "selection_threshold_proba",
    "crossover_threshold_proba",
    "mutation_threshold_proba",
}


def optimizer_catalog() -> list[dict[str, Any]]:
    configurations = []
    for selection, crossover, mutation in product(SELECTIONS, GA_CROSSOVERS, MUTATIONS):
        configurations.append({
            "id": f"GA__{selection}__{crossover}__{mutation}",
            "method": "GA",
            "optimizer": GeneticAlgorithm.__name__,
            "args": dict(selection=selection, crossover=crossover, mutation=mutation, elitism=True),
        })
    for selection, crossover in product(SELECTIONS, SHAGA_CROSSOVERS):
        configurations.append({
            "id": f"SHAGA__{selection}__{crossover}",
            "method": "SHAGA",
            "optimizer": SHAGA.__name__,
            "args": dict(selection=selection, crossover=crossover, elitism=True),
        })
    for optimizer in ADAPTIVE_OPTIMIZERS:
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(optimizer).parameters.items()
            if name in OPERATOR_PARAMETERS
        }
        configurations.append({
            "id": optimizer.__name__,
            "method": optimizer.__name__,
            "optimizer": optimizer.__name__,
            "args": defaults,
        })
    return configurations


def task_catalog() -> list[dict[str, str]]:
    tasks = [
        {"id": name, "kind": "classification", "source": f"thefittest.benchmarks.{name}"}
        for name in CLASSIFICATION_TASKS
    ]
    tasks.extend(
        {"id": f"F{i}", "kind": "regression",
         "source": f"thefittest.benchmarks.symbolicregression17.F{i}"}
        for i in range(1, 18)
    )
    tasks.extend([
        {"id": "CombinedCycleDataset", "kind": "regression",
         "source": "thefittest.benchmarks.CombinedCycleDataset"},
        {"id": "Diabetes", "kind": "regression", "source": "sklearn.datasets.load_diabetes"},
    ])
    return tasks


class SearchPlan:
    def __init__(self, seeds: tuple[int, ...], n_jobs: int = 1) -> None:
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Provide at least one seed; seeds must be unique.")
        if any(seed < 0 or seed >= 2**32 for seed in seeds):
            raise ValueError("Seeds must be in [0, 2**32).")
        if n_jobs != -1 and n_jobs < 1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        self.seeds = seeds
        self.n_jobs = n_jobs
        self.tasks = task_catalog()
        self.configurations = optimizer_catalog()
        self.budgets = tuple(range(25, 201, 25))

    def summary(self) -> dict[str, Any]:
        pairs = len(self.tasks) * len(FAMILIES)
        per_budget = len(self.configurations) + 1
        return {
            "tasks": len(self.tasks),
            "classification_tasks": sum(t["kind"] == "classification" for t in self.tasks),
            "regression_tasks": sum(t["kind"] == "regression" for t in self.tasks),
            "task_family_pairs": pairs,
            "optimizer_configurations": len(self.configurations),
            "budgets": len(self.budgets),
            "search_seeds": list(self.seeds),
            "search_n_jobs": self.n_jobs,
            "stochastic_searches": pairs * len(self.seeds) * len(self.budgets) * per_budget,
            "grid_searches": pairs,
            "total_searches": pairs * (len(self.seeds) * len(self.budgets) * per_budget + 1),
        }

    def jobs(self) -> Iterator[dict[str, Any]]:
        for task, family in product(self.tasks, FAMILIES):
            for seed, size in product(self.seeds, self.budgets):
                for config in self.configurations:
                    yield {
                        "id": f"{task['id']}__{family}__{config['id']}__p{size}_g{size}__s{seed}",
                        "task": task["id"], "family": family, "method": config["method"],
                        "configuration": config["id"], "seed": seed,
                        "pop_size": size, "iters": size, "nominal_budget": size * size,
                    }
                yield {
                    "id": f"{task['id']}__{family}__Random__b{size * size}__s{seed}",
                    "task": task["id"], "family": family, "method": "Random",
                    "seed": seed, "nominal_budget": size * size,
                }
        for task, family in product(self.tasks, FAMILIES):
            yield {
                "id": f"{task['id']}__{family}__Grid",
                "task": task["id"], "family": family, "method": "Grid",
            }

    def write(self, output: Path, write_jobs: bool) -> dict[str, Any]:
        spec = Path(__file__).with_name("EXPERIMENT_SPEC.md")
        manifest = {
            "kind": "search_job_plan",
            "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
            "versions": {name: version(name) for name in ["thefittest", "scikit-learn", "numpy"]},
            "summary": self.summary(),
            "parallelism": {
                "search_n_jobs": self.n_jobs,
                "optimizer_n_jobs": 1,
                "estimator_n_jobs": 1,
                "stacking_n_jobs": 1,
                "inner_max_num_threads": 1,
                "backend": "loky",
                "pre_dispatch": "n_jobs",
            },
            "tasks": self.tasks,
            "families": {
                name: {"classification": pair[0], "regression": pair[1]}
                for name, pair in FAMILIES.items()
            },
            "optimizer_configurations": self.configurations,
            "budgets": [{"pop_size": q, "iters": q} for q in self.budgets],
        }
        normalized = json.loads(json.dumps(manifest))
        output.mkdir(parents=True, exist_ok=True)
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != normalized:
            raise ValueError("Output contains a different plan; select another --output.")
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        if write_jobs:
            temporary = output / "jobs.tmp"
            count = 0
            with temporary.open("w", encoding="utf-8") as stream:
                for job in self.jobs():
                    stream.write(json.dumps(job, ensure_ascii=False) + "\n")
                    count += 1
            if count != self.summary()["total_searches"]:
                raise RuntimeError("Job count does not match the plan.")
            temporary.replace(output / "jobs.jsonl")
        return self.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the experiment job plan without training."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--jobs", type=int, default=1, help="Parallel CV workers; -1 uses all available CPUs"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "output" / "plan",
    )
    parser.add_argument(
        "--write-jobs", action="store_true", help="Stream the complete job list to JSONL"
    )
    args = parser.parse_args()
    result = SearchPlan(tuple(args.seeds), n_jobs=args.jobs).write(args.output, args.write_jobs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
