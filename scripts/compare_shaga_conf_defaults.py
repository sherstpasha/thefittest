from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from thefittest.benchmarks import OneMax
from thefittest.benchmarks import Rastrigin
from thefittest.benchmarks import Sphere
from thefittest.benchmarks import IrisDataset
from thefittest.classifiers import MLPEAClassifier
from thefittest.optimizers._shaga import SHAGA as BaseSHAGA
from thefittest.optimizers._shaga_conf import SHAGA as ConfSHAGA
from thefittest.utils.transformations import GrayCode


@dataclass(frozen=True)
class Problem:
    name: str
    fitness_function: Callable[[NDArray], NDArray[np.float64]]
    dimensions: tuple[int, ...]
    minimization: bool
    left_border: float | None = None
    right_border: float | None = None


@dataclass(frozen=True)
class AlgorithmCase:
    name: str
    optimizer: type
    extra_params: dict


PROBLEMS = (
    Problem(
        name="OneMax",
        fitness_function=OneMax(),
        dimensions=(32, 64, 128),
        minimization=False,
    ),
    Problem(
        name="Sphere",
        fitness_function=Sphere(),
        dimensions=(2, 5, 10),
        minimization=True,
        left_border=-5.0,
        right_border=5.0,
    ),
    Problem(
        name="Rastrigin",
        fitness_function=Rastrigin(),
        dimensions=(2, 5, 10),
        minimization=True,
        left_border=-5.12,
        right_border=5.12,
    ),
)

ALGORITHMS = (
    AlgorithmCase("base_shaga_default", BaseSHAGA, {}),
    AlgorithmCase("conf_shaga_default", ConfSHAGA, {}),
)


def build_encoding(
    problem: Problem, dimension: int, bits_per_variable: int
) -> tuple[int, Callable | None]:
    if not problem.minimization:
        return dimension, None

    encoder = GrayCode().fit(
        left_border=problem.left_border,
        right_border=problem.right_border,
        num_variables=dimension,
        bits_per_variable=bits_per_variable,
    )
    return int(encoder.get_str_len()), encoder.transform


def objective_value(optimizer, minimization: bool) -> float:
    fitness = float(optimizer.get_fittest()["fitness"])
    return -fitness if minimization else fitness


def objective_curve(optimizer, minimization: bool) -> NDArray[np.float64]:
    curve = np.asarray(optimizer.get_stats()["max_fitness"], dtype=np.float64)
    return -curve if minimization else curve


def _as_float_array(value) -> NDArray[np.float64] | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def stats_to_series(optimizer, minimization: bool) -> dict[tuple[str, str], NDArray[np.float64]]:
    series = {}
    for stat_name, stat_values in optimizer.get_stats().items():
        reducers = {"mean": [], "min": [], "max": []}
        for value in stat_values:
            array = _as_float_array(value)
            if array is None:
                break

            if stat_name in {"fitness", "max_fitness"} and minimization:
                array = -array

            reducers["mean"].append(float(np.mean(array)))
            reducers["min"].append(float(np.min(array)))
            reducers["max"].append(float(np.max(array)))
        else:
            for reducer_name, reducer_values in reducers.items():
                series[(stat_name, reducer_name)] = np.asarray(reducer_values, dtype=np.float64)
    return series


def run_case(
    problem: Problem,
    dimension: int,
    algorithm: AlgorithmCase,
    seed: int,
    iters: int,
    pop_size: int,
    bits_per_variable: int,
) -> tuple[dict, NDArray[np.float64], dict[tuple[str, str], NDArray[np.float64]]]:
    str_len, genotype_to_phenotype = build_encoding(problem, dimension, bits_per_variable)
    optimizer = algorithm.optimizer(
        fitness_function=problem.fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        genotype_to_phenotype=genotype_to_phenotype,
        minimization=problem.minimization,
        keep_history=True,
        random_state=seed,
        **algorithm.extra_params,
    )
    optimizer.fit()

    row = {
        "problem": problem.name,
        "dimension": dimension,
        "algorithm": algorithm.name,
        "seed": seed,
        "iters": iters,
        "pop_size": pop_size,
        "str_len": str_len,
        "objective": objective_value(optimizer, problem.minimization),
        "calls": optimizer.get_calls(),
    }
    return (
        row,
        objective_curve(optimizer, problem.minimization),
        stats_to_series(optimizer, problem.minimization),
    )


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(r["problem"], r["dimension"], r["algorithm"]) for r in rows})
    for problem, dimension, algorithm in keys:
        values = np.array(
            [
                float(r["objective"])
                for r in rows
                if r["problem"] == problem
                and r["dimension"] == dimension
                and r["algorithm"] == algorithm
            ],
            dtype=np.float64,
        )
        summary.append(
            {
                "problem": problem,
                "dimension": dimension,
                "algorithm": algorithm,
                "mean_objective": float(np.mean(values)),
                "std_objective": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "best_objective": float(np.min(values)),
                "worst_objective": float(np.max(values)),
                "runs": len(values),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    out_dir: Path, curves: dict[tuple[str, int, str], list[NDArray[np.float64]]]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_keys = sorted({(k[0], k[1]) for k in curves})

    for problem, dimension in grouped_keys:
        plt.figure(figsize=(8, 4.5))
        for algorithm in [algorithm.name for algorithm in ALGORITHMS]:
            values = curves.get((problem, dimension, algorithm))
            if not values:
                continue

            stacked = np.vstack(values)
            mean_curve = np.mean(stacked, axis=0)
            if len(values) > 1:
                std_curve = np.std(stacked, axis=0, ddof=1)
            else:
                std_curve = np.zeros_like(mean_curve)
            x = np.arange(len(mean_curve))

            plt.plot(x, mean_curve, label=algorithm)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)

        plt.xlabel("generation")
        plt.ylabel("objective")
        plt.title(f"{problem}, dim={dimension}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{problem}_dim{dimension}_overlay.png", dpi=140)
        plt.close()


def plot_problem_collages(
    out_dir: Path, curves: dict[tuple[str, int, str], list[NDArray[np.float64]]]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = sorted({k[0] for k in curves})

    for problem in problems:
        dimensions = sorted({k[1] for k in curves if k[0] == problem})
        fig, axes = plt.subplots(
            1,
            len(dimensions),
            figsize=(5 * len(dimensions), 4),
            squeeze=False,
        )

        for ax, dimension in zip(axes[0], dimensions):
            for algorithm in [algorithm.name for algorithm in ALGORITHMS]:
                values = curves.get((problem, dimension, algorithm))
                if not values:
                    continue

                mean_curve = np.mean(np.vstack(values), axis=0)
                ax.plot(mean_curve, label=algorithm)

            ax.set_title(f"dim={dimension}")
            ax.set_xlabel("generation")
            ax.set_ylabel("objective")
            ax.grid(alpha=0.3)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.suptitle(problem)
        fig.legend(handles, labels, loc="lower center", ncol=len(ALGORITHMS))
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        fig.savefig(out_dir / f"{problem}_collage.png", dpi=140)
        plt.close(fig)


def plot_stat_curves(
    out_dir: Path,
    stat_curves: dict[tuple[str, int, str, str, str], list[NDArray[np.float64]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_keys = sorted({(k[0], k[1], k[3], k[4]) for k in stat_curves})

    for problem, dimension, stat_name, reducer_name in grouped_keys:
        plt.figure(figsize=(8, 4.5))
        for algorithm in [algorithm.name for algorithm in ALGORITHMS]:
            key = (problem, dimension, algorithm, stat_name, reducer_name)
            values = stat_curves.get(key)
            if not values:
                continue

            stacked = np.vstack(values)
            mean_curve = np.mean(stacked, axis=0)
            if len(values) > 1:
                std_curve = np.std(stacked, axis=0, ddof=1)
            else:
                std_curve = np.zeros_like(mean_curve)
            x = np.arange(len(mean_curve))

            plt.plot(x, mean_curve, label=algorithm)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)

        plt.xlabel("generation")
        plt.ylabel(f"{stat_name}.{reducer_name}")
        plt.title(f"{problem}, dim={dimension}: {stat_name}.{reducer_name}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{problem}_dim{dimension}_{stat_name}_{reducer_name}.png", dpi=140)
        plt.close()


def run_neural_case(
    algorithm: AlgorithmCase,
    seed: int,
    iters: int,
    pop_size: int,
) -> tuple[dict, dict[tuple[str, str], NDArray[np.float64]]]:
    data = IrisDataset()
    X = minmax_scale(data.get_X())
    y = data.get_y()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    model = MLPEAClassifier(
        n_iter=iters,
        pop_size=pop_size,
        hidden_layers=(5,),
        weights_optimizer=algorithm.optimizer,
        weights_optimizer_args=algorithm.extra_params,
        random_state=seed,
        device="cpu",
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = float(f1_score(y_test, prediction, average="macro"))

    row = {
        "problem": "IrisMLPEA",
        "dimension": 0,
        "algorithm": algorithm.name,
        "seed": seed,
        "iters": iters,
        "pop_size": pop_size,
        "str_len": "",
        "objective": score,
        "calls": "",
    }
    return row, stats_to_series(model, minimization=True)


def plot_neural_scores(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [algorithm.name for algorithm in ALGORITHMS]
    means = []
    stds = []

    for label in labels:
        values = np.array(
            [
                float(row["objective"])
                for row in rows
                if row["problem"] == "IrisMLPEA" and row["algorithm"] == label
            ],
            dtype=np.float64,
        )
        if len(values) == 0:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("macro F1")
    plt.title("Iris MLPEAClassifier")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "IrisMLPEA_macro_f1.png", dpi=140)
    plt.close()


def print_code_paths() -> None:
    print("Code paths:")
    print(
        "  base_shaga_default: _shaga.SHAGA -> tournament_selection(fitness_i, fitness_i, 2, 1) -> binomialGA -> flip_mutation"
    )
    print(
        "  conf_shaga_default: _shaga_conf.SHAGA(selection='tournament_2', crossover='uniform_1') -> tournament_selection -> binomial_selfshaga -> flip_mutation"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--bits-per-variable", type=int, default=12)
    parser.add_argument("--nn-runs", type=int, default=5)
    parser.add_argument("--nn-iters", type=int, default=500)
    parser.add_argument("--nn-pop-size", type=int, default=30)
    parser.add_argument("--skip-nn", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/shaga_conf_compare"))
    args = parser.parse_args()

    rows: list[dict] = []
    curves: dict[tuple[str, int, str], list[NDArray[np.float64]]] = {}
    stat_curves: dict[tuple[str, int, str, str, str], list[NDArray[np.float64]]] = {}

    print_code_paths()
    for problem in PROBLEMS:
        for dimension in problem.dimensions:
            for seed in range(args.runs):
                for algorithm in ALGORITHMS:
                    row, curve, stats = run_case(
                        problem=problem,
                        dimension=dimension,
                        algorithm=algorithm,
                        seed=seed,
                        iters=args.iters,
                        pop_size=args.pop_size,
                        bits_per_variable=args.bits_per_variable,
                    )
                    rows.append(row)
                    key = (problem.name, dimension, algorithm.name)
                    curves.setdefault(key, []).append(curve)
                    for (stat_name, reducer_name), stat_curve in stats.items():
                        stat_key = (
                            problem.name,
                            dimension,
                            algorithm.name,
                            stat_name,
                            reducer_name,
                        )
                        stat_curves.setdefault(stat_key, []).append(stat_curve)
            print(f"finished {problem.name}, dimension={dimension}")

    summary_rows = summarize(rows)
    write_csv(args.out_dir / "runs.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    plot_curves(args.out_dir / "plots", curves)
    plot_problem_collages(args.out_dir / "collages", curves)
    plot_stat_curves(args.out_dir / "stat_plots", stat_curves)

    neural_rows: list[dict] = []
    if not args.skip_nn:
        for seed in range(args.nn_runs):
            for algorithm in ALGORITHMS:
                row, stats = run_neural_case(
                    algorithm=algorithm,
                    seed=seed,
                    iters=args.nn_iters,
                    pop_size=args.nn_pop_size,
                )
                neural_rows.append(row)
                for (stat_name, reducer_name), stat_curve in stats.items():
                    stat_key = (
                        row["problem"],
                        row["dimension"],
                        algorithm.name,
                        stat_name,
                        reducer_name,
                    )
                    stat_curves.setdefault(stat_key, []).append(stat_curve)
            print(f"finished IrisMLPEA, seed={seed}")

        write_csv(args.out_dir / "neural_runs.csv", neural_rows)
        plot_neural_scores(args.out_dir / "neural_plots", neural_rows)
        plot_stat_curves(args.out_dir / "stat_plots", stat_curves)

    print()
    print("Summary, objective is minimized for Sphere/Rastrigin and maximized for OneMax:")
    for row in summary_rows:
        print(
            f"{row['problem']:9s} dim={row['dimension']:3d} "
            f"{row['algorithm']:26s} mean={row['mean_objective']:.6g} "
            f"std={row['std_objective']:.6g}"
        )
    print()
    print(f"Wrote: {args.out_dir / 'runs.csv'}")
    print(f"Wrote: {args.out_dir / 'summary.csv'}")
    print(f"Wrote plots to: {args.out_dir / 'plots'}")
    print(f"Wrote collages to: {args.out_dir / 'collages'}")
    print(f"Wrote stat plots to: {args.out_dir / 'stat_plots'}")
    if not args.skip_nn:
        print(f"Wrote: {args.out_dir / 'neural_runs.csv'}")
        print(f"Wrote neural plots to: {args.out_dir / 'neural_plots'}")


if __name__ == "__main__":
    main()
