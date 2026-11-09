from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupShuffleSplit,
    KFold,
    ParameterGrid,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR

import thefittest.benchmarks as benchmarks
from thefittest.benchmarks.symbolicregression17 import problems_dict

if __package__:
    from .search_runtime import FeatureGroupCV
else:
    from search_runtime import FeatureGroupCV


@dataclass
class TaskData:
    X: np.ndarray
    y: np.ndarray
    train: np.ndarray
    test: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    classification: bool


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(describe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def describe(value: Any) -> Any:
    if isinstance(value, BaseCrossValidator):
        return {
            "class": type(value).__module__ + "." + type(value).__name__,
            "params": describe(vars(value)),
        }
    if isinstance(value, BaseEstimator):
        return {
            "class": type(value).__module__ + "." + type(value).__name__,
            "params": describe(value.get_params(deep=False)),
        }
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, np.ndarray):
        return describe(value.tolist())
    if isinstance(value, np.generic):
        return describe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    if isinstance(value, Path):
        return str(value)
    return value


def prepare_task(
    task: dict[str, str],
    output: Path,
    samples: int,
    noise: float,
    data_seed: int,
    max_samples: int,
) -> TaskData:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{task['id']}.npz"
    classification = task["kind"] == "classification"
    if path.exists() and path.with_suffix(".json").exists():
        metadata = json.loads(path.with_suffix(".json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["snapshot_sha256"]:
            raise ValueError(f"Data snapshot checksum mismatch: {path}")
        with np.load(path) as data:
            return TaskData(
                data["X"],
                data["y"],
                data["train"],
                data["test"],
                [(data[f"cv_train_{i}"], data[f"cv_test_{i}"]) for i in range(3)],
                classification,
            )
    generation = None
    if task["id"] in problems_dict:
        problem = problems_dict[task["id"]]
        seed = data_seed + int(task["id"][1:])
        rng = np.random.RandomState(seed)
        lower, upper = problem["bounds"]
        X = rng.uniform(low=lower, high=upper, size=(samples, problem["n_vars"]))
        y = problem["function"](X)
        y = y + rng.normal(0, noise * y.std(), size=len(y))
        generation = dict(
            samples=samples,
            noise_fraction=noise,
            seed=seed,
            bounds=problem["bounds"],
            n_vars=problem["n_vars"],
            sampling="uniform",
        )
    elif task["id"] == "Diabetes":
        X, y = load_diabetes(return_X_y=True)
    else:
        dataset = getattr(benchmarks, task["id"])()
        X, y = dataset.get_X(), dataset.get_y()
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite data in {task['id']}")
    original_samples = len(y)
    source_hash = joblib.hash((X, y))
    _, retained = np.unique(np.column_stack([X, y]), axis=0, return_index=True)
    retained.sort()
    X, y = X[retained], y[retained]
    duplicate_rows = original_samples - len(y)
    if max_samples and len(y) > max_samples:
        selected, _ = train_test_split(
            np.arange(len(y)),
            train_size=max_samples,
            random_state=42,
            stratify=y if classification else None,
        )
        X, y = X[selected], y[selected]
        retained = retained[selected]
    _, groups = np.unique(X, axis=0, return_inverse=True)
    grouped = len(np.unique(groups)) != len(y)
    if grouped:
        train, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups)
        )
        if classification:
            splitter = StratifiedGroupKFold(3, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import GroupKFold

            splitter = GroupKFold(3)
        splits = list(splitter.split(X[train], y[train], groups[train]))
    else:
        train, test = train_test_split(
            np.arange(len(y)),
            test_size=0.25,
            random_state=42,
            stratify=y if classification else None,
        )
        splitter_type = StratifiedKFold if classification else KFold
        splits = list(splitter_type(3, shuffle=True, random_state=42).split(X[train], y[train]))
    arrays = dict(X=X, y=y, train=train, test=test, retained_source_indices=retained)
    for i, (a, b) in enumerate(splits):
        arrays[f"cv_train_{i}"], arrays[f"cv_test_{i}"] = a, b
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    atomic_json(
        path.with_suffix(".json"),
        dict(
            source=task["source"],
            source_hash=source_hash,
            data_hash=joblib.hash(arrays),
            snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_samples=original_samples,
            samples=len(y),
            duplicate_rows_removed=duplicate_rows,
            grouped_identical_features=grouped,
            generation=generation,
        ),
    )
    return TaskData(X, y, train, test, splits, classification)


def _scalers() -> list[Any]:
    return ["passthrough", StandardScaler(), RobustScaler(), MinMaxScaler()]


def _features(p: int, classification: bool, scaled: bool) -> list[dict[str, list]]:
    choices: list[dict[str, list[Any]]] = [{"features": ["passthrough"]}]
    if scaled and p > 1:
        choices.append(
            {
                "features": [PCA(svd_solver="full")],
                "features__n_components": [0.8, 0.9, 0.95, 0.99],
                "features__whiten": [False, True],
            }
        )
    ks = sorted({math.ceil(p * q) for q in [0.25, 0.5, 0.75]} - {p})
    if ks:
        choices.append(
            {
                "features": [SelectKBest(f_classif if classification else f_regression)],
                "features__k": ks,
            }
        )
    return [dict(choice, scaler=_scalers()) if scaled else choice for choice in choices]


def _stacking(classification: bool, smoke: bool) -> tuple[BaseEstimator, list[dict[str, list]]]:
    svm_choices = []
    for scaler in _scalers():
        for C in [0.1, 1.0, 10.0, 100.0]:
            for gamma in [0.001, 0.01]:
                model = (
                    SVC(C=C, gamma=gamma, cache_size=200, random_state=0)
                    if classification
                    else SVR(C=C, gamma=gamma, epsilon=0.1, cache_size=200)
                )
                estimator = Pipeline([("scaler", scaler), ("model", model)])
                if not classification:
                    estimator = TransformedTargetRegressor(
                        regressor=estimator, transformer=StandardScaler()
                    )
                svm_choices.append(estimator)
    forest_type = RandomForestClassifier if classification else RandomForestRegressor
    boosting_type = (
        HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
    )
    from itertools import product

    forests = [
        forest_type(
            n_estimators=5 if smoke else 200,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=features,
            random_state=0,
            n_jobs=1,
        )
        for depth, leaf, features in product([None, 8], [1, 5], [0.5, 1.0])
    ]
    boosting = [
        boosting_type(
            learning_rate=rate,
            max_leaf_nodes=leaves,
            l2_regularization=l2,
            max_iter=5 if smoke else 200,
            early_stopping=False,
            random_state=0,
        )
        for rate, leaves, l2 in product([0.03, 0.1], [15, 63], [0.0, 1.0])
    ]
    meta = (
        LogisticRegression(solver="lbfgs", max_iter=2000, random_state=0)
        if classification
        else Ridge(random_state=0)
    )
    if "n_jobs" in meta.get_params():
        meta.set_params(n_jobs=1)
    stack_type = StackingClassifier if classification else StackingRegressor
    estimator = stack_type(
        estimators=[("svm", svm_choices[0]), ("forest", forests[0]), ("boosting", boosting[0])],
        final_estimator=Pipeline([("scaler", StandardScaler()), ("model", meta)]),
        cv=FeatureGroupCV(classification),
        n_jobs=1,
    )
    grid = []
    for active in [
        ("svm", "forest"),
        ("svm", "boosting"),
        ("forest", "boosting"),
        ("svm", "forest", "boosting"),
    ]:
        branch = {
            name: values if name in active else ["drop"]
            for name, values in [("svm", svm_choices), ("forest", forests), ("boosting", boosting)]
        }
        parameter = "C" if classification else "alpha"
        branch[f"final_estimator__model__{parameter}"] = np.logspace(
            -4 if classification else -5, 4 if classification else 5, 32
        ).tolist()
        branch["passthrough"] = [False, True]
        grid.append(branch)
    return estimator, grid


def search_space(
    family: str,
    p: int,
    classification: bool,
    smoke: bool = False,
) -> tuple[BaseEstimator, list[dict[str, list]]]:
    if family == "stacking":
        estimator, grid = _stacking(classification, smoke)
    else:
        scaled = family in ("svm", "mlp")
        if family == "svm":
            model = SVC(cache_size=200, random_state=0) if classification else SVR(cache_size=200)
            parameters = {
                "C": np.logspace(-3, 4, 64 if classification else 32).tolist(),
                "gamma": np.logspace(-6, 0, 64 if classification else 32).tolist(),
            }
            parameters.update(
                {"class_weight": [None, "balanced"]}
                if classification
                else {"epsilon": np.logspace(-3, -0.3, 8).tolist()}
            )
        elif family == "forest":
            model_type = RandomForestClassifier if classification else RandomForestRegressor
            model = model_type(random_state=0, n_jobs=1)
            parameters = {
                "n_estimators": [50, 100, 200, 400, 800],
                "max_depth": [None, 3, 5, 8, 12, 16, 24, 32],
                "min_samples_split": [2, 3, 5, 8, 13, 21],
                "min_samples_leaf": [1, 2, 3, 5, 8, 13],
                "max_features": [0.25, 0.4, 0.55, 0.7, 0.85, 1.0],
                "bootstrap": [False, True],
                "ccp_alpha": [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            }
            if classification:
                parameters["class_weight"] = [None, "balanced"]
        elif family == "boosting":
            model_type = (
                HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
            )
            model = model_type(random_state=0, early_stopping=False)
            parameters = {
                "learning_rate": np.logspace(-2, -0.4, 8).tolist(),
                "max_iter": [50, 100, 200, 400, 600, 800],
                "max_leaf_nodes": [7, 15, 31, 63, 95, 127],
                "min_samples_leaf": [3, 5, 10, 20, 40, 80],
                "l2_regularization": [0, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],
                "max_depth": [None, 3, 5, 8, 12],
                "max_bins": [63, 127, 255],
            }
        elif family == "mlp":
            model_type = MLPClassifier if classification else MLPRegressor
            model = model_type(
                random_state=0,
                solver="adam",
                max_iter=5 if smoke else 1000,
                early_stopping=not smoke,
                validation_fraction=0.1,
                n_iter_no_change=20,
                tol=1e-4,
            )
            parameters = {
                "hidden_layer_sizes": [
                    (16,),
                    (32,),
                    (64,),
                    (128,),
                    (256,),
                    (32, 16),
                    (64, 32),
                    (64, 64),
                    (128, 64),
                    (128, 128),
                    (64, 32, 16),
                    (128, 64, 32),
                ],
                "activation": ["relu", "tanh", "logistic"],
                "alpha": np.logspace(-7, -1, 8).tolist(),
                "learning_rate_init": np.logspace(-5, -2, 8).tolist(),
                "batch_size": [16, 32, 64, 128],
            }
        else:
            raise ValueError(f"Unknown family: {family}")
        steps = ([("scaler", "passthrough")] if scaled else []) + [
            ("features", "passthrough"),
            ("model", model),
        ]
        estimator = Pipeline(steps)
        grid = [
            dict({f"model__{key}": values for key, values in parameters.items()}, **branch)
            for branch in _features(p, classification, scaled)
        ]
        if not classification and scaled:
            estimator = TransformedTargetRegressor(
                regressor=estimator, transformer=StandardScaler()
            )
            grid = [
                {f"regressor__{key}": values for key, values in branch.items()} for branch in grid
            ]
    if smoke:
        grid = [
            {
                key: values[:2] if i < 2 else values[:1]
                for i, (key, values) in enumerate(branch.items())
            }
            for branch in grid
        ]
        for branch in grid:
            for key in branch:
                if key.endswith("n_estimators") or key.endswith("max_iter"):
                    branch[key] = [5]
    ParameterGrid(grid)
    return estimator, grid
