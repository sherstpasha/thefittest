from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from joblib import hash as joblib_hash

from sklearn.base import BaseEstimator
from sklearn.base import is_classifier
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import BaseSearchCV

from ..base._ea import MultiObjectiveEvolutionaryAlgorithm
from ..base._mlp import check_optimizer_args
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHAGA


class _ParameterGridEncoder:
    """Decode binary blocks as big-endian indices modulo the number of choices."""

    def __init__(self, param_grid: Union[dict[str, Any], list[dict[str, Any]]]) -> None:
        if not isinstance(param_grid, (dict, list)):
            raise TypeError("param_grid must be a dict or a list of dicts.")
        ParameterGrid(param_grid)
        grids = [param_grid] if isinstance(param_grid, dict) else param_grid
        if not grids:
            raise ValueError("param_grid must contain at least one grid.")
        self._values = [
            {name: tuple(values) for name, values in sorted(grid.items())} for grid in grids
        ]
        self._bits = [
            {name: (len(values) - 1).bit_length() for name, values in grid.items()}
            for grid in self._values
        ]
        self._selector_bits = (len(grids) - 1).bit_length()
        self._starts: list[int] = []
        self.str_len = self._selector_bits
        for bits in self._bits:
            self._starts.append(self.str_len)
            self.str_len += sum(bits.values())

    @staticmethod
    def _decode_index(bits: NDArray[np.int8], size: int) -> int:
        index = 0
        for bit in bits:
            index = (index * 2 + int(bit)) % size
        return index

    def transform(self, population: NDArray[np.int8]) -> NDArray[np.object_]:
        if population.ndim != 2 or population.shape[1] != self.str_len:
            raise ValueError(f"population must have shape (n_individuals, {self.str_len}).")
        if not np.all((population == 0) | (population == 1)):
            raise ValueError("population must contain only binary values.")

        candidates = np.empty(len(population), dtype=object)
        for row, individual in enumerate(population):
            grid_index = self._decode_index(individual[: self._selector_bits], len(self._values))
            parameters = {}
            start = self._starts[grid_index]
            for name, values in self._values[grid_index].items():
                stop = start + self._bits[grid_index][name]
                index = self._decode_index(individual[start:stop], len(values))
                parameters[name] = values[index]
                start = stop
            candidates[row] = parameters
        return candidates


class GeneticSearchCV(BaseSearchCV):
    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Union[dict[str, Any], list[dict[str, Any]]],
        *,
        optimizer: type[Union[GeneticAlgorithm, SHAGA]] = GeneticAlgorithm,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        scoring: Optional[
            Union[str, Callable[..., Any], list[str], tuple[str, ...], dict[str, Any]]
        ] = None,
        fitness_metric: Optional[str] = None,
        n_jobs: Optional[int] = None,
        refit: Union[bool, str, Callable[..., int]] = True,
        cv: Optional[Union[int, BaseCrossValidator, Iterable[tuple[ArrayLike, ArrayLike]]]] = None,
        verbose: int = 0,
        pre_dispatch: Union[int, str] = "2*n_jobs",
        error_score: Union[str, float] = np.nan,
        return_train_score: bool = False,
    ) -> None:
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.random_state = random_state
        self.fitness_metric = fitness_metric

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, **params: Any
    ) -> GeneticSearchCV:
        original_cv = self.cv
        self.cv = _CachedCV(check_cv(original_cv, y, classifier=is_classifier(self.estimator)))
        try:
            return super().fit(X, y, **params)
        finally:
            self.cv = original_cv

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.fitness_metric is not None and (
            not isinstance(self.fitness_metric, str) or not self.fitness_metric
        ):
            raise ValueError("fitness_metric must be a non-empty string or None.")
        if (
            not isinstance(self.optimizer, type)
            or not issubclass(self.optimizer, (GeneticAlgorithm, SHAGA))
            or issubclass(
                self.optimizer, (GeneticProgramming, MultiObjectiveEvolutionaryAlgorithm)
            )
        ):
            raise ValueError("optimizer must be a single-objective genetic algorithm class.")

        if self.optimizer_args is not None:
            if not isinstance(self.optimizer_args, dict):
                raise TypeError("optimizer_args must be a dict or None.")
            check_optimizer_args(
                self.optimizer_args,
                args_auto_defined=[
                    "fitness_function",
                    "fitness_function_args",
                    "genotype_to_phenotype",
                    "genotype_to_phenotype_args",
                    "str_len",
                    "init_population",
                    "minimization",
                ],
                args_in_class=["n_jobs", "random_state"],
            )

    def _run_search(self, evaluate_candidates: Callable[..., dict[str, Any]]) -> None:
        encoder = _ParameterGridEncoder(self.param_grid)
        metric = self.fitness_metric
        if metric is None and isinstance(self.refit, str):
            metric = self.refit
        if isinstance(self.scoring, (dict, list, tuple)):
            if metric is None:
                raise ValueError(
                    "Set fitness_metric or a string refit when scoring contains multiple metrics."
                )
            if metric not in self.scoring:
                raise ValueError(f"Fitness metric {metric!r} is not present in scoring.")
        elif not callable(self.scoring) and self.fitness_metric is None:
            metric = None
        evaluator = _CandidateEvaluator(evaluate_candidates, metric)
        if encoder.str_len == 0:
            evaluator.evaluate(encoder.transform(np.empty((1, 0), dtype=np.int8)))
            return

        optimizer_args = dict(self.optimizer_args or {})
        optimizer_args.setdefault("iters", 30)
        optimizer_args.setdefault("pop_size", 20)
        optimizer = self.optimizer(
            **optimizer_args,
            fitness_function=evaluator.evaluate,
            genotype_to_phenotype=encoder.transform,
            str_len=encoder.str_len,
            minimization=False,
            n_jobs=1,
            random_state=self.random_state,
        )
        optimizer.fit()


class _CandidateEvaluator:
    def __init__(
        self,
        evaluate_candidates: Callable[..., dict[str, Any]],
        fitness_metric: Optional[str] = None,
    ) -> None:
        self._evaluate_candidates = evaluate_candidates
        self._fitness_metric = fitness_metric
        self._scores: dict[str, float] = {}

    def evaluate(self, population: NDArray[np.object_]) -> NDArray[np.float64]:
        keys = [joblib_hash(parameters) for parameters in population]
        pending = {}
        for key, parameters in zip(keys, population):
            if key not in self._scores:
                pending[key] = parameters

        if pending:
            results = self._evaluate_candidates(list(pending.values()))
            score_key = f"mean_test_{self._fitness_metric or 'score'}"
            if score_key not in results:
                if self._fitness_metric is None:
                    raise ValueError(
                        "Set fitness_metric or a string refit "
                        "when scoring returns multiple metrics."
                    )
                raise ValueError(
                    f"Fitness metric {self._fitness_metric!r} is not present in scores."
                )
            scores = np.asarray(results[score_key][-len(pending) :], dtype=float)
            if np.any(np.isinf(scores)):
                raise ValueError(
                    "GeneticSearchCV does not support infinite cross-validation scores."
                )
            self._scores.update(zip(pending, scores.tolist()))

        scores = np.array([self._scores[key] for key in keys], dtype=np.float64)
        # A fixed monotone scale keeps valid scores above failed candidates across generations.
        fitness = 2.0 + (2.0 / np.pi) * np.arctan(scores)
        fitness[np.isnan(scores)] = 0.0
        return fitness


class _CachedCV(BaseCrossValidator):
    def __init__(self, splitter: BaseCrossValidator) -> None:
        self._splitter = splitter
        self._splits: Optional[list[tuple[NDArray[np.intp], NDArray[np.intp]]]] = None

    def split(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, **params: Any
    ) -> Iterable[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        if self._splits is None:
            self._splits = [
                (train.copy(), test.copy())
                for train, test in self._splitter.split(X, y, **params)
            ]
        return iter(self._splits)

    def get_n_splits(
        self, X: Optional[ArrayLike] = None, y: Optional[ArrayLike] = None, **params: Any
    ) -> int:
        return self._splitter.get_n_splits(X, y, **params)

    def get_metadata_routing(self) -> Any:
        return self._splitter.get_metadata_routing()
