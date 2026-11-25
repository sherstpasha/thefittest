from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._nsgaii import NSGAII
from ..utils.random import random_weighted_sample
from ..utils.random import random_sample


class SelfCNSGAII(NSGAII):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        str_len: int,
        minimization: Tuple[bool, ...],
        mutation_rate: float = 0.05,
        parents_num: int = 2,
        selections: Tuple[str, ...] = ("rank_crowding", "dominance_crowding"),
        crossovers: Tuple[str, ...] = (
            "empty",
            "one_point",
            "two_point",
            "uniform_2",
            "uniform_7",
            "uniform_tour_3",
            "uniform_tour_7",
        ),
        mutations: Tuple[str, ...] = ("weak", "average", "strong"),
        init_population: Optional[NDArray[np.byte]] = None,
        K: float = 2.0,
        selection_threshold_proba: float = 0.05,
        crossover_threshold_proba: float = 0.05,
        mutation_threshold_proba: float = 0.05,
        genotype_to_phenotype: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        optimal_value: Optional[Tuple[float, ...]] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        on_generation: Optional[Callable] = None,
        fitness_update_eps: float = 0.0,
        use_fitness_cache: bool = False,
        fitness_cache_size: Optional[int] = 1000,
    ) -> None:
        if not np.isfinite(K) or K < 0:
            raise ValueError("K must be finite and non-negative.")
        for kind, names in (
            ("selection", selections),
            ("crossover", crossovers),
            ("mutation", mutations),
        ):
            if not names or len(names) != len(set(names)):
                raise ValueError(f"{kind} operators must be non-empty and unique.")
        super().__init__(
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            minimization=minimization,
            selection=selections[0],
            crossover=crossovers[0],
            mutation=mutations[0],
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            init_population=init_population,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            n_jobs=n_jobs,
            fitness_function_args=fitness_function_args,
            genotype_to_phenotype_args=genotype_to_phenotype_args,
            random_state=random_state,
            on_generation=on_generation,
            fitness_update_eps=fitness_update_eps,
            use_fitness_cache=use_fitness_cache,
            fitness_cache_size=fitness_cache_size,
        )
        self._K: float = K
        self._thresholds: Dict[str, float] = {
            "selection": selection_threshold_proba,
            "crossover": crossover_threshold_proba,
            "mutation": mutation_threshold_proba,
        }
        for kind, names, pool in (
            ("selection", selections, self._selection_pool),
            ("crossover", crossovers, self._crossover_pool),
            ("mutation", mutations, self._mutation_pool),
        ):
            unknown = set(names) - pool.keys()
            if unknown:
                raise ValueError(f"Unknown {kind} operators: {sorted(unknown)}.")
            threshold = self._thresholds[kind]
            if not np.isfinite(threshold) or not 0 <= threshold <= 1 / len(names):
                raise ValueError(f"{kind} threshold must be between 0 and 1 / operator count.")

        self._selection_set = {name: self._selection_pool[name] for name in sorted(selections)}
        self._crossover_set = {name: self._crossover_pool[name] for name in sorted(crossovers)}
        self._mutation_set = {name: self._mutation_pool[name] for name in sorted(mutations)}
        self._selection_proba: Dict[str, float] = {
            name: 1 / len(self._selection_set) for name in self._selection_set
        }
        self._crossover_proba: Dict[str, float] = {
            name: 1 / len(self._crossover_set) for name in self._crossover_set
        }
        empty_crossover = next(
            (name for name in ("empty", "gp_empty") if name in self._crossover_set), None
        )
        if empty_crossover is not None and len(self._crossover_set) > 1:
            self._crossover_proba = {
                name: 0.1 if name == empty_crossover else 0.9 / (len(self._crossover_set) - 1)
                for name in self._crossover_set
            }
        self._mutation_proba: Dict[str, float] = {
            name: 1 / len(self._mutation_set) for name in self._mutation_set
        }
        self._selection_operators: NDArray[np.str_]
        self._crossover_operators: NDArray[np.str_]
        self._mutation_operators: NDArray[np.str_]
        self._selection_success_rates: Dict[str, float] = {}
        self._crossover_success_rates: Dict[str, float] = {}
        self._mutation_success_rates: Dict[str, float] = {}

    def _choice_operators(self, proba_dict: Dict[str, float]) -> NDArray[np.str_]:
        operators = np.array(list(proba_dict))
        probabilities = np.array(list(proba_dict.values()), dtype=np.float64)
        indices = random_weighted_sample(probabilities, self._pop_size, True)
        return operators[indices]

    def _get_offspring_population(self) -> NDArray[Any]:
        self._selection_operators = self._choice_operators(self._selection_proba)
        self._crossover_operators = self._choice_operators(self._crossover_proba)
        self._mutation_operators = self._choice_operators(self._mutation_proba)
        return np.array(
            [
                self._get_new_individ_g(
                    self._selection_operators[i],
                    self._crossover_operators[i],
                    self._mutation_operators[i],
                )
                for i in range(self._pop_size)
            ],
            dtype=self._population_g_i.dtype,
        )

    @staticmethod
    def _get_operator_success_rates(
        operators: NDArray[np.str_], pareto_ranks: NDArray[np.int64]
    ) -> Dict[str, float]:
        """Calculate the fraction of rank-one offspring for each used operator.

        Parameters
        ----------
        operators : NDArray[np.str_]
            Operator names aligned with all offspring, including rejected ones.
        pareto_ranks : NDArray[np.int64]
            Offspring ranks computed in the parent-offspring union before
            survival. Lower ranks are better; the first front has rank 1.
        """
        if operators.ndim != 1 or pareto_ranks.ndim != 1:
            raise ValueError("operators and pareto_ranks must be one-dimensional.")
        if len(operators) != len(pareto_ranks):
            raise ValueError("operators and pareto_ranks must have the same length.")
        success_rates = {}
        for operator in np.unique(operators):
            operator_ranks = pareto_ranks[operators == operator]
            success_rates[str(operator)] = float(np.mean(operator_ranks == 1))
        return success_rates

    def _get_new_proba(
        self,
        proba_dict: Dict[str, float],
        success_rates: Dict[str, float],
        threshold: float,
    ) -> Dict[str, float]:
        if not success_rates or max(success_rates.values()) == 0 or self._K == 0:
            return proba_dict.copy()
        if len(proba_dict) == 1:
            return proba_dict.copy()

        best_rate = max(success_rates.values())
        best_operators = [name for name, rate in success_rates.items() if rate == best_rate]
        winner = best_operators[random_sample(len(best_operators), 1, False)[0]]
        names = list(proba_dict)
        probabilities = np.array(list(proba_dict.values()), dtype=np.float64)
        step = self._K / self._iters
        probabilities[names.index(winner)] += step
        probabilities -= step / len(names)

        # Normalize only the mass above the lower bounds, preserving those bounds.
        excess = np.maximum(probabilities - threshold, 0.0)
        remaining = 1.0 - len(names) * threshold
        if remaining == 0:
            probabilities = np.full(len(names), threshold)
        else:
            probabilities = threshold + remaining * excess / excess.sum()
        return dict(zip(names, probabilities))

    def _adapt(self) -> None:
        self._selection_proba = self._get_new_proba(
            self._selection_proba, self._selection_success_rates, self._thresholds["selection"]
        )
        self._crossover_proba = self._get_new_proba(
            self._crossover_proba, self._crossover_success_rates, self._thresholds["crossover"]
        )
        self._mutation_proba = self._get_new_proba(
            self._mutation_proba, self._mutation_success_rates, self._thresholds["mutation"]
        )

    def _update_data(self) -> None:
        super()._update_data()
        self._update_stats(
            s_proba=self._selection_proba,
            c_proba=self._crossover_proba,
            m_proba=self._mutation_proba,
        )

    def _replace_population(
        self,
        population_g: NDArray[np.byte],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
        ranks: NDArray[np.int64],
        distances: NDArray[np.float64],
        survivors: NDArray[np.int64],
    ) -> None:
        # Parents occupy the first N rows; operator labels describe the offspring rows.
        offspring_ranks = ranks[self._pop_size :]
        self._selection_success_rates = self._get_operator_success_rates(
            self._selection_operators, offspring_ranks
        )
        self._crossover_success_rates = self._get_operator_success_rates(
            self._crossover_operators, offspring_ranks
        )
        self._mutation_success_rates = self._get_operator_success_rates(
            self._mutation_operators, offspring_ranks
        )
        super()._replace_population(
            population_g, population_ph, fitness, ranks, distances, survivors
        )
        self._adapt()
