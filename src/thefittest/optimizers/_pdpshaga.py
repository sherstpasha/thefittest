from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..utils import numpy_group_by
from ..base._ea import EvolutionaryAlgorithm
from ._pdpga import PDPGA
from ._shaga import SHAGA


class PDPSHAGA(SHAGA, PDPGA):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        str_len: int,
        tour_size: int = 2,
        parents_num: int = 2,
        elitism: bool = True,
        selections: Tuple[str, ...] = (
            "proportional",
            "rank",
            "tournament_2",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = (
            "empty",
            "one_point_1",
            "one_point_2",
            "one_point_7",
            "one_point_prop_2",
            "one_point_prop_7",
            "one_point_rank_2",
            "one_point_rank_7",
            "one_point_tour_3",
            "one_point_tour_7",
            "two_point_1",
            "two_point_2",
            "two_point_7",
            "two_point_prop_2",
            "two_point_prop_7",
            "two_point_rank_2",
            "two_point_rank_7",
            "two_point_tour_3",
            "two_point_tour_7",
            "uniform_1",
            "uniform_2",
            "uniform_7",
            "uniform_prop_2",
            "uniform_prop_7",
            "uniform_rank_2",
            "uniform_rank_7",
            "uniform_tour_3",
            "uniform_tour_7",
        ),
        init_population: Optional[NDArray[np.byte]] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[np.byte]], NDArray[Any]]] = None,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
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
    ):
        SHAGA.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            tour_size=tour_size,
            parents_num=parents_num,
            elitism=elitism,
            init_population=init_population,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
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

        self._thresholds: Dict[str, float] = {
            "selection": 0.2 / len(selections),
            "crossover": 0.2 / len(crossovers),
        }

        self._selection_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._crossover_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}

        for operator_name in selections:
            self._selection_set[operator_name] = self._selection_pool[operator_name]
        self._selection_set = dict(sorted(self._selection_set.items()))

        for operator_name in crossovers:
            self._crossover_set[operator_name] = self._crossover_pool[operator_name]
        self._crossover_set = dict(sorted(self._crossover_set.items()))

        self._z_selection = len(self._selection_set)
        self._z_crossover = len(self._crossover_set)

        self._selection_proba = dict(
            zip(list(self._selection_set.keys()), np.full(self._z_selection, 1 / self._z_selection))
        )
        if "empty" in self._crossover_set:
            self._crossover_proba = dict(
                zip(
                    list(self._crossover_set.keys()),
                    np.full(self._z_crossover, 0.9 / (self._z_crossover - 1)),
                )
            )
            self._crossover_proba["empty"] = 0.1
        else:
            self._crossover_proba = dict(
                zip(
                    list(self._crossover_set.keys()),
                    np.full(self._z_crossover, 1 / self._z_crossover),
                )
            )

        self._selection_operators: NDArray
        self._crossover_operators: NDArray
        self._success_i: Optional[NDArray[np.bool_]] = None

    def _get_individ_operators(self: SHAGA, index: int) -> Tuple[str, str]:
        return self._selection_operators[index], self._crossover_operators[index]

    def _get_init_population(self: SHAGA) -> None:
        self._selection_operators = self._choice_operators(proba_dict=self._selection_proba)
        self._crossover_operators = self._choice_operators(proba_dict=self._crossover_proba)
        SHAGA._get_init_population(self)

    def _update_data(self: SHAGA) -> None:
        EvolutionaryAlgorithm._update_data(self)
        self._update_stats(
            H_MR=self._H_MR,
            H_CR=self._H_CR,
            s_proba=self._selection_proba,
            c_proba=self._crossover_proba,
        )

    def _replace_population(
        self,
        mutant_g: NDArray,
        mutant_ph: NDArray,
        mutant_fit: NDArray,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        self._success_i = mutant_fit > self._fitness_i
        return super()._replace_population(mutant_g, mutant_ph, mutant_fit)

    def _culc_r_i(
        self, success: NDArray[np.bool_], operators: NDArray, operator_keys: List[str]
    ) -> Tuple[NDArray, NDArray[np.float64]]:
        keys, groups = numpy_group_by(group=success, by=operators)
        r_values = np.array(list(map(lambda x: (np.sum(x) ** 2 + 1) / (len(x) + 1), groups)))
        for key in operator_keys:
            if key not in keys:
                keys = np.append(keys, key)
                r_values = np.append(r_values, 0.0)

        return keys, r_values

    def _get_new_proba_pdp(
        self,
        proba_dict: Dict[str, float],
        operators: NDArray,
        threshold: float,
    ) -> Dict[str, float]:
        n = len(proba_dict)
        operators, r_values = self._culc_r_i(self._success_i, operators, list(proba_dict.keys()))
        scale = np.sum(r_values)
        proba_value = threshold + (r_values * ((1.0 - n * threshold) / scale))
        new_proba_dict = dict(zip(operators, proba_value))
        return dict(sorted(new_proba_dict.items()))

    def _adapt(self) -> None:
        if self._success_i is None:
            return None

        self._selection_proba = self._get_new_proba_pdp(
            self._selection_proba, self._selection_operators, self._thresholds["selection"]
        )
        self._crossover_proba = self._get_new_proba_pdp(
            self._crossover_proba, self._crossover_operators, self._thresholds["crossover"]
        )

        self._success_i = None
        self._selection_operators = self._choice_operators(proba_dict=self._selection_proba)
        self._crossover_operators = self._choice_operators(proba_dict=self._crossover_proba)
