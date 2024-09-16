from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
from numpy.typing import NDArray

from ..base._ea import EvolutionaryAlgorithm
from ..tools import donothing
from ..tools.operators import binomialGA
from ..tools.operators import flip_mutation
from ..tools.operators import tournament_selection
from ..tools.random import binary_string_population
from ..tools.random import cauchy_distribution
from ..tools.transformations import lehmer_mean

from ..tools.operators import empty_crossover_shaga
from ..tools.operators import flip_mutation
from ..tools.operators import growing_mutation
from ..tools.operators import one_point_crossover
from ..tools.operators import one_point_crossoverGP
from ..tools.operators import point_mutation
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.operators import shrink_mutation
from ..tools.operators import standart_crossover
from ..tools.operators import swap_mutation
from ..tools.operators import tournament_selection
from ..tools.operators import two_point_crossover
from ..tools.operators import uniform_crossover_shaga
from ..tools.operators import uniform_crossoverGP
from ..tools.operators import uniform_prop_crossover_shaga
from ..tools.operators import uniform_prop_crossover_GP
from ..tools.operators import uniform_rank_crossover_shaga
from ..tools.operators import uniform_rank_crossover_GP
from ..tools.operators import uniform_tour_crossover_shaga
from ..tools.operators import uniform_tour_crossover_GP
from ..tools.transformations import rank_data
from ..tools.transformations import scale_data

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._shaga_conf import SHAGACONF
from ..tools import donothing
from ..tools.transformations import numpy_group_by


class SelfCSHAGA(SHAGACONF):

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
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = (
            # "empty",
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
        K: float = 2,
        selection_threshold_proba: float = 0.05,
        crossover_threshold_proba: float = 0.05,
        genotype_to_phenotype: Callable[[NDArray[np.byte]], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
    ):
        SHAGACONF.__init__(
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
        )

        self._K: float = K
        self._thresholds: Dict[str, float] = {
            "selection": selection_threshold_proba,
            "crossover": crossover_threshold_proba,
        }

        self._selection_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._crossover_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}

        self._selection_proba: Dict[str, float]
        self._crossover_proba: Dict[str, float]

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
        if "empty" in self._crossover_set.keys():
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

        self._selection_operators: NDArray = self._choice_operators(
            proba_dict=self._selection_proba
        )
        self._crossover_operators: NDArray = self._choice_operators(
            proba_dict=self._crossover_proba
        )

    def _choice_operators(self: SelfCSHAGA, proba_dict: Dict["str", float]) -> NDArray:
        operators = list(proba_dict.keys())
        proba = list(proba_dict.values())
        chosen_operator = np.random.choice(operators, self._pop_size, p=proba)
        return chosen_operator

    def _get_new_proba(
        self: SelfCSHAGA,
        proba_dict: Dict["str", float],
        operator: str,
        threshold: float,
    ) -> Dict["str", float]:
        K = np.random.uniform(0, 10)
        proba_dict[operator] += K / self._iters
        proba_value = np.array(list(proba_dict.values()))
        proba_value -= K / (len(proba_dict) * self._iters)
        proba_value = proba_value.clip(threshold, 1)
        proba_value = proba_value / proba_value.sum()
        new_proba_dict = dict(zip(proba_dict.keys(), proba_value))
        return new_proba_dict

    def _find_fittest_operator(
        self: SelfCSHAGA, operators: NDArray, fitness: NDArray[np.float64]
    ) -> str:
        keys, groups = numpy_group_by(group=fitness, by=operators)
        mean_fit = np.array(list(map(np.mean, groups)))
        fittest_operator = keys[np.argmax(mean_fit)]
        return fittest_operator

    def _update_data(self: SelfCSHAGA) -> None:
        super()._update_data()
        self._update_stats(
            s_proba=self._selection_proba,
            c_proba=self._crossover_proba,
        )

    def _adapt(self: SelfCSHAGA) -> None:
        s_fittest_oper = self._find_fittest_operator(self._selection_operators, self._fitness_i)
        self._selection_proba = self._get_new_proba(
            self._selection_proba, s_fittest_oper, self._thresholds["selection"]
        )

        c_fittest_oper = self._find_fittest_operator(self._crossover_operators, self._fitness_i)
        self._crossover_proba = self._get_new_proba(
            self._crossover_proba, c_fittest_oper, self._thresholds["crossover"]
        )

        self._selection_operators = self._choice_operators(proba_dict=self._selection_proba)
        self._crossover_operators = self._choice_operators(proba_dict=self._crossover_proba)

    def _get_new_population(self: SHAGACONF) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )
        self._MR, self._CR = self._generate_MR_CR()

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(
                    specified_selection=self._selection_operators[i],
                    specified_crossover=self._crossover_operators[i],
                    individ_g=self._population_g_i[i],
                    MR=self._MR[i],
                    CR=self._CR[i],
                )
                for i in range(self._pop_size)
            ],
            dtype=np.float64,
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= self._fitness_i
        succeses = mutant_cr_fit > self._fitness_i

        succeses_MR = self._MR[succeses]
        succeses_CR = self._CR[succeses]

        will_be_replaced_fit = self._fitness_i[succeses].copy()

        self._population_g_i[mask] = mutant_cr_b_g[mask]
        self._population_ph_i[mask] = mutant_cr_ph[mask]
        self._fitness_i[mask] = mutant_cr_fit[mask]
        self._fitness_scale_i = scale_data(self._fitness_i)
        self._fitness_rank_i = rank_data(self._fitness_i)

        d_fitness = np.abs(will_be_replaced_fit - self._fitness_i[succeses])

        if self._k + 1 == self._H_size:
            next_k = 0
        else:
            next_k = self._k + 1

        self._H_MR[next_k] = self._update_u(self._H_MR[self._k], succeses_MR, d_fitness)
        self._H_CR[next_k] = self._update_u(self._H_CR[self._k], succeses_CR, d_fitness)

        if self._k == self._H_size - 1:
            self._k = 0
        else:
            self._k += 1
