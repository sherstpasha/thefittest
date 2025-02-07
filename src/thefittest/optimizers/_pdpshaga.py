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
from ._selfcshaga import SelfCSHAGA

import random
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._selfcga import SelfCGA
from ..base import Tree
from ..tools import donothing
from ..tools.transformations import numpy_group_by


class PDPSHAGA(SelfCSHAGA):

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float32]],
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
            "empty",
            "uniform_1",
            "uniform_2",
            "one_point",
            "two_point",
            "one_point_7",
            "two_point_7",
            "uniform_7",
            "uniform_prop_2",
            "uniform_prop_7",
            "uniform_rank_2",
            "uniform_rank_7",
            "uniform_tour_3",
            "uniform_tour_7",
        ),
        init_population: Optional[NDArray[np.byte]] = None,
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
        SelfCSHAGA.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            tour_size=tour_size,
            parents_num=parents_num,
            elitism=elitism,
            selections=selections,
            crossovers=crossovers,
            selection_threshold_proba=0.2 / len(selections),
            crossover_threshold_proba=0.2 / len(crossovers),
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

        self._previous_fitness_i: List = []
        self._success_i: NDArray[np.bool_]

    def _choice_parent(self: PDPSHAGA, fitness_i_selected: NDArray[np.float32]) -> np.float32:
        choosen = random.choice(list(fitness_i_selected))
        return choosen

    def _get_new_proba_pdp(
        self: PDPSHAGA,
        proba_dict: Dict["str", float],
        operators: NDArray,
        threshold: float,
    ) -> Dict["str", float]:
        n = len(proba_dict)
        operators, r_values = self._culc_r_i(self._success_i, operators, list(proba_dict.keys()))
        scale = np.sum(r_values)
        proba_value = threshold + (r_values * ((1.0 - n * threshold) / scale))
        new_proba_dict = dict(zip(operators, proba_value))
        new_proba_dict = dict(sorted(new_proba_dict.items()))
        return new_proba_dict

    def _culc_r_i(
        self: PDPSHAGA, success: NDArray[np.bool_], operator: NDArray, operator_keys: List[str]
    ) -> Tuple:
        keys, groups = numpy_group_by(group=success, by=operator)
        r_values = np.array(list(map(lambda x: (np.sum(x) ** 2 + 1) / (len(x) + 1), groups)))
        for key in operator_keys:
            if key not in keys:
                keys = np.append(keys, key)
                r_values = np.append(r_values, 0.0)

        return keys, r_values

    def _adapt(self: PDPSHAGA) -> None:
        if len(self._previous_fitness_i):
            self._success_i = np.array(self._previous_fitness_i, dtype=np.float32) < self._fitness_i

            self._selection_proba = self._get_new_proba_pdp(
                self._selection_proba, self._selection_operators, self._thresholds["selection"]
            )

            self._crossover_proba = self._get_new_proba_pdp(
                self._crossover_proba, self._crossover_operators, self._thresholds["crossover"]
            )

            self._previous_fitness_i = []

    def _get_new_individ_g(
        self: SHAGACONF,
        specified_selection: str,
        specified_crossover: str,
        individ_g: NDArray[np.float32],
        fitness_g_i: np.float32,
        MR: float,
        CR: float,
    ) -> NDArray[np.float32]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]

        selected_id = selection_func(
            self._fitness_scale_i,
            self._fitness_rank_i,
            np.int64(tour_size),
            np.int64(quantity),
        )
        previous_fitness = self._choice_parent(np.append(self._fitness_i[selected_id], fitness_g_i))
        self._previous_fitness_i.append(previous_fitness)

        second_parents = self._population_g_i[selected_id].copy()

        offspring = crossover_func(
            individ_g,
            second_parents,
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            CR,
        )
        mutant = flip_mutation(offspring, MR)
        return mutant

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
                    fitness_g_i=self._fitness_i[i],
                    MR=self._MR[i],
                    CR=self._CR[i],
                )
                for i in range(self._pop_size)
            ],
            dtype=np.float32,
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
