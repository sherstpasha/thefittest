from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import List
import numpy as np
from numpy.typing import NDArray

from ..base import Tree
import random
from ._geneticprogramming import GeneticProgramming
from ._selfcga import SelfCGA
from ..base import UniversalSet
from ..tools import donothing
from ..base._ea import EvolutionaryAlgorithm
from . import SHAGA
from ..tools.random import half_and_half
from ..tools.operators import tournament_selection
from ..tools.operators import uniform_crossoverSHAGP
from ..tools.operators import standart_crossover_shagp
from ..tools.operators import one_point_crossover_SHAGP
from ..tools.operators import uniform_prop_crossoverSHAGP
from ..tools.operators import uniform_rank_crossoverSHAGP
from ..tools.operators import uniform_tour_crossoverSHAGP
from ..tools.operators import empty_crossover_SHAGP
from ..tools.operators import point_mutation, growing_mutation, swap_mutation, shrink_mutation
from ..tools.random import cauchy_distribution
from typing import Union
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.transformations import rank_data
from ..tools.transformations import scale_data

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

from ._cshagp import CSHAGP
from ..tools import donothing
from ..tools.transformations import numpy_group_by
from ..optimizers._selfcshagp import SelfCSHAGP


class PDPSHAGP(SelfCSHAGP):

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float32]],
        uniset: UniversalSet,
        iters: int,
        pop_size: int,
        elitism: bool = True,
        max_level: int = 16,
        init_level: int = 4,
        init_population: Optional[NDArray[np.byte]] = None,
        genotype_to_phenotype: Callable[[NDArray[np.byte]], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        parents_num: int = 2,
        tour_size: int = 2,
        selections: Tuple[str, ...] = (
            "proportional",
            "rank",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = (
            #"empty",
            "gp_standart",
            "gp_one_point",
            "gp_uniform_2",
            "gp_uniform_7",
            "gp_uniform_rank_2",
            "gp_uniform_rank_7",
            "gp_uniform_tour_3",
            "gp_uniform_tour_7",
        ),
        mutations: Tuple[str, ...] = ("point", "grow"),
        selection_threshold_proba: float = 0.05,
        crossover_threshold_proba: float = 0.05,
        mutation_threshold_proba: float = 0.05,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
    ):

        SelfCSHAGP.__init__(
            self,
            fitness_function=fitness_function,
            uniset=uniset,
            iters=iters,
            pop_size=pop_size,
            elitism=elitism,
            max_level=max_level,
            init_level=init_level,
            init_population=init_population,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            parents_num=parents_num,
            tour_size=tour_size,
            selections=selections,
            crossovers=crossovers,
            mutations=mutations,
            selection_threshold_proba=selection_threshold_proba,
            crossover_threshold_proba=crossover_threshold_proba,
            mutation_threshold_proba=mutation_threshold_proba,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            n_jobs=n_jobs,
            fitness_function_args=fitness_function_args,
            genotype_to_phenotype_args=genotype_to_phenotype_args,
        )

        self._previous_fitness_i: List = []
        self._success_i: NDArray[np.bool_]

    def _choice_parent(self: PDPSHAGP, fitness_i_selected: NDArray[np.float32]) -> np.float32:
        # print(fitness_i_selected)
        choosen = random.choice(list(fitness_i_selected))
        return choosen

    def _get_new_proba_pdp(
        self: PDPSHAGP,
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
        self: PDPSHAGP, success: NDArray[np.bool_], operator: NDArray, operator_keys: List[str]
    ) -> Tuple:
        keys, groups = numpy_group_by(group=success, by=operator)
        r_values = np.array(list(map(lambda x: (np.sum(x) ** 2 + 1) / (len(x) + 1), groups)))
        for key in operator_keys:
            if key not in keys:
                keys = np.append(keys, key)
                r_values = np.append(r_values, 0.0)

        return keys, r_values

    def _adapt(self: PDPSHAGP) -> None:
        if len(self._previous_fitness_i):
            self._success_i = np.array(self._previous_fitness_i, dtype=np.float32) < self._fitness_i

            self._selection_proba = self._get_new_proba_pdp(
                self._selection_proba, self._selection_operators, self._thresholds["selection"]
            )

            self._crossover_proba = self._get_new_proba_pdp(
                self._crossover_proba, self._crossover_operators, self._thresholds["crossover"]
            )

            self._mutation_proba = self._get_new_proba_pdp(
                self._mutation_proba, self._mutation_operators, self._thresholds["mutation"]
            )

            self._previous_fitness_i = []

    def _get_new_individ_g(
        self: SHAGA,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
        individ_g: NDArray[np.float32],
        fitness_g_i: np.float32,
        MR: float,
        CR: float,
    ) -> NDArray[np.float32]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func = self._mutation_pool[specified_mutation]

        selected_id = selection_func(
            self._fitness_scale_i,
            self._fitness_rank_i,
            np.int64(tour_size),
            np.int64(quantity),
        )

        previous_fitness = self._choice_parent(np.append(self._fitness_i[selected_id], fitness_g_i))
        self._previous_fitness_i.append(previous_fitness)

        second_parent = self._population_g_i[selected_id].copy()

        offspring = crossover_func(
            individ_g,
            second_parent,
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
            CR,
        )
        mutant = mutation_func(offspring, self._uniset, MR, self._max_level)
        return mutant

    def _get_new_population(self: SelfCSHAGP) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )
        self._MR, self._CR = self._generate_MR_CR()

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(
                    specified_selection=self._selection_operators[i],
                    specified_crossover=self._crossover_operators[i],
                    specified_mutation=self._mutation_operators[i],
                    individ_g=self._population_g_i[i],
                    fitness_g_i=self._fitness_i[i],
                    MR=self._MR[i],
                    CR=self._CR[i],
                )
                for i in range(self._pop_size)
            ],
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
        d_fitness[d_fitness == np.inf] = 1e4

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
