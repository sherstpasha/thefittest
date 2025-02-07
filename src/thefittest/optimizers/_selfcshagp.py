from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..base import Tree

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


class SelfCSHAGP(CSHAGP):

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
            "tournament_k",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = (
            "empty",
            "gp_standart",
            "gp_one_point",
            "gp_uniform_1",
            "gp_uniform_2",
            "gp_uniform_7",
            "gp_uniform_prop_2",
            "gp_uniform_prop_7",
            "gp_uniform_rank_2",
            "gp_uniform_rank_7",
            "gp_uniform_tour_3",
            "gp_uniform_tour_7",
        ),
        mutations: Tuple[str, ...] = ("point", "grow", "swap", "shrink"),
        K: float = 2,
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

        CSHAGP.__init__(
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
            "mutation": mutation_threshold_proba,
        }

        self._selection_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._crossover_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._mutation_set: Dict[str, Tuple[Callable, Union[float, int], bool]] = {}

        self._selection_proba: Dict[str, float]
        self._crossover_proba: Dict[str, float]
        self._mutation_proba: Dict[str, float]

        for operator_name in selections:
            self._selection_set[operator_name] = self._selection_pool[operator_name]
        self._selection_set = dict(sorted(self._selection_set.items()))

        for operator_name in crossovers:
            self._crossover_set[operator_name] = self._crossover_pool[operator_name]
        self._crossover_set = dict(sorted(self._crossover_set.items()))

        for operator_name in mutations:
            self._mutation_set[operator_name] = self._mutation_pool[operator_name]
        self._mutation_set = dict(sorted(self._mutation_set.items()))

        self._z_selection = len(self._selection_set)
        self._z_crossover = len(self._crossover_set)
        self._z_mutation = len(self._mutation_set)

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
        self._mutation_proba = dict(
            zip(list(self._mutation_set.keys()), np.full(self._z_mutation, 1 / self._z_mutation))
        )

        self._selection_operators: NDArray = self._choice_operators(
            proba_dict=self._selection_proba
        )
        self._crossover_operators: NDArray = self._choice_operators(
            proba_dict=self._crossover_proba
        )
        self._mutation_operators: NDArray = self._choice_operators(proba_dict=self._mutation_proba)

    def _choice_operators(self: SelfCSHAGP, proba_dict: Dict["str", float]) -> NDArray:
        operators = list(proba_dict.keys())
        proba = list(proba_dict.values())
        chosen_operator = np.random.choice(operators, self._pop_size, p=proba)
        return chosen_operator

    def _get_new_proba(
        self: SelfCSHAGP,
        proba_dict: Dict["str", float],
        operator: str,
        threshold: float,
    ) -> Dict["str", float]:
        proba_dict[operator] += self._K / self._iters
        proba_value = np.array(list(proba_dict.values()))
        proba_value -= self._K / (len(proba_dict) * self._iters)
        proba_value = proba_value.clip(threshold, 1)
        proba_value = proba_value / proba_value.sum()
        new_proba_dict = dict(zip(proba_dict.keys(), proba_value))
        return new_proba_dict

    def _find_fittest_operator(
        self: SelfCSHAGP, operators: NDArray, fitness: NDArray[np.float32]
    ) -> str:
        keys, groups = numpy_group_by(group=fitness, by=operators)
        mean_fit = np.array(list(map(np.mean, groups)))
        fittest_operator = keys[np.argmax(mean_fit)]
        return fittest_operator

    def _update_data(self: SelfCSHAGP) -> None:
        super()._update_data()
        self._update_stats(
            s_proba=self._selection_proba,
            c_proba=self._crossover_proba,
            m_proba=self._mutation_proba,
        )

    def _adapt(self: SelfCSHAGP) -> None:
        s_fittest_oper = self._find_fittest_operator(self._selection_operators, self._fitness_i)
        self._selection_proba = self._get_new_proba(
            self._selection_proba, s_fittest_oper, self._thresholds["selection"]
        )

        c_fittest_oper = self._find_fittest_operator(self._crossover_operators, self._fitness_i)
        self._crossover_proba = self._get_new_proba(
            self._crossover_proba, c_fittest_oper, self._thresholds["crossover"]
        )

        m_fittest_oper = self._find_fittest_operator(self._mutation_operators, self._fitness_i)
        self._mutation_proba = self._get_new_proba(
            self._mutation_proba, m_fittest_oper, self._thresholds["mutation"]
        )

        self._selection_operators = self._choice_operators(proba_dict=self._selection_proba)
        self._crossover_operators = self._choice_operators(proba_dict=self._crossover_proba)
        self._mutation_operators = self._choice_operators(proba_dict=self._mutation_proba)

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
