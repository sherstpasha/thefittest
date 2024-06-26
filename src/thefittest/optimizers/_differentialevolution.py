from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

from numba import float64
from numba import njit

import numpy as np
from numpy.typing import NDArray

from ..base._ea import EvolutionaryAlgorithm
from ..utils import if_single_or_array_to_float_array
from ..utils.mutations import best_1
from ..utils.mutations import best_2
from ..utils.crossovers import binomial
from ..utils.mutations import current_to_best_1
from ..utils.mutations import rand_1
from ..utils.mutations import rand_2
from ..utils.mutations import rand_to_best1
from ..utils.random import uniform


@njit(float64[:](float64[:], float64[:], float64[:]))
def bounds_control(
    array: NDArray[np.float64], left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    to_return = array.copy()

    size = len(array)
    for i in range(size):
        if array[i] < left[i]:
            to_return[i] = left[i]
        elif array[i] > right[i]:
            to_return[i] = right[i]
    return to_return


class DifferentialEvolution(EvolutionaryAlgorithm):
    """Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient
    Adaptive Scheme for Global Optimization Over Continuous Spaces.
    Journal of Global Optimization. 23"""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        left_border: Union[float, int, np.number, NDArray[np.number]],
        right_border: Union[float, int, np.number, NDArray[np.number]],
        num_variables: int,
        mutation: str = "rand_1",
        F: float = 0.5,
        CR: float = 0.5,
        elitism: bool = True,
        init_population: Optional[NDArray[np.float64]] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[np.float64]], NDArray[Any]]] = None,
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
    ):
        EvolutionaryAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
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
        )

        self._num_variables = num_variables
        self._left: NDArray[np.float64] = if_single_or_array_to_float_array(
            left_border, self._num_variables
        )
        self._right: NDArray[np.float64] = if_single_or_array_to_float_array(
            right_border, self._num_variables
        )
        self._specified_mutation: str = mutation
        self._F: Union[float, NDArray[np.float64]] = F
        self._CR: Union[float, NDArray[np.float64]] = CR

        self._mutation_pool: Dict[str, Callable] = {
            "best_1": best_1,
            "rand_1": rand_1,
            "current_to_best_1": current_to_best_1,
            "rand_to_best1": rand_to_best1,
            "best_2": best_2,
            "rand_2": rand_2,
        }

    @staticmethod
    def float_population(
        pop_size: int,
        left_border: Union[float, int, np.number, NDArray[np.number]],
        right_border: Union[float, int, np.number, NDArray[np.number]],
        num_variables: int,
    ) -> NDArray[np.float64]:
        left = if_single_or_array_to_float_array(left_border, num_variables)
        right = if_single_or_array_to_float_array(right_border, num_variables)
        points_along_axis = [
            uniform(left_i, right_i, pop_size) for left_i, right_i in zip(left, right)
        ]
        return np.array(points_along_axis, dtype=np.float64).T

    def _first_generation(self: DifferentialEvolution) -> None:
        if self._init_population is None:
            self._population_g_i = self.float_population(
                pop_size=self._pop_size,
                left_border=self._left,
                right_border=self._right,
                num_variables=self._num_variables,
            )
        else:
            self._population_g_i = self._init_population.copy()

    def _get_init_population(self: DifferentialEvolution) -> None:
        self._first_generation()
        self._population_ph_i = self._get_phenotype(self._population_g_i)
        self._fitness_i = self._get_fitness(self._population_ph_i)

    def _get_new_individ_g(
        self: DifferentialEvolution,
        individ_g: NDArray[np.float64],
        F: float,
        CR: float,
    ) -> NDArray[np.float64]:
        mutation_func = self._mutation_pool[self._specified_mutation]

        mutant_g = mutation_func(
            individ_g, self._thefittest._genotype, self._population_g_i, np.float64(F)
        )

        mutant_cr_g = binomial(individ_g, mutant_g, np.float64(CR))
        return bounds_control(mutant_cr_g, self._left, self._right)

    def _get_new_population(self: DifferentialEvolution) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
            F=self._F,
            CR=self._CR,
        )

        mutant_cr_b_g = np.array(
            [get_new_individ_g(individ_g=self._population_g_i[i]) for i in range(self._pop_size)],
            dtype=np.float64,
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= self._fitness_i

        self._population_g_i[mask] = mutant_cr_b_g[mask]
        self._population_ph_i[mask] = mutant_cr_ph[mask]
        self._fitness_i[mask] = mutant_cr_fit[mask]

    def _from_population_g_to_fitness(self: EvolutionaryAlgorithm) -> None:
        self._update_data()

        if self._elitism:
            (
                self._population_g_i[-1],
                self._population_ph_i[-1],
                self._fitness_i[-1],
            ) = self._thefittest.get().values()

        self._adapt()
