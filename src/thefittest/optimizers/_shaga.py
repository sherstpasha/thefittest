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
from ..optimizers._shade import lehmer_mean
from ..utils.crossovers import binomialGA
from ..utils.mutations import flip_mutation
from ..utils.selections import tournament_selection
from ..utils.random import cauchy_distribution
from ..utils.random import randint


class SHAGA(EvolutionaryAlgorithm):
    """Stanovov, Vladimir & Akhmedova, Shakhnaz & Semenkin, Eugene. (2019).
    Genetic Algorithm with Success History based Parameter Adaptation. 180-187.
    10.5220/0008071201800187."""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        str_len: int,
        elitism: bool = True,
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

        self._str_len = str_len
        self._MR: NDArray[np.float64]
        self._CR: NDArray[np.float64]
        self._H_size: int = pop_size
        self._H_MR = np.full(self._H_size, 1 / (self._str_len), dtype=np.float64)
        self._H_CR = np.full(self._H_size, 0.5, dtype=np.float64)
        self._k: int = 0

    @staticmethod
    def binary_string_population(pop_size: int, str_len: int) -> NDArray[np.byte]:

        population = np.array(
            [randint(low=0, high=2, size=str_len) for _ in range(pop_size)],
            dtype=np.byte,
        )

        return population

    def _first_generation(self: SHAGA) -> None:
        if self._init_population is None:
            self._population_g_i = self.binary_string_population(self._pop_size, self._str_len)
        else:
            self._population_g_i = self._init_population.copy()

    def _get_init_population(self: SHAGA) -> None:
        self._first_generation()
        self._population_ph_i = self._get_phenotype(self._population_g_i)
        self._fitness_i = self._get_fitness(self._population_ph_i)

    def _randc(self: SHAGA, u: float, scale: float) -> NDArray[np.float64]:
        value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        while value <= 0 or value > 5 / self._str_len:
            value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        return value

    def _randn(self: SHAGA, u: float, scale: float) -> float:
        value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        return value

    def _generate_MR_CR(self: SHAGA) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        MR_i = np.zeros(self._pop_size)
        CR_i = np.zeros(self._pop_size)
        for i in range(self._pop_size):
            r_i = randint(0, self._H_size, 1)[0]
            u_MR = self._H_MR[r_i]
            u_CR = self._H_CR[r_i]
            MR_i[i] = self._randc(u_MR, 0.1 / self._str_len)
            CR_i[i] = self._randn(u_CR, 0.1)
        return MR_i, CR_i

    def _update_u(self: SHAGA, u: float, S: NDArray[np.float64], df: NDArray[np.float64]) -> float:
        if len(S):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df / sum_
                return lehmer_mean(x=S, weight=weight_i)
        return u

    def _get_new_individ_g(
        self: SHAGA,
        individ_g: NDArray[np.float64],
        MR: float,
        CR: float,
    ) -> NDArray[np.float64]:
        second_parent_id = tournament_selection(self._fitness_i, self._fitness_i, 2, 1)[0]
        second_parent = self._population_g_i[second_parent_id].copy()
        offspring = binomialGA(individ_g, second_parent, CR)
        mutant = flip_mutation(offspring, MR)
        return mutant

    def _get_new_population(self: SHAGA) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )
        self._MR, self._CR = self._generate_MR_CR()

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(individ_g=self._population_g_i[i], MR=self._MR[i], CR=self._CR[i])
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

    def _update_data(self: SHAGA) -> None:
        super()._update_data()
        self._update_stats(H_MR=self._H_MR, H_CR=self._H_CR)

    def _from_population_g_to_fitness(self: EvolutionaryAlgorithm) -> None:
        self._update_data()

        if self._elitism:
            (
                self._population_g_i[-1],
                self._population_ph_i[-1],
                self._fitness_i[-1],
            ) = self._thefittest.get().values()

        self._adapt()
