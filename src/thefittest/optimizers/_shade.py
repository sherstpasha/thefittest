from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ._differentialevolution import DifferentialEvolution
from ..tools import donothing
from ..tools import find_pbest_id
from ..tools.operators import binomial
from ..tools.operators import current_to_pbest_1_archive_p_min
from ..tools.random import randc01
from ..tools.random import randn01
from ..tools.transformations import bounds_control_mean
from ..tools.transformations import lehmer_mean


class SHADE(DifferentialEvolution):
    """Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based parameter adaptation
    for Differential Evolution. 2013 IEEE Congress on Evolutionary Computation,
    CEC 2013. 71-78. 10.1109/CEC.2013.6557555."""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        elitism: bool = True,
        init_population: Optional[NDArray[np.float64]] = None,
        genotype_to_phenotype: Callable[[NDArray[np.float64]], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        terminate_function: Optional[Callable[[NDArray[Any]], NDArray[np.bool]]] = None,
        terminate_function_args: Optional[Dict] = None,
    ):
        DifferentialEvolution.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            left=left,
            right=right,
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
            terminate_function=terminate_function,
            terminate_function_args=terminate_function_args,
        )

        self._F: NDArray[np.float64]
        self._CR: NDArray[np.float64]
        self._population_g_archive_i: NDArray[np.float64] = np.zeros(shape=(0, len(self._left)))
        self._population_archive: NDArray[np.float64]
        self._pbest_id: NDArray[np.int64]
        self._H_size: int = pop_size
        self._H_F = np.full(self._H_size, 0.5, dtype=np.float64)
        self._H_CR = np.full(self._H_size, 0.5, dtype=np.float64)
        self._k: int = 0
        self._p: float = 0.05

    def _generate_F_CR(self: SHADE) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        F_i = np.zeros(self._pop_size, dtype=np.float64)
        CR_i = np.zeros(self._pop_size, dtype=np.float64)
        for i in range(self._pop_size):
            r_i = np.random.randint(0, self._H_size)
            u_F = self._H_F[r_i]
            u_CR = self._H_CR[r_i]
            F_i[i] = randc01(np.float64(u_F))
            CR_i[i] = randn01(np.float64(u_CR))
        return (F_i, CR_i)

    def _append_archive(
        self, archive: NDArray[np.float64], worse_g: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        archive = np.append(archive, worse_g, axis=0)
        if len(archive) > self._pop_size:
            np.random.shuffle(archive)
            archive = archive[: self._pop_size]
        return archive

    def _update_u_F(self, u_F: float, S_F: NDArray[np.float64]) -> float:
        if len(S_F):
            return lehmer_mean(S_F)
        return u_F

    def _update_u_CR(
        self, u_CR: float, S_CR: NDArray[np.float64], df: NDArray[np.float64]
    ) -> float:
        if len(S_CR):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df / sum_
                return np.sum(weight_i * S_CR)
        return u_CR

    def _get_new_individ_g(
        self: SHADE,
        individ_g: NDArray[np.float64],
        F: float,
        CR: float,
    ) -> NDArray[np.float64]:
        mutant_g = current_to_pbest_1_archive_p_min(
            individ_g, self._population_g_i, self._pbest_id, F, self._population_archive
        )

        mutant_cr_g = binomial(individ_g, mutant_g, np.float64(CR))
        return bounds_control_mean(mutant_cr_g, self._left, self._right)

    def _get_new_population(self: SHADE) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )

        self._F, self._CR = self._generate_F_CR()
        self._pbest_id = find_pbest_id(self._fitness_i, np.float64(self._p))
        self._population_archive = np.vstack([self._population_g_i, self._population_g_archive_i])

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(individ_g=self._population_g_i[i], F=self._F[i], CR=self._CR[i])
                for i in range(self._pop_size)
            ],
            dtype=np.float64,
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= self._fitness_i
        succeses = mutant_cr_fit > self._fitness_i

        succeses_F = self._F[succeses]
        succeses_CR = self._CR[succeses]

        will_be_replaced_pop = self._population_g_i[succeses].copy()
        will_be_replaced_fit = self._fitness_i[succeses].copy()

        self._population_g_archive_i = self._append_archive(
            self._population_g_archive_i, will_be_replaced_pop
        )

        self._population_g_i[mask] = mutant_cr_b_g[mask]
        self._population_ph_i[mask] = mutant_cr_ph[mask]
        self._fitness_i[mask] = mutant_cr_fit[mask]

        d_fitness = np.abs(will_be_replaced_fit - self._fitness_i[succeses])

        if self._k + 1 == self._H_size:
            next_k = 0
        else:
            next_k = self._k + 1

        self._H_F[next_k] = self._update_u_F(self._H_F[self._k], succeses_F)
        self._H_CR[next_k] = self._update_u_CR(self._H_CR[self._k], succeses_CR, d_fitness)

        if self._k == self._H_size - 1:
            self._k = 0
        else:
            self._k += 1

    def _update_data(self: SHADE) -> None:
        super()._update_data()
        self._update_stats(H_F=self._H_F, H_CR=self._H_CR)
