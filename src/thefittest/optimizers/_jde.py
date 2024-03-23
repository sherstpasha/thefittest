from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._differentialevolution import DifferentialEvolution
from ..utils.random import uniform


class jDE(DifferentialEvolution):
    """Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan & Zumer, Viljem. (2007).
    Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical
    Benchmark Problems. Evolutionary Computation, IEEE Transactions on. 10.
      646 - 657. 10.1109/TEVC.2006.872133."""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        mutation: str = "rand_1",
        F_min: float = 0.1,
        F_max: float = 0.9,
        t_F: float = 0.1,
        t_CR: float = 0.1,
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
        DifferentialEvolution.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            left=left,
            right=right,
            mutation=mutation,
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

        self._F_min: float = F_min
        self._F_max: float = F_max
        self._t_F: float = t_F
        self._t_CR: float = t_CR

        self._F: NDArray[np.float64] = np.full(self._pop_size, 0.5, dtype=np.float64)
        self._CR: NDArray[np.float64] = np.full(self._pop_size, 0.9, dtype=np.float64)

    def _get_mutate_F(self: jDE) -> NDArray[np.float64]:
        mutate_F = self._F.copy()
        mask = uniform(0.0, 1.0, size=self._pop_size) < self._t_F

        random_values = uniform(0.0, 1.0, size=np.sum(mask))
        mutate_F[mask] = self._F_min + random_values * self._F_max
        return mutate_F

    def _get_mutate_CR(self: jDE) -> NDArray[np.float64]:
        mutate_CR = self._CR.copy()
        mask = uniform(0.0, 1.0, size=self._pop_size) < self._t_CR

        random_values = uniform(0.0, 1.0, size=np.sum(mask))
        mutate_CR[mask] = random_values
        return mutate_CR

    def _get_new_population(self: jDE) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )

        mutate_F = self._get_mutate_F()
        mutate_CR = self._get_mutate_CR()

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(individ_g=self._population_g_i[i], F=mutate_F[i], CR=mutate_CR[i])
                for i in range(self._pop_size)
            ],
            dtype=np.float64,
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= self._fitness_i

        self._population_g_i[mask] = mutant_cr_b_g[mask]
        self._population_ph_i[mask] = mutant_cr_ph[mask]
        self._fitness_i[mask] = mutant_cr_fit[mask]
        self._F[mask] = mutate_F[mask]
        self._CR[mask] = mutate_CR[mask]

    def _update_data(self: jDE) -> None:
        super()._update_data()
        self._update_stats(F=self._F, CR=self._CR)
