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

from scipy.stats import rankdata

from ..base._ea import EvolutionaryAlgorithm

from ..utils.selections import proportional_selection
from ..utils.selections import rank_selection
from ..utils.selections import tournament_selection

from ..utils.crossovers import empty_crossover_selfcshaga
from ..utils.crossovers import one_point_crossover_selfcshaga
from ..utils.crossovers import one_point_prop_crossover_selfcshaga
from ..utils.crossovers import one_point_rank_crossover_selfcshaga
from ..utils.crossovers import one_point_tour_crossover_selfcshaga
from ..utils.crossovers import uniform_crossover_selfcshaga
from ..utils.crossovers import uniform_prop_crossover_selfcshaga
from ..utils.crossovers import uniform_rank_crossover_selfcshaga
from ..utils.crossovers import uniform_tour_crossover_selfcshaga
from ..utils.crossovers import binomial_selfshaga
from ..utils.mutations import flip_mutation
from ..optimizers._shade import lehmer_mean

from ..utils.random import cauchy_distribution
from ..utils.random import randint
from ..utils.transformations import minmax_scale


class SHAGA(EvolutionaryAlgorithm):
    """Stanovov, Vladimir & Akhmedova, Shakhnaz & Semenkin, Eugene. (2019).
    Genetic Algorithm with Success History based Parameter Adaptation. 180-187.
    10.5220/0008071201800187."""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float32]],
        iters: int,
        pop_size: int,
        str_len: int,
        tour_size: int = 2,
        parents_num: int = 2,
        elitism: bool = True,
        selection: str = "tournament_2",
        crossover: str = "uniform_1",
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
            fitness_update_eps=fitness_update_eps,
            use_fitness_cache=use_fitness_cache,
            fitness_cache_size=fitness_cache_size,
        )

        self._str_len = str_len
        self._MR: NDArray[np.float64]
        self._CR: NDArray[np.float64]
        self._H_size: int = pop_size
        self._H_MR = np.full(self._H_size, 1 / (self._str_len), dtype=np.float64)
        self._H_CR = np.full(self._H_size, 0.5, dtype=np.float64)
        self._k: int = 0
        self._parents_num: int = parents_num
        self._tour_size: int = tour_size

        self._specified_selection: str = selection
        self._specified_crossover: str = crossover

        self._selection_pool: Dict[str, Tuple[Callable, Union[float, int]]] = {
            "proportional": (proportional_selection, 0),
            "rank": (rank_selection, 0),
            "tournament_k": (tournament_selection, self._tour_size),
            "tournament_2": (tournament_selection, 2),
            "tournament_3": (tournament_selection, 3),
            "tournament_5": (tournament_selection, 5),
            "tournament_7": (tournament_selection, 7),
        }

        self._crossover_pool: Dict[str, Tuple[Callable, Union[float, int]]] = {
            "empty": (empty_crossover_selfcshaga, 1),
            "one_point_1": (one_point_crossover_selfcshaga, 1),
            "one_point_2": (one_point_crossover_selfcshaga, 2),
            "one_point_7": (one_point_crossover_selfcshaga, 7),
            "one_point_k": (one_point_crossover_selfcshaga, self._parents_num),
            "one_point_prop_2": (one_point_prop_crossover_selfcshaga, 1),
            "one_point_prop_7": (one_point_prop_crossover_selfcshaga, 7),
            "one_point_prop_k": (one_point_prop_crossover_selfcshaga, self._parents_num),
            "one_point_rank_2": (one_point_rank_crossover_selfcshaga, 1),
            "one_point_rank_7": (one_point_rank_crossover_selfcshaga, 7),
            "one_point_rank_k": (one_point_rank_crossover_selfcshaga, self._parents_num),
            "one_point_tour_2": (one_point_tour_crossover_selfcshaga, 1),
            "one_point_tour_7": (one_point_tour_crossover_selfcshaga, 7),
            "one_point_tour_k": (one_point_tour_crossover_selfcshaga, self._parents_num),
            "uniform_1": (binomial_selfshaga, 1),
            "uniform_2": (uniform_crossover_selfcshaga, 2),
            "uniform_7": (uniform_crossover_selfcshaga, 7),
            "uniform_k": (uniform_crossover_selfcshaga, self._parents_num),
            "uniform_prop_2": (uniform_prop_crossover_selfcshaga, 2),
            "uniform_prop_7": (uniform_prop_crossover_selfcshaga, 7),
            "uniform_prop_k": (uniform_prop_crossover_selfcshaga, self._parents_num),
            "uniform_rank_2": (uniform_rank_crossover_selfcshaga, 2),
            "uniform_rank_7": (uniform_rank_crossover_selfcshaga, 7),
            "uniform_rank_k": (uniform_rank_crossover_selfcshaga, self._parents_num),
            "uniform_tour_3": (uniform_tour_crossover_selfcshaga, 3),
            "uniform_tour_7": (uniform_tour_crossover_selfcshaga, 7),
            "uniform_tour_k": (uniform_tour_crossover_selfcshaga, self._parents_num),
        }

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
        self._population_ph_i, self._fitness_i = self._evaluate_population(self._population_g_i)
        self._fitness_scale_i = minmax_scale(self._fitness_i)
        self._fitness_rank_i = rankdata(self._fitness_i)

    def _randc(
        self: SHAGA,
        u: float,
        scale: float,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> NDArray[np.float64]:
        value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        while value <= lower or value > upper:
            value = cauchy_distribution(loc=u, scale=scale, size=1)[0]

        return value

    def _randn(
        self: SHAGA,
        u: float,
        scale: float,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> float:
        value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        if value < lower:
            value = lower
        elif value > upper:
            value = upper
        return value

    def _generate_MR_CR(
        self: SHAGA,
        randc_scale: float,
        randc_lower: float,
        randc_upper: float,
        randn_scale: float,
        randn_lower: float,
        randn_upper: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        MR_i = np.zeros(self._pop_size)
        CR_i = np.zeros(self._pop_size)
        for i in range(self._pop_size):
            r_i = randint(0, self._H_size, 1)[0]
            u_MR = self._H_MR[r_i]
            u_CR = self._H_CR[r_i]
            MR_i[i] = self._randc(u=u_MR, scale=randc_scale, lower=randc_lower, upper=randc_upper)
            CR_i[i] = self._randn(u=u_CR, scale=randn_scale, lower=randn_lower, upper=randn_upper)
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
        specified_selection: str,
        specified_crossover: str,
        individ_g: NDArray[np.float64],
        MR: float,
        CR: float,
    ) -> NDArray[np.float64]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]

        selected_id = selection_func(
            self._fitness_scale_i,
            self._fitness_rank_i,
            np.int64(tour_size),
            np.int64(quantity),
        )

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

    def _replace_population(
        self,
        mutant_g: NDArray,
        mutant_ph: NDArray,
        mutant_fit: NDArray,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        mask = mutant_fit >= self._fitness_i
        succeses = mutant_fit > self._fitness_i

        succeses_MR = self._MR[succeses]
        succeses_CR = self._CR[succeses]

        will_be_replaced_fit = self._fitness_i[succeses].copy()

        self._population_g_i[mask] = mutant_g[mask]
        self._population_ph_i[mask] = mutant_ph[mask]
        self._fitness_i[mask] = mutant_fit[mask]
        self._fitness_scale_i = minmax_scale(self._fitness_i)
        self._fitness_rank_i = rankdata(self._fitness_i)

        d_fitness = np.abs(will_be_replaced_fit - self._fitness_i[succeses])
        d_fitness[d_fitness == np.inf] = 1e4

        return succeses_MR, succeses_CR, d_fitness

    def _get_new_population(self: SHAGA) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
            self._specified_selection,
            self._specified_crossover,
        )
        self._MR, self._CR = self._generate_MR_CR(
            randc_scale=0.1 / self._str_len,
            randc_lower=0.0,
            randc_upper=5 / self._str_len,
            randn_scale=0.1,
            randn_lower=0.0,
            randn_upper=1.0,
        )

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(individ_g=self._population_g_i[i], MR=self._MR[i], CR=self._CR[i])
                for i in range(self._pop_size)
            ],
            dtype=np.float64,
        )

        mutant_cr_ph, mutant_cr_fit = self._evaluate_population(mutant_cr_b_g)
        succeses_MR, succeses_CR, d_fitness = self._replace_population(
            mutant_cr_b_g,
            mutant_cr_ph,
            mutant_cr_fit,
        )

        next_k = (self._k + 1) % self._H_size
        self._H_MR[next_k] = self._update_u(self._H_MR[self._k], succeses_MR, d_fitness)
        self._H_CR[next_k] = self._update_u(self._H_CR[self._k], succeses_CR, d_fitness)
        self._k = next_k

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
