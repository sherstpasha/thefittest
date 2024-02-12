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

from ..base import Tree
from ..base._ea import EvolutionaryAlgorithm
from ..utils.crossovers import empty_crossover
from ..utils.crossovers import empty_crossoverGP
from ..utils.mutations import flip_mutation
from ..utils.mutations import growing_mutation
from ..utils.crossovers import one_point_crossover
from ..utils.crossovers import one_point_crossoverGP
from ..utils.mutations import point_mutation
from ..utils.selections import proportional_selection
from ..utils.selections import rank_selection
from ..utils.mutations import shrink_mutation
from ..utils.crossovers import standard_crossover
from ..utils.mutations import swap_mutation
from ..utils.selections import tournament_selection
from ..utils.crossovers import two_point_crossover
from ..utils.crossovers import uniform_crossover
from ..utils.crossovers import uniform_crossoverGP
from ..utils.crossovers import uniform_proportional_crossover
from ..utils.crossovers import uniform_proportional_crossover_GP
from ..utils.crossovers import uniform_rank_crossover
from ..utils.crossovers import uniform_rank_crossover_GP
from ..utils.crossovers import uniform_tournament_crossover
from ..utils.crossovers import uniform_tournament_crossover_GP
from ..utils.transformations import minmax_scale


class GeneticAlgorithm(EvolutionaryAlgorithm):
    """Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-72"""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        str_len: int,
        tour_size: int = 2,
        mutation_rate: float = 0.05,
        parents_num: int = 2,
        elitism: bool = True,
        selection: str = "tournament_5",
        crossover: str = "uniform_2",
        mutation: str = "weak",
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
    ):
        EvolutionaryAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype,
            iters=iters,
            pop_size=pop_size,
            elitism=elitism,
            init_population=init_population,
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
        )
        self._str_len: int = str_len
        self._tour_size: int = tour_size
        self._parents_num: int = parents_num
        self._mutation_rate: float = mutation_rate
        self._specified_selection: str = selection
        self._specified_crossover: str = crossover
        self._specified_mutation: str = mutation

        self._selection_pool: Dict[str, Tuple[Callable, Union[float, int]]] = {
            "proportional": (proportional_selection, 0),
            "rank": (rank_selection, 0),
            "tournament_k": (tournament_selection, self._tour_size),
            "tournament_3": (tournament_selection, 3),
            "tournament_5": (tournament_selection, 5),
            "tournament_7": (tournament_selection, 7),
        }

        self._crossover_pool: Dict[str, Tuple[Callable, Union[float, int]]] = {
            "empty": (empty_crossover, 1),
            "one_point": (one_point_crossover, 2),
            "two_point": (two_point_crossover, 2),
            "uniform_2": (uniform_crossover, 2),
            "uniform_7": (uniform_crossover, 7),
            "uniform_k": (uniform_crossover, self._parents_num),
            "uniform_prop_2": (uniform_proportional_crossover, 2),
            "uniform_prop_7": (uniform_proportional_crossover, 7),
            "uniform_prop_k": (uniform_proportional_crossover, self._parents_num),
            "uniform_rank_2": (uniform_rank_crossover, 2),
            "uniform_rank_7": (uniform_rank_crossover, 7),
            "uniform_rank_k": (uniform_rank_crossover, self._parents_num),
            "uniform_tour_3": (uniform_tournament_crossover, 3),
            "uniform_tour_7": (uniform_tournament_crossover, 7),
            "uniform_tour_k": (uniform_tournament_crossover, self._parents_num),
            "gp_empty": (empty_crossoverGP, 1),
            "gp_standard": (standard_crossover, 2),
            "gp_one_point": (one_point_crossoverGP, 2),
            "gp_uniform_2": (uniform_crossoverGP, 2),
            "gp_uniform_7": (uniform_crossoverGP, 7),
            "gp_uniform_k": (uniform_crossoverGP, self._parents_num),
            "gp_uniform_prop_2": (uniform_proportional_crossover_GP, 2),
            "gp_uniform_prop_7": (uniform_proportional_crossover_GP, 7),
            "gp_uniform_prop_k": (uniform_proportional_crossover_GP, self._parents_num),
            "gp_uniform_rank_2": (uniform_rank_crossover_GP, 2),
            "gp_uniform_rank_7": (uniform_rank_crossover_GP, 7),
            "gp_uniform_rank_k": (uniform_rank_crossover_GP, self._parents_num),
            "gp_uniform_tour_3": (uniform_tournament_crossover_GP, 3),
            "gp_uniform_tour_7": (uniform_tournament_crossover_GP, 7),
            "gp_uniform_tour_k": (uniform_tournament_crossover_GP, self._parents_num),
        }

        self._mutation_pool: Dict[str, Tuple[Callable, Union[float, int], bool]] = {
            "weak": (flip_mutation, 1 / 3, False),
            "average": (flip_mutation, 1, False),
            "strong": (flip_mutation, 3, False),
            "custom_rate": (flip_mutation, self._mutation_rate, True),
            "gp_weak_point": (point_mutation, 0.25, False),
            "gp_average_point": (point_mutation, 1, False),
            "gp_strong_point": (point_mutation, 4, False),
            "gp_custom_rate_point": (point_mutation, self._mutation_rate, True),
            "gp_weak_grow": (growing_mutation, 0.25, False),
            "gp_average_grow": (growing_mutation, 1, False),
            "gp_strong_grow": (growing_mutation, 4, False),
            "gp_custom_rate_grow": (growing_mutation, self._mutation_rate, True),
            "gp_weak_swap": (swap_mutation, 0.25, False),
            "gp_average_swap": (swap_mutation, 1, False),
            "gp_strong_swap": (swap_mutation, 4, False),
            "gp_custom_rate_swap": (swap_mutation, self._mutation_rate, True),
            "gp_weak_shrink": (shrink_mutation, 0.25, False),
            "gp_average_shrink": (shrink_mutation, 1, False),
            "gp_strong_shrink": (shrink_mutation, 4, False),
            "gp_custom_rate_shrink": (shrink_mutation, self._mutation_rate, True),
        }

        self._fitness_scale_i: NDArray[np.float64]
        self._fitness_rank_i: NDArray[np.float64]

    def binary_string_population(self, pop_size: int, str_len: int) -> NDArray[np.byte]:
        size = (pop_size, str_len)
        population = self._random_state.randint(low=0, high=2, size=size, dtype=np.byte).astype(
            np.byte
        )
        return population

    def _first_generation(self: GeneticAlgorithm) -> None:
        if self._init_population is None:
            self._population_g_i = self.binary_string_population(self._pop_size, self._str_len)
        else:
            self._population_g_i = self._init_population.copy()

    def _get_new_individ_g(
        self: GeneticAlgorithm,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
    ) -> Union[Tree, np.byte]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func, proba, is_constant_rate = self._mutation_pool[specified_mutation]

        selected_id = selection_func(
            self._fitness_scale_i, self._fitness_rank_i, np.int64(tour_size), np.int64(quantity)
        )

        offspring_no_mutated = crossover_func(
            self._population_g_i[selected_id],
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
        )

        if is_constant_rate:
            proba = proba
        else:
            proba = proba / len(offspring_no_mutated)

        offspring = mutation_func(offspring_no_mutated, np.float64(proba))
        return offspring

    def _get_new_population(self: GeneticAlgorithm) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
            self._specified_selection,
            self._specified_crossover,
            self._specified_mutation,
        )

        self._population_g_i = np.array(
            [get_new_individ_g() for _ in range(self._pop_size)], dtype=self._population_g_i.dtype
        )

    def _from_population_g_to_fitness(self: GeneticAlgorithm) -> None:
        super()._from_population_g_to_fitness()

        self._fitness_scale_i = minmax_scale(self._fitness_i)
        self._fitness_rank_i = rankdata(self._fitness_i)

        self._adapt()
