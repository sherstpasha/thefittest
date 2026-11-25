from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base import Tree
from ..base import UniversalSet
from ..utils.crossovers import empty_crossoverGP
from ..utils.crossovers import one_point_crossoverGP
from ..utils.crossovers import standard_crossover
from ..utils.crossovers import uniform_crossoverGP
from ..utils.mutations import growing_mutation
from ..utils.mutations import point_mutation
from ..utils.mutations import shrink_mutation
from ..utils.mutations import swap_mutation
from ..utils.random import flip_coin
from ..utils.multiobjective import fast_non_dominated_sort
from ..utils.multiobjective import crowding_distance_assignment
from ._nsgaii import NSGAII
from ._geneticprogramming import GeneticProgramming


class NSGPII(NSGAII):
    """Multiobjective genetic programming using NSGA-II."""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset: UniversalSet,
        iters: int,
        pop_size: int,
        minimization: Tuple[bool, ...],
        selection: str = "dominance_crowding",
        crossover: str = "gp_standard",
        mutation: str = "gp_weak_grow",
        mutation_rate: float = 0.05,
        parents_num: int = 7,
        max_level: int = 16,
        init_level: int = 5,
        init_population: Optional[NDArray[Any]] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        optimal_value: Optional[Tuple[float, ...]] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        on_generation: Optional[Callable] = None,
        fitness_update_eps: float = 0.0,
        crossover_rate: float = 1.0,
        penalize_duplicates: bool = True,
        use_fitness_cache: bool = False,
        fitness_cache_size: Optional[int] = 1000,
    ) -> None:
        super().__init__(
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=1,
            minimization=minimization,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            init_population=init_population,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            n_jobs=n_jobs,
            fitness_function_args=fitness_function_args,
            genotype_to_phenotype_args=genotype_to_phenotype_args,
            random_state=random_state,
            on_generation=on_generation,
            fitness_update_eps=fitness_update_eps,
            crossover_rate=crossover_rate,
            use_fitness_cache=use_fitness_cache,
            fitness_cache_size=fitness_cache_size,
        )
        self._uniset: UniversalSet = uniset
        self._max_level: int = max_level
        self._init_level: int = init_level
        self._penalize_duplicates: bool = penalize_duplicates
        self._duplicate_mask_i: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)

    def _rank_and_crowding(
        self,
        fitness: NDArray[np.float64],
        population_g: Optional[NDArray[Any]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        self._duplicate_mask_i = np.zeros(len(fitness), dtype=np.bool_)
        if not self._penalize_duplicates:
            return super()._rank_and_crowding(fitness, population_g)
        if population_g is None:
            raise ValueError("Tree population is required to penalize duplicates.")

        ranks = fast_non_dominated_sort(fitness)
        seen_genotypes: set[Any] = set()
        for index in np.argsort(ranks, kind="mergesort"):
            genotype_key = self._get_genotype_cache_key(population_g[index])
            if genotype_key in seen_genotypes:
                self._duplicate_mask_i[index] = True
            else:
                seen_genotypes.add(genotype_key)
        if np.any(self._duplicate_mask_i):
            ranks[self._duplicate_mask_i] = ranks.max() + 1
        distances = np.empty(len(fitness), dtype=np.float64)
        for rank in np.unique(ranks):
            indices = np.flatnonzero(ranks == rank)
            distances[indices] = crowding_distance_assignment(fitness[indices])
        return ranks, distances

    def _replace_population(
        self,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
        ranks: NDArray[np.int64],
        distances: NDArray[np.float64],
        survivors: NDArray[np.int64],
    ) -> None:
        self._duplicate_mask_i = self._duplicate_mask_i[survivors]
        super()._replace_population(
            population_g, population_ph, fitness, ranks, distances, survivors
        )

    def _first_generation(self) -> None:
        if self._init_population is None:
            self._population_g_i = GeneticProgramming.half_and_half(
                pop_size=self._pop_size, uniset=self._uniset, max_level=self._init_level
            )
        else:
            self._population_g_i = self._init_population.copy()

    def _get_new_individ_g(
        self,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
    ) -> Tree:
        if specified_selection == "rank_crowding":
            selection_data = self._pareto_rank_i
        elif specified_selection == "dominance_crowding":
            selection_data = self._fitness_i
            if self._penalize_duplicates and np.any(self._duplicate_mask_i):
                selection_data = selection_data.copy()
                selection_data[self._duplicate_mask_i] = -np.inf
        else:
            raise ValueError(f"Unsupported selection: {specified_selection}.")

        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func, proba, is_constant_rate = self._mutation_pool[specified_mutation]

        selected_id = selection_func(selection_data, self._crowding_distance_i, tour_size, quantity)

        if self._crossover_rate == 1.0 or flip_coin(self._crossover_rate):
            # These GP crossovers accept scalar metadata but do not use its values.
            offspring_no_mutated = crossover_func(
                self._population_g_i[selected_id],
                self._crowding_distance_i[selected_id],
                self._pareto_rank_i[selected_id].astype(np.float64),
                self._max_level,
            )
        else:
            offspring_no_mutated = self._population_g_i[selected_id[0]].copy()

        if not is_constant_rate:
            proba = proba / len(offspring_no_mutated)
        return mutation_func(offspring_no_mutated, self._uniset, proba, self._max_level)

    def _init_variation_pools(self) -> None:
        self._crossover_pool: Dict[str, Tuple[Callable, int]] = {
            "gp_empty": (empty_crossoverGP, 1),
            "gp_standard": (standard_crossover, 2),
            "gp_one_point": (one_point_crossoverGP, 2),
            "gp_uniform_2": (uniform_crossoverGP, 2),
            "gp_uniform_7": (uniform_crossoverGP, 7),
            "gp_uniform_k": (uniform_crossoverGP, self._parents_num),
        }
        self._mutation_pool: Dict[str, Tuple[Callable, Union[float, int], bool]] = {
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
