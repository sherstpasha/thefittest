from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base import Tree, UniversalSet
from ..utils.random import flip_coin
from ._nsgpii import NSGPII
from ._pdpnsgaii import PDPNSGAII


class PDPNSGPII(NSGPII, PDPNSGAII):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset: UniversalSet,
        iters: int,
        pop_size: int,
        minimization: Tuple[bool, ...],
        mutation_rate: float = 0.05,
        parents_num: int = 7,
        selections: Tuple[str, ...] = ("rank_crowding", "dominance_crowding"),
        crossovers: Tuple[str, ...] = (
            "gp_empty",
            "gp_standard",
            "gp_one_point",
            "gp_uniform_2",
            "gp_uniform_7",
        ),
        mutations: Tuple[str, ...] = (
            "gp_weak_point",
            "gp_average_point",
            "gp_strong_point",
            "gp_weak_grow",
            "gp_average_grow",
            "gp_strong_grow",
        ),
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
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        PDPNSGAII.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=1,
            minimization=minimization,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            selections=selections,
            crossovers=crossovers,
            mutations=mutations,
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
            use_fitness_cache=use_fitness_cache,
            fitness_cache_size=fitness_cache_size,
        )
        self._uniset: UniversalSet = uniset
        self._max_level: int = max_level
        self._init_level: int = init_level
        self._crossover_rate: float = crossover_rate
        self._penalize_duplicates: bool = penalize_duplicates
        self._duplicate_mask_i: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)

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
        previous_fitness = self._choice_parent(self._fitness_i[selected_id])
        self._previous_fitness_i.append(previous_fitness)

        if self._crossover_rate == 1.0 or flip_coin(self._crossover_rate):
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
