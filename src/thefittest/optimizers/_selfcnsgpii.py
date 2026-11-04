from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base import UniversalSet
from ._nsgpii import NSGPII
from ._selfcnsgaii import SelfCNSGAII


class SelfCNSGPII(NSGPII, SelfCNSGAII):
    """Self-configuring multiobjective genetic programming using NSGA-II."""

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
        K: float = 2.0,
        selection_threshold_proba: float = 0.05,
        crossover_threshold_proba: float = 0.05,
        mutation_threshold_proba: float = 0.05,
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
        SelfCNSGAII.__init__(
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
            K=K,
            selection_threshold_proba=selection_threshold_proba,
            crossover_threshold_proba=crossover_threshold_proba,
            mutation_threshold_proba=mutation_threshold_proba,
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
