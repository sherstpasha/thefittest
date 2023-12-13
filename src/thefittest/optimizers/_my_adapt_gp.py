from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import List

import numpy as np
from numpy.typing import NDArray

from ._geneticprogramming import GeneticProgramming
from thefittest.optimizers._my_adapt_ga import MyAdaptGA
from ..base import UniversalSet
from ..tools import donothing


class MyAdaptGP(GeneticProgramming, MyAdaptGA):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset: UniversalSet,
        iters: int,
        pop_size: int,
        tour_size: int = 2,
        mutation_rate: float = 0.05,
        parents_num: int = 2,
        elitism: bool = True,
        selections: Tuple[str, ...] = (
            "proportional",
            "rank",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = ("gp_standart", "gp_one_point", "gp_uniform_rank_2"),
        mutations: Tuple[str, ...] = (
            "gp_weak_point",
            "gp_average_point",
            "gp_strong_point",
            "gp_weak_grow",
            "gp_average_grow",
            "gp_strong_grow",
        ),
        max_level: int = 16,
        init_level: int = 4,
        init_population: Optional[NDArray] = None,
        adaptation_operator: str = "tournament_k",
        adaptation_tour_size: int = 2,
        genotype_to_phenotype: Callable[[NDArray], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
    ):
        MyAdaptGA.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=1,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            selections=selections,
            crossovers=crossovers,
            mutations=mutations,
            init_population=init_population,
            adaptation_operator=adaptation_operator,
            adaptation_tour_size=adaptation_tour_size,
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
        )

        GeneticProgramming.__init__(
            self,
            fitness_function=fitness_function,
            uniset=uniset,
            iters=iters,
            pop_size=pop_size,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            max_level=max_level,
            init_level=init_level,
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
        )
