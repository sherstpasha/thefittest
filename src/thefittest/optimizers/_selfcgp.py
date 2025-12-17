from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._geneticprogramming import GeneticProgramming
from ._selfcga import SelfCGA
from ..base import UniversalSet


class SelfCGP(GeneticProgramming, SelfCGA):
    """
    Self-Configuring Genetic Programming with modified uniform crossover.

    SelfCGP extends genetic programming with self-adaptive operator selection.
    It automatically configures GP-specific selection, crossover, and mutation
    operators during evolution based on their performance, using the same
    adaptation mechanism as SelfCGA but applied to tree-based operators.

    Parameters
    ----------
    fitness_function : Callable[[NDArray[Any]], NDArray[np.float64]]
        Function to evaluate fitness of tree-based solutions.
    uniset : UniversalSet
        Universal set defining terminal and function nodes.
    iters : int
        Maximum number of iterations (generations).
    pop_size : int
        Number of individuals (trees) in the population.
    tour_size : int, optional (default=2)
        Tournament size for tournament selection.
    mutation_rate : float, optional (default=0.05)
        Mutation rate for custom mutation strategies.
    parents_num : int, optional (default=2)
        Number of parents used in crossover.
    elitism : bool, optional (default=True)
        If True, the best solution is preserved.
    selections : Tuple[str, ...], optional
        Tuple of selection operator names to use in the adaptive pool.
        Available operators (same as GeneticAlgorithm):

        - 'proportional': Fitness proportional selection
        - 'rank': Rank-based selection
        - 'tournament_k': Tournament selection with tour_size
        - 'tournament_3', 'tournament_5', 'tournament_7': Fixed size tournaments

        Default: ('proportional', 'rank', 'tournament_3', 'tournament_5', 'tournament_7')
    crossovers : Tuple[str, ...], optional
        Tuple of GP crossover operator names to use in the adaptive pool.
        Available GP crossover operators:

        - 'gp_empty': No crossover (cloning)
        - 'gp_standard': Standard GP subtree crossover
        - 'gp_one_point': One-point crossover for trees
        - 'gp_uniform_2', 'gp_uniform_7', 'gp_uniform_k': Uniform crossover variants
        - 'gp_uniform_prop_2', 'gp_uniform_prop_7', 'gp_uniform_prop_k': Proportional
        - 'gp_uniform_rank_2', 'gp_uniform_rank_7', 'gp_uniform_rank_k': Rank-based
        - 'gp_uniform_tour_3', 'gp_uniform_tour_7', 'gp_uniform_tour_k': Tournament

        Default: ('gp_standard', 'gp_one_point', 'gp_uniform_rank_2')
    mutations : Tuple[str, ...], optional
        Tuple of GP mutation operator names to use in the adaptive pool.
        Available GP mutation operators:

        - Point mutations: 'gp_weak_point', 'gp_average_point', 'gp_strong_point'
        - Growing mutations: 'gp_weak_grow', 'gp_average_grow', 'gp_strong_grow'
        - Swap mutations: 'gp_weak_swap', 'gp_average_swap', 'gp_strong_swap'
        - Shrink mutations: 'gp_weak_shrink', 'gp_average_shrink', 'gp_strong_shrink'
        - Custom rate: 'gp_custom_rate_point', 'gp_custom_rate_grow',
          'gp_custom_rate_swap', 'gp_custom_rate_shrink'

        Default: ('gp_weak_point', 'gp_average_point', 'gp_strong_point',
                  'gp_weak_grow', 'gp_average_grow', 'gp_strong_grow')
    max_level : int, optional (default=16)
        Maximum tree depth allowed.
    init_level : int, optional (default=4)
        Initial tree depth.
    init_population : Optional[NDArray], optional (default=None)
        Initial population of trees.
    K : float, optional (default=2)
        Coefficient for probability adjustment based on operator success.
    selection_threshold_proba : float, optional (default=0.05)
        Minimum probability threshold for selection operators.
    crossover_threshold_proba : float, optional (default=0.05)
        Minimum probability threshold for crossover operators.
    mutation_threshold_proba : float, optional (default=0.05)
        Minimum probability threshold for mutation operators.
    genotype_to_phenotype : Optional[Callable], optional (default=None)
        Function to decode tree to phenotype.
    optimal_value : Optional[float], optional (default=None)
        Known optimal value for termination.
    termination_error_value : float, optional (default=0.0)
        Acceptable error from optimal value.
    no_increase_num : Optional[int], optional (default=None)
        Stop if no improvement for this many iterations.
    minimization : bool, optional (default=False)
        If True, minimize; if False, maximize.
    show_progress_each : Optional[int], optional (default=None)
        Print progress every N iterations.
    keep_history : bool, optional (default=False)
        If True, keeps history of populations.
    n_jobs : int, optional (default=1)
        Number of parallel jobs.
    fitness_function_args : Optional[Dict], optional (default=None)
        Additional arguments to fitness function.
    genotype_to_phenotype_args : Optional[Dict], optional (default=None)
        Additional arguments to genotype_to_phenotype.
    random_state : Optional[Union[int, np.random.RandomState]], optional (default=None)
        Random state for reproducibility.
    on_generation : Optional[Callable], optional (default=None)
        Callback after each generation.
    fitness_update_eps : float, optional (default=0.0)
        Minimum improvement threshold.

    References
    ----------
    .. [1] Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic
           programming algorithm with modified uniform crossover.
           IEEE Congress on Evolutionary Computation, 1-6.
           http://dx.doi.org/10.1109/CEC.2012.6256587
    """

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
        crossovers: Tuple[str, ...] = ("gp_standard", "gp_one_point", "gp_uniform_rank_2"),
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
        K: float = 2,
        selection_threshold_proba: float = 0.05,
        crossover_threshold_proba: float = 0.05,
        mutation_threshold_proba: float = 0.05,
        genotype_to_phenotype: Optional[Callable[[NDArray], NDArray[Any]]] = None,
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
    ):
        SelfCGA.__init__(
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
            K=K,
            selection_threshold_proba=selection_threshold_proba,
            crossover_threshold_proba=crossover_threshold_proba,
            mutation_threshold_proba=mutation_threshold_proba,
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
            random_state=random_state,
            on_generation=on_generation,
        )
