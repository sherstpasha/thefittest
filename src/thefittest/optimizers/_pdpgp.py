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
from ._pdpga import PDPGA
from ..base import Tree
from ..base import UniversalSet


class PDPGP(GeneticProgramming, PDPGA):
    """
    Genetic Programming with Population-level Dynamic Probabilities (PDP).

    PDPGP extends genetic programming with the PDP method for adaptive operator
    selection. It dynamically adjusts probabilities of GP-specific selection,
    crossover, and mutation operators based on their success rates during evolution.

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
    .. [1] Niehaus, J., Banzhaf, W. (2001). Adaption of Operator Probabilities in
           Genetic Programming. In: Miller, J., Tomassini, M., Lanzi, P.L., Ryan, C.,
           Tettamanzi, A.G.B., Langdon, W.B. (eds) Genetic Programming. EuroGP 2001.
           Lecture Notes in Computer Science, vol 2038. Springer, Berlin, Heidelberg.
           https://doi.org/10.1007/3-540-45355-5_26
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
        PDPGA.__init__(
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

    def _get_new_individ_g(
        self: PDPGP,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
    ) -> Tree:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func, proba, is_constant_rate = self._mutation_pool[specified_mutation]

        selected_id = selection_func(
            self._fitness_scale_i, self._fitness_rank_i, np.int64(tour_size), np.int64(quantity)
        )

        previous_fitness = self._choice_parent(self._fitness_i[selected_id])
        self._previous_fitness_i.append(previous_fitness)

        offspring_no_mutated = crossover_func(
            self._population_g_i[selected_id],
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
        )

        if is_constant_rate:
            proba = proba
        else:
            proba = proba / len(offspring_no_mutated)

        offspring = mutation_func(offspring_no_mutated, self._uniset, proba, self._max_level)
        return offspring
