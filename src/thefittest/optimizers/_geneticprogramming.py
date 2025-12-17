from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._geneticalgorithm import GeneticAlgorithm
from ..base import Tree
from ..base import UniversalSet
from ..utils.random import randint


class GeneticProgramming(GeneticAlgorithm):
    """
    Genetic Programming optimizer for symbolic regression and tree-based evolution.

    Genetic Programming (GP) evolves computer programs represented as tree structures
    using evolutionary operators adapted for tree manipulation. This implementation
    follows Koza's approach to genetic programming.

    Parameters
    ----------
    fitness_function : Callable[[NDArray[Any]], NDArray[np.float64]]
        Function to evaluate fitness of tree-based solutions.
    uniset : UniversalSet
        Universal set defining terminal and function nodes for tree construction.
    iters : int
        Maximum number of iterations (generations).
    pop_size : int
        Number of individuals (trees) in the population.
    tour_size : int, optional (default=2)
        Tournament size for tournament selection.
    mutation_rate : float, optional (default=0.05)
        Mutation rate for custom mutation strategies.
    parents_num : int, optional (default=7)
        Number of parents used in crossover operation.
    elitism : bool, optional (default=True)
        If True, the best solution is preserved.
    selection : str, optional (default="rank")
        Selection strategy. Available: 'proportional', 'rank', 'tournament_k',
        'tournament_3', 'tournament_5', 'tournament_7'.
    crossover : str, optional (default="gp_standard")
        Crossover strategy for trees. Available GP crossover operators:

        - 'gp_empty': No crossover (cloning)
        - 'gp_standard': Standard GP subtree crossover
        - 'gp_one_point': One-point crossover for trees
        - 'gp_uniform_2', 'gp_uniform_7', 'gp_uniform_k': Uniform crossover variants
        - 'gp_uniform_prop_2', 'gp_uniform_prop_7', 'gp_uniform_prop_k': Proportional uniform
        - 'gp_uniform_rank_2', 'gp_uniform_rank_7', 'gp_uniform_rank_k': Rank-based uniform
        - 'gp_uniform_tour_3', 'gp_uniform_tour_7', 'gp_uniform_tour_k': Tournament uniform
    mutation : str, optional (default="gp_weak_grow")
        Mutation strategy for trees. Available GP mutation operators:

        - 'gp_weak_point', 'gp_average_point', 'gp_strong_point': Point mutations
        - 'gp_weak_grow', 'gp_average_grow', 'gp_strong_grow': Growing mutations
        - 'gp_weak_swap', 'gp_average_swap', 'gp_strong_swap': Swap mutations
        - 'gp_weak_shrink', 'gp_average_shrink', 'gp_strong_shrink': Shrink mutations
        - 'gp_custom_rate_point', 'gp_custom_rate_grow', 'gp_custom_rate_swap',
          'gp_custom_rate_shrink': Custom rate variants
    max_level : int, optional (default=16)
        Maximum tree depth allowed during evolution.
    init_level : int, optional (default=5)
        Initial tree depth for population initialization.
    init_population : Optional[NDArray], optional (default=None)
        Initial population of trees. If None, randomly initialized.
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
        If True, keeps history of populations and fitness.
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
    .. [1] Koza, John R. (1993). Genetic Programming: On the Programming of
           Computers by Means of Natural Selection. MIT Press.
    """

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset: UniversalSet,
        iters: int,
        pop_size: int,
        tour_size: int = 2,
        mutation_rate: float = 0.05,
        parents_num: int = 7,
        elitism: bool = True,
        selection: str = "rank",
        crossover: str = "gp_standard",
        mutation: str = "gp_weak_grow",
        max_level: int = 16,
        init_level: int = 5,
        init_population: Optional[NDArray] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
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
        GeneticAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=1,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
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

        self._uniset: UniversalSet = uniset
        self._max_level: int = max_level
        self._init_level: int = init_level

    @staticmethod
    def half_and_half(pop_size: int, uniset: UniversalSet, max_level: int) -> NDArray:
        level = randint(2, max_level, 1)[0]
        population = [Tree.random_tree(uniset, level) for _ in range(pop_size)]
        population_numpy = np.array(population, dtype=object)
        return population_numpy

    def _first_generation(self: GeneticProgramming) -> None:
        if self._init_population is None:
            self._population_g_i = self._population_g_i = self.half_and_half(
                pop_size=self._pop_size, uniset=self._uniset, max_level=self._init_level
            )
        else:
            self._population_g_i = self._init_population.copy()

    def _get_new_individ_g(
        self: GeneticProgramming,
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
