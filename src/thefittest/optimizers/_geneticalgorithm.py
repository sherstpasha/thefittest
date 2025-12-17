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
from ..utils.random import randint


class GeneticAlgorithm(EvolutionaryAlgorithm):
    """
    Genetic Algorithm optimizer for binary and combinatorial optimization problems.

    Genetic Algorithm (GA) is a population-based evolutionary algorithm that uses
    selection, crossover, and mutation operators to evolve solutions encoded as
    binary strings. It is particularly effective for discrete optimization problems.

    Parameters
    ----------
    fitness_function : Callable[[NDArray[Any]], NDArray[np.float64]]
        Function to evaluate fitness of solutions. Should accept a 2D array
        of shape (pop_size, ...) and return a 1D array of fitness values.
    iters : int
        Maximum number of iterations (generations) to run the algorithm.
    pop_size : int
        Number of individuals in the population.
    str_len : int
        Length of the binary string (genotype length).
    tour_size : int, optional (default=2)
        Tournament size for tournament selection.
    mutation_rate : float, optional (default=0.05)
        Mutation rate for custom mutation strategies.
    parents_num : int, optional (default=2)
        Number of parents used in crossover operation.
    elitism : bool, optional (default=True)
        If True, the best solution is always preserved in the next generation.
    selection : str, optional (default="tournament_5")
        Selection strategy. Available strategies:

        - 'proportional': Fitness proportional selection
        - 'rank': Rank-based selection
        - 'tournament_k': Tournament selection with tour_size
        - 'tournament_3', 'tournament_5', 'tournament_7': Fixed size tournaments
    crossover : str, optional (default="uniform_2")
        Crossover strategy. Available strategies for binary GAs:

        - 'empty': No crossover (cloning)
        - 'one_point': Single-point crossover
        - 'two_point': Two-point crossover
        - 'uniform_2': Uniform crossover with 2 parents
        - 'uniform_7': Uniform crossover with 7 parents
        - 'uniform_k': Uniform crossover with parents_num parents
        - 'uniform_prop_2': Fitness-proportional uniform crossover with 2 parents
        - 'uniform_prop_7': Fitness-proportional uniform crossover with 7 parents
        - 'uniform_prop_k': Fitness-proportional uniform crossover with parents_num parents
        - 'uniform_rank_2': Rank-based uniform crossover with 2 parents
        - 'uniform_rank_7': Rank-based uniform crossover with 7 parents
        - 'uniform_rank_k': Rank-based uniform crossover with parents_num parents
        - 'uniform_tour_3': Tournament-based uniform crossover with 3 parents
        - 'uniform_tour_7': Tournament-based uniform crossover with 7 parents
        - 'uniform_tour_k': Tournament-based uniform crossover with parents_num parents

        Note: Operators starting with ``'gp_'`` are for Genetic Programming only.
    mutation : str, optional (default="weak")
        Mutation strategy. Available strategies:

        - 'weak': Flip 1/3 of bits on average
        - 'average': Flip 1 bit on average
        - 'strong': Flip 3 bits on average
        - 'custom_rate': Use specified mutation_rate
    init_population : Optional[NDArray[np.byte]], optional (default=None)
        Initial population. If None, population is randomly initialized.
        Shape should be (pop_size, str_len).
    genotype_to_phenotype : Optional[Callable], optional (default=None)
        Function to decode genotype to phenotype. If None, genotype equals phenotype.
    optimal_value : Optional[float], optional (default=None)
        Known optimal value for termination. Algorithm stops if this value is reached.
    termination_error_value : float, optional (default=0.0)
        Acceptable error from optimal value for termination.
    no_increase_num : Optional[int], optional (default=None)
        Stop if no improvement for this many iterations. If None, runs all iterations.
    minimization : bool, optional (default=False)
        If True, minimize the fitness function; if False, maximize.
    show_progress_each : Optional[int], optional (default=None)
        Print progress every N iterations. If None, no progress is shown.
    keep_history : bool, optional (default=False)
        If True, keeps history of all populations and fitness values.
    n_jobs : int, optional (default=1)
        Number of parallel jobs for fitness evaluation. -1 uses all processors.
    fitness_function_args : Optional[Dict], optional (default=None)
        Additional arguments to pass to fitness function.
    genotype_to_phenotype_args : Optional[Dict], optional (default=None)
        Additional arguments to pass to genotype_to_phenotype function.
    random_state : Optional[Union[int, np.random.RandomState]], optional (default=None)
        Random state for reproducibility.
    on_generation : Optional[Callable], optional (default=None)
        Callback function called after each generation.
    fitness_update_eps : float, optional (default=0.0)
        Minimum improvement threshold to consider a solution as better.

    References
    ----------
    .. [1] Holland, J. H. (1992). Genetic algorithms. Scientific American,
           267(1), 66-72.

    Examples
    --------
    **Example 1: OneMax Problem with 1000 bits**

    >>> from thefittest.benchmarks import OneMax
    >>> from thefittest.optimizers import GeneticAlgorithm
    >>>
    >>> number_of_iterations = 200
    >>> population_size = 200
    >>> string_length = 1000
    >>>
    >>> optimizer = GeneticAlgorithm(
    ...     fitness_function=OneMax(),
    ...     iters=number_of_iterations,
    ...     pop_size=population_size,
    ...     str_len=string_length,
    ...     show_progress_each=10
    ... )
    >>>
    >>> optimizer.fit()
    >>> fittest = optimizer.get_fittest()
    >>> print("Best fitness:", fittest["fitness"])
    >>> print("Solution found:", fittest["genotype"])

    **Example 2: Custom Binary Optimization**

    >>> import numpy as np
    >>>
    >>> # Define custom fitness function
    >>> def custom_fitness(X):
    ...     # Count ones in each solution
    ...     return X.sum(axis=1).astype(np.float64)
    >>>
    >>> optimizer = GeneticAlgorithm(
    ...     fitness_function=custom_fitness,
    ...     iters=100,
    ...     pop_size=50,
    ...     str_len=100,
    ...     selection='tournament_5',
    ...     crossover='uniform_2',
    ...     mutation='weak'
    ... )
    >>>
    >>> optimizer.fit()
    >>> fittest = optimizer.get_fittest()
    >>> print('Best fitness:', fittest['fitness'])
    """

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
        on_generation: Optional[Callable] = None,
        fitness_update_eps: float = 0.0,
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
            on_generation=on_generation,
            fitness_update_eps=fitness_update_eps,
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

    @staticmethod
    def binary_string_population(pop_size: int, str_len: int) -> NDArray[np.byte]:

        population = np.array(
            [randint(low=0, high=2, size=str_len) for _ in range(pop_size)],
            dtype=np.byte,
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
