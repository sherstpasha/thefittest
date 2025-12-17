from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._differentialevolution import DifferentialEvolution
from ..utils.random import uniform


class jDE(DifferentialEvolution):
    """
    Self-adaptive Differential Evolution with control parameter adaptation.

    jDE (self-adaptive Differential Evolution) is a variant of DE that self-adapts
    the control parameters F (mutation factor) and CR (crossover rate) during
    evolution. Each individual has its own F and CR values that evolve along
    with the solution, allowing the algorithm to automatically tune these
    parameters for the problem at hand.

    Parameters
    ----------
    fitness_function : Callable[[NDArray[Any]], NDArray[np.float64]]
        Function to evaluate fitness of solutions. Should accept a 2D array
        of shape (pop_size, num_variables) and return a 1D array of fitness
        values of shape (pop_size,).
    iters : int
        Maximum number of iterations (generations) to run the algorithm.
    pop_size : int
        Number of individuals in the population.
    left_border : Union[float, int, np.number, NDArray[np.number]]
        Lower bound(s) for decision variables. Can be a scalar (same bound for
        all variables) or an array of shape (num_variables,).
    right_border : Union[float, int, np.number, NDArray[np.number]]
        Upper bound(s) for decision variables. Can be a scalar (same bound for
        all variables) or an array of shape (num_variables,).
    num_variables : int
        Number of decision variables (problem dimensionality).
    mutation : str, optional (default="rand_1")
        Mutation strategy to use. See DifferentialEvolution for available strategies.
    F_min : float, optional (default=0.1)
        Minimum value for mutation factor F.
    F_max : float, optional (default=0.9)
        Maximum value for mutation factor F.
    t_F : float, optional (default=0.1)
        Probability of updating F parameter for each individual.
    t_CR : float, optional (default=0.1)
        Probability of updating CR parameter for each individual.
    elitism : bool, optional (default=True)
        If True, the best solution is always preserved in the next generation.
    init_population : Optional[NDArray[np.float64]], optional (default=None)
        Initial population. If None, population is randomly initialized.
        Shape should be (pop_size, num_variables).
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
        If True, keeps history of all populations, fitness values, and F/CR parameters.
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
    .. [1] Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan &
           Zumer, Viljem. (2007). Self-Adapting Control Parameters in
           Differential Evolution: A Comparative Study on Numerical Benchmark
           Problems. Evolutionary Computation, IEEE Transactions on. 10.
           646 - 657. 10.1109/TEVC.2006.872133.

    Examples
    --------
    >>> from thefittest.benchmarks import Rastrigin
    >>> from thefittest.optimizers import jDE
    >>>
    >>> # Define problem parameters
    >>> n_dimension = 30
    >>> left_border = -5.12
    >>> right_border = 5.12
    >>> number_of_generations = 200
    >>> population_size = 100
    >>>
    >>> # Create jDE optimizer with self-adaptive parameters
    >>> optimizer = jDE(
    ...     fitness_function=Rastrigin(),
    ...     iters=number_of_generations,
    ...     pop_size=population_size,
    ...     left_border=left_border,
    ...     right_border=right_border,
    ...     num_variables=n_dimension,
    ...     mutation="rand_1",
    ...     F_min=0.1,
    ...     F_max=0.9,
    ...     t_F=0.1,
    ...     t_CR=0.1,
    ...     minimization=True,
    ...     show_progress_each=20,
    ...     keep_history=True
    ... )
    >>>
    >>> # Run optimization
    >>> optimizer.fit()
    >>>
    >>> # Get results
    >>> fittest = optimizer.get_fittest()
    >>> stats = optimizer.get_stats()
    >>>
    >>> print('The fittest individ:', fittest['phenotype'])
    >>> print('with fitness', fittest['fitness'])
    >>> print('Final F parameters:', stats['F'][-1])
    >>> print('Final CR parameters:', stats['CR'][-1])
    """

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        left_border: Union[float, int, np.number, NDArray[np.number]],
        right_border: Union[float, int, np.number, NDArray[np.number]],
        num_variables: int,
        mutation: str = "rand_1",
        F_min: float = 0.1,
        F_max: float = 0.9,
        t_F: float = 0.1,
        t_CR: float = 0.1,
        elitism: bool = True,
        init_population: Optional[NDArray[np.float64]] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[np.float64]], NDArray[Any]]] = None,
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
        DifferentialEvolution.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            left_border=left_border,
            right_border=right_border,
            num_variables=num_variables,
            mutation=mutation,
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
        )

        self._F_min: float = F_min
        self._F_max: float = F_max
        self._t_F: float = t_F
        self._t_CR: float = t_CR

        self._F: NDArray[np.float64] = np.full(self._pop_size, 0.5, dtype=np.float64)
        self._CR: NDArray[np.float64] = np.full(self._pop_size, 0.9, dtype=np.float64)

    def _get_mutate_F(self: jDE) -> NDArray[np.float64]:
        mutate_F = self._F.copy()
        mask = uniform(0.0, 1.0, size=self._pop_size) < self._t_F

        random_values = uniform(0.0, 1.0, size=np.sum(mask))
        mutate_F[mask] = self._F_min + random_values * self._F_max
        return mutate_F

    def _get_mutate_CR(self: jDE) -> NDArray[np.float64]:
        mutate_CR = self._CR.copy()
        mask = uniform(0.0, 1.0, size=self._pop_size) < self._t_CR

        random_values = uniform(0.0, 1.0, size=np.sum(mask))
        mutate_CR[mask] = random_values
        return mutate_CR

    def _get_new_population(self: jDE) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )

        mutate_F = self._get_mutate_F()
        mutate_CR = self._get_mutate_CR()

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(individ_g=self._population_g_i[i], F=mutate_F[i], CR=mutate_CR[i])
                for i in range(self._pop_size)
            ],
            dtype=np.float64,
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= self._fitness_i

        self._population_g_i[mask] = mutant_cr_b_g[mask]
        self._population_ph_i[mask] = mutant_cr_ph[mask]
        self._fitness_i[mask] = mutant_cr_fit[mask]
        self._F[mask] = mutate_F[mask]
        self._CR[mask] = mutate_CR[mask]

    def _update_data(self: jDE) -> None:
        super()._update_data()
        self._update_stats(F=self._F, CR=self._CR)
