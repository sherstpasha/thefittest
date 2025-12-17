from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from numba import float64
from numba import njit

import numpy as np
from numpy.typing import NDArray

from ._differentialevolution import DifferentialEvolution
from ..utils import find_pbest_id
from ..utils.crossovers import binomial
from ..utils.mutations import current_to_pbest_1_archive_p_min
from ..utils.random import cauchy_distribution
from ..utils.random import randint
from ..utils.random import sattolo_shuffle_2d


@njit(float64[:](float64[:], float64[:], float64[:]))
def bounds_control_mean(
    array: NDArray[np.float64], left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    to_return = array.copy()
    size = len(array)
    for i in range(size):
        if array[i] < left[i]:
            to_return[i] = (left[i] + array[i]) / 2
        elif array[i] > right[i]:
            to_return[i] = (right[i] + array[i]) / 2
    return to_return


def lehmer_mean(
    x: NDArray[np.float64], power: int = 2, weight: Optional[NDArray[np.float64]] = None
) -> float:
    weight_arg: Union[NDArray[np.float64], int]
    if weight is None:
        weight_arg = 1
    else:
        weight_arg = weight

    x_up = weight_arg * np.power(x, power)
    x_down = weight_arg * np.power(x, power - 1)
    return np.sum(x_up) / np.sum(x_down)


@njit(float64(float64))
def randc01(u: np.float64) -> np.float64:
    value = cauchy_distribution(loc=u, scale=np.float64(0.1), size=np.int64(1))[0]
    while value <= 0:
        value = cauchy_distribution(loc=u, scale=np.float64(0.1), size=np.int64(1))[0]
    if value > 1:
        value = 1
    return value


@njit(float64(float64))
def randn01(u: np.float64) -> Union[float, np.float64]:
    value = np.random.normal(u, 0.1, size=1)[0]
    if value < 0:
        return 0.0
    elif value > 1:
        return 1.0
    return value


class SHADE(DifferentialEvolution):
    """
    Success-History based Adaptive Differential Evolution optimizer.

    SHADE is an advanced variant of Differential Evolution that adaptively adjusts
    its control parameters (F and CR) based on the success history of previous
    generations. It uses historical memory to guide parameter selection and
    incorporates an archive of recently replaced solutions.

    Parameters
    ----------
    fitness_function : Callable[[NDArray[Any]], NDArray[np.float64]]
        Function to evaluate fitness of solutions. Should accept a 2D array
        of shape (pop_size, num_variables) and return a 1D array of fitness
        values of shape (pop_size,).
    iters : int
        Maximum number of iterations (generations) to run the algorithm.
    pop_size : int
        Number of individuals in the population. Also determines the size
        of the historical memory for F and CR parameters.
    left_border : Union[float, int, np.number, NDArray[np.number]]
        Lower bound(s) for decision variables. Can be a scalar (same bound for
        all variables) or an array of shape (num_variables,).
    right_border : Union[float, int, np.number, NDArray[np.number]]
        Upper bound(s) for decision variables. Can be a scalar (same bound for
        all variables) or an array of shape (num_variables,).
    num_variables : int
        Number of decision variables (problem dimensionality).
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
        If True, keeps history of all populations, fitness values, and parameter histories.
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

    Notes
    -----
    SHADE uses:

    - Current-to-pbest/1 mutation strategy with archive
    - Adaptive F parameter sampled from Cauchy distribution
    - Adaptive CR parameter sampled from normal distribution
    - Success-history based parameter adaptation using Lehmer mean
    - External archive of inferior solutions

    References
    ----------
    .. [1] Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based
           parameter adaptation for Differential Evolution. 2013 IEEE Congress
           on Evolutionary Computation, CEC 2013. 71-78.
           10.1109/CEC.2013.6557555.

    Examples
    --------
    >>> from thefittest.optimizers import SHADE
    >>>
    >>> # Define a custom optimization problem
    >>> def custom_problem(x):
    ...     return (5 - x[:, 0])**2 + (12 - x[:, 1])**2
    >>>
    >>> # Set up problem parameters
    >>> n_dimension = 2
    >>> left_border = -100.
    >>> right_border = 100.
    >>> number_of_generations = 100
    >>> population_size = 100
    >>>
    >>> # Create SHADE optimizer
    >>> optimizer = SHADE(
    ...     fitness_function=custom_problem,
    ...     iters=number_of_generations,
    ...     pop_size=population_size,
    ...     left_border=left_border,
    ...     right_border=right_border,
    ...     num_variables=n_dimension,
    ...     show_progress_each=10,
    ...     minimization=True
    ... )
    >>>
    >>> # Run optimization
    >>> optimizer.fit()
    >>>
    >>> # Get results
    >>> fittest = optimizer.get_fittest()
    >>> print('The fittest individ:', fittest['phenotype'])
    >>> print('with fitness', fittest['fitness'])
    """

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        left_border: Union[float, int, np.number, NDArray[np.number]],
        right_border: Union[float, int, np.number, NDArray[np.number]],
        num_variables: int,
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

        self._F: NDArray[np.float64]
        self._CR: NDArray[np.float64]
        self._population_g_archive_i: NDArray[np.float64] = np.zeros(shape=(0, len(self._left)))
        self._population_archive: NDArray[np.float64]
        self._pbest_id: NDArray[np.int64]
        self._H_size: int = pop_size
        self._H_F = np.full(self._H_size, 0.5, dtype=np.float64)
        self._H_CR = np.full(self._H_size, 0.5, dtype=np.float64)
        self._k: int = 0
        self._p: float = 0.05

    def _generate_F_CR(self: SHADE) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        F_i = np.zeros(self._pop_size, dtype=np.float64)
        CR_i = np.zeros(self._pop_size, dtype=np.float64)
        for i in range(self._pop_size):
            r_i = randint(0, self._H_size, size=1)[0]
            u_F = self._H_F[r_i]
            u_CR = self._H_CR[r_i]
            F_i[i] = randc01(np.float64(u_F))
            CR_i[i] = randn01(np.float64(u_CR))
        return (F_i, CR_i)

    def _append_archive(
        self,
        archive: Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]],
        worse_g: Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]],
    ) -> Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]:
        archive = np.append(archive, worse_g, axis=0)
        if len(archive) > self._pop_size:
            archive = sattolo_shuffle_2d(archive)

            archive = archive[: self._pop_size]
        return archive

    def _update_u_F(self, u_F: float, S_F: NDArray[np.float64]) -> float:
        if len(S_F):
            return lehmer_mean(S_F)
        return u_F

    def _update_u_CR(
        self, u_CR: float, S_CR: NDArray[np.float64], df: NDArray[np.float64]
    ) -> float:
        if len(S_CR):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df / sum_
                return np.sum(weight_i * S_CR)
        return u_CR

    def _get_new_individ_g(
        self: SHADE,
        individ_g: NDArray[np.float64],
        F: float,
        CR: float,
    ) -> NDArray[np.float64]:
        mutant_g = current_to_pbest_1_archive_p_min(
            individ_g, self._population_g_i, self._pbest_id, F, self._population_archive
        )

        mutant_cr_g = binomial(individ_g, mutant_g, np.float64(CR))
        return bounds_control_mean(mutant_cr_g, self._left, self._right)

    def _get_new_population(self: SHADE) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )

        self._F, self._CR = self._generate_F_CR()
        self._pbest_id = find_pbest_id(self._fitness_i, np.float64(self._p))
        self._population_archive = np.vstack([self._population_g_i, self._population_g_archive_i])

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(individ_g=self._population_g_i[i], F=self._F[i], CR=self._CR[i])
                for i in range(self._pop_size)
            ],
            dtype=np.float64,
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= self._fitness_i
        succeses = mutant_cr_fit > self._fitness_i

        succeses_F = self._F[succeses]
        succeses_CR = self._CR[succeses]

        will_be_replaced_pop = self._population_g_i[succeses].copy()
        will_be_replaced_fit = self._fitness_i[succeses].copy()

        self._population_g_archive_i = self._append_archive(
            self._population_g_archive_i, will_be_replaced_pop
        )

        self._population_g_i[mask] = mutant_cr_b_g[mask]
        self._population_ph_i[mask] = mutant_cr_ph[mask]
        self._fitness_i[mask] = mutant_cr_fit[mask]

        d_fitness = np.abs(will_be_replaced_fit - self._fitness_i[succeses])

        if self._k + 1 == self._H_size:
            next_k = 0
        else:
            next_k = self._k + 1

        self._H_F[next_k] = self._update_u_F(self._H_F[self._k], succeses_F)
        self._H_CR[next_k] = self._update_u_CR(self._H_CR[self._k], succeses_CR, d_fitness)

        if self._k == self._H_size - 1:
            self._k = 0
        else:
            self._k += 1

    def _update_data(self: SHADE) -> None:
        super()._update_data()
        self._update_stats(H_F=self._H_F, H_CR=self._H_CR)
