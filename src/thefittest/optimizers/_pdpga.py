from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._selfcga import SelfCGA
from ..base import Tree
from ..utils import numpy_group_by
from ..utils.random import random_sample


class PDPGA(SelfCGA):
    """
    Genetic Algorithm with Population-level Dynamic Probabilities (PDP).

    PDPGA implements the PDP method for operator probability adaptation, which
    assigns each operator a minimum application probability (threshold) and
    dynamically adjusts probabilities based on operator success rates.

    Parameters
    ----------
    fitness_function : Callable[[NDArray[Any]], NDArray[np.float64]]
        Function to evaluate fitness of solutions. Should accept a 2D array
        and return a 1D array of fitness values.
    iters : int
        Maximum number of iterations (generations) to run the algorithm.
    pop_size : int
        Number of individuals in the population.
    str_len : int
        Length of the binary string (genotype length).
    tour_size : int, optional (default=2)
        Tournament size for tournament selection operators.
    mutation_rate : float, optional (default=0.05)
        Mutation rate for custom mutation strategies.
    parents_num : int, optional (default=2)
        Number of parents used in crossover operations.
    elitism : bool, optional (default=True)
        If True, the best solution is always preserved in the next generation.
    selections : Tuple[str, ...], optional
        Tuple of selection operator names to use in the adaptive pool.
        Available operators (all from GeneticAlgorithm):

        - 'proportional': Fitness proportional selection
        - 'rank': Rank-based selection
        - 'tournament_k': Tournament selection with tour_size
        - 'tournament_3', 'tournament_5', 'tournament_7': Fixed size tournaments

        Default: ('proportional', 'rank', 'tournament_3', 'tournament_5', 'tournament_7')
    crossovers : Tuple[str, ...], optional
        Tuple of crossover operator names to use in the adaptive pool.
        Available operators (all from GeneticAlgorithm except gp_* variants):

        - 'empty': No crossover (cloning)
        - 'one_point': Single-point crossover
        - 'two_point': Two-point crossover
        - 'uniform_2', 'uniform_7', 'uniform_k': Uniform crossover with N parents
        - 'uniform_prop_2', 'uniform_prop_7', 'uniform_prop_k': Fitness-proportional uniform
        - 'uniform_rank_2', 'uniform_rank_7', 'uniform_rank_k': Rank-based uniform
        - 'uniform_tour_3', 'uniform_tour_7', 'uniform_tour_k': Tournament-based uniform

        Default: ('empty', 'one_point', 'two_point', 'uniform_2', 'uniform_7',
                  'uniform_prop_2', 'uniform_prop_7', 'uniform_rank_2',
                  'uniform_rank_7', 'uniform_tour_3', 'uniform_tour_7')
    mutations : Tuple[str, ...], optional
        Tuple of mutation operator names to use in the adaptive pool.
        Available operators (all from GeneticAlgorithm except gp_* variants):

        - 'weak': Flip 1/3 of bits on average
        - 'average': Flip 1 bit on average
        - 'strong': Flip 3 bits on average
        - 'custom_rate': Use specified mutation_rate

        Default: ('weak', 'average', 'strong')
    init_population : Optional[NDArray[np.byte]], optional (default=None)
        Initial population. If None, population is randomly initialized.
    genotype_to_phenotype : Optional[Callable], optional (default=None)
        Function to decode genotype to phenotype. If None, genotype equals phenotype.
    optimal_value : Optional[float], optional (default=None)
        Known optimal value for termination.
    termination_error_value : float, optional (default=0.0)
        Acceptable error from optimal value for termination.
    no_increase_num : Optional[int], optional (default=None)
        Stop if no improvement for this many iterations.
    minimization : bool, optional (default=False)
        If True, minimize; if False, maximize.
    show_progress_each : Optional[int], optional (default=None)
        Print progress every N iterations.
    keep_history : bool, optional (default=False)
        If True, keeps history of populations and fitness values.
    n_jobs : int, optional (default=1)
        Number of parallel jobs. -1 uses all processors.
    fitness_function_args : Optional[Dict], optional (default=None)
        Additional arguments to pass to fitness function.
    genotype_to_phenotype_args : Optional[Dict], optional (default=None)
        Additional arguments to pass to genotype_to_phenotype function.
    random_state : Optional[Union[int, np.random.RandomState]], optional (default=None)
        Random state for reproducibility.
    on_generation : Optional[Callable], optional (default=None)
        Callback function called after each generation.
    fitness_update_eps : float, optional (default=0.0)
        Minimum improvement threshold.

    Notes
    -----
    The threshold probabilities are automatically set to :math:`0.2 / n` for each
    operator type, where :math:`n` is the number of operators. This provides a
    balanced initial distribution that adapts based on operator performance during
    evolution. The method evaluates operator success by comparing offspring fitness
    with a randomly selected parent from the crossover pool.

    References
    ----------
    .. [1] Niehaus, J., Banzhaf, W. (2001). Adaption of Operator Probabilities in
           Genetic Programming. In: Miller, J., Tomassini, M., Lanzi, P.L., Ryan, C.,
           Tettamanzi, A.G.B., Langdon, W.B. (eds) Genetic Programming. EuroGP 2001.
           Lecture Notes in Computer Science, vol 2038. Springer, Berlin, Heidelberg.
           https://doi.org/10.1007/3-540-45355-5_26

    Examples
    --------
    **Example 1: OneMax Problem**

    >>> from thefittest.benchmarks import OneMax
    >>> from thefittest.optimizers import PDPGA
    >>>
    >>> optimizer = PDPGA(
    ...     fitness_function=OneMax(),
    ...     iters=150,
    ...     pop_size=100,
    ...     str_len=500,
    ...     show_progress_each=30
    ... )
    >>>
    >>> optimizer.fit()
    >>> fittest = optimizer.get_fittest()
    >>> print('Best fitness:', fittest['fitness'])

    **Example 2: Custom Binary Problem with Operator Selection**

    >>> import numpy as np
    >>>
    >>> def custom_fitness(X):
    ...     return X.sum(axis=1).astype(np.float64)
    >>>
    >>> # Create optimizer with specific operator pools
    >>> optimizer = PDPGA(
    ...     fitness_function=custom_fitness,
    ...     iters=100,
    ...     pop_size=50,
    ...     str_len=100,
    ...     selections=('rank', 'tournament_3', 'tournament_5'),
    ...     crossovers=('one_point', 'two_point', 'uniform_2'),
    ...     mutations=('weak', 'average', 'strong'),
    ...     keep_history=True
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
        selections: Tuple[str, ...] = (
            "proportional",
            "rank",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = (
            "empty",
            "one_point",
            "two_point",
            "uniform_2",
            "uniform_7",
            "uniform_prop_2",
            "uniform_prop_7",
            "uniform_rank_2",
            "uniform_rank_7",
            "uniform_tour_3",
            "uniform_tour_7",
        ),
        mutations: Tuple[str, ...] = ("weak", "average", "strong"),
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
        SelfCGA.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            selections=selections,
            crossovers=crossovers,
            mutations=mutations,
            selection_threshold_proba=0.2 / len(selections),
            crossover_threshold_proba=0.2 / len(crossovers),
            mutation_threshold_proba=0.2 / len(mutations),
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

        self._previous_fitness_i: List = []
        self._success_i: NDArray[np.bool_]

    def _choice_parent(self: PDPGA, fitness_i_selected: NDArray[np.float64]) -> np.float64:
        index = random_sample(len(fitness_i_selected), 1, True)[0]
        choosen = fitness_i_selected[index]
        return choosen

    def _get_new_proba_pdp(
        self: PDPGA,
        proba_dict: Dict["str", float],
        operators: NDArray,
        threshold: float,
    ) -> Dict["str", float]:
        n = len(proba_dict)
        operators, r_values = self._culc_r_i(self._success_i, operators, list(proba_dict.keys()))
        scale = np.sum(r_values)
        proba_value = threshold + (r_values * ((1.0 - n * threshold) / scale))
        new_proba_dict = dict(zip(operators, proba_value))
        new_proba_dict = dict(sorted(new_proba_dict.items()))
        return new_proba_dict

    def _culc_r_i(
        self: PDPGA, success: NDArray[np.bool_], operator: NDArray, operator_keys: List[str]
    ) -> Tuple:
        keys, groups = numpy_group_by(group=success, by=operator)
        r_values = np.array(list(map(lambda x: (np.sum(x) ** 2 + 1) / (len(x) + 1), groups)))
        for key in operator_keys:
            if key not in keys:
                keys = np.append(keys, key)
                r_values = np.append(r_values, 0.0)

        return keys, r_values

    def _adapt(self: PDPGA) -> None:
        if len(self._previous_fitness_i):
            self._success_i = np.array(self._previous_fitness_i, dtype=np.float64) < self._fitness_i

            self._selection_proba = self._get_new_proba_pdp(
                self._selection_proba, self._selection_operators, self._thresholds["selection"]
            )

            self._crossover_proba = self._get_new_proba_pdp(
                self._crossover_proba, self._crossover_operators, self._thresholds["crossover"]
            )

            self._mutation_proba = self._get_new_proba_pdp(
                self._mutation_proba, self._mutation_operators, self._thresholds["mutation"]
            )

            self._previous_fitness_i = []

    def _get_new_individ_g(
        self: PDPGA,
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

        previous_fitness = self._choice_parent(self._fitness_i[selected_id])
        self._previous_fitness_i.append(previous_fitness)

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
