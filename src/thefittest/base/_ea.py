from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from joblib import Parallel
from joblib import cpu_count
from joblib import delayed

import numpy as np
from numpy.typing import NDArray

from ..utils.random import check_random_state
from ..utils.multiobjective import fast_non_dominated_sort
from ..utils.multiobjective import crowding_distance_assignment


class TheFittest:
    def __init__(self) -> None:
        self._genotype: Any
        self._phenotype: Any
        self._fitness: float = -np.inf
        self._no_update_counter: int = 0

    def _update(
        self,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
        eps: float = 0.0,
    ) -> None:
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness - self._fitness > eps:
            self._replace(
                new_genotype=population_g[temp_best_id],
                new_phenotype=population_ph[temp_best_id],
                new_fitness=temp_best_fitness,
            )
            self._no_update_counter = 0
        else:
            self._no_update_counter += 1

    def _replace(self, new_genotype: Any, new_phenotype: Any, new_fitness: float) -> None:
        self._genotype = new_genotype.copy()
        self._phenotype = new_phenotype.copy()
        self._fitness = new_fitness

    def get(self) -> Dict:
        return {
            "genotype": self._genotype.copy(),
            "phenotype": self._phenotype.copy(),
            "fitness": self._fitness,
        }


class ParetoFront:
    """Store a snapshot of the first front of the most recently supplied population.

    All fitness objectives are maximized. Updates replace the current front;
    this class does not maintain an archive across generations or calculate
    crowding distances. Equal fitness vectors are retained as separate solutions.
    Numeric arrays and mutable individuals in object arrays are copied both
    when storing data and when returning it.

    ``_no_update_counter`` counts consecutive updates with an unchanged set
    of fitness vectors, ignoring row order, duplicates, genotypes, and phenotypes.
    With a positive ``eps``, every vector in each compared front must have a
    counterpart within absolute tolerance in every objective. Comparison uses
    the last significantly changed front, so small shifts can accumulate.
    """

    def __init__(self) -> None:
        self._genotype: NDArray[Any] = np.empty(0)
        self._phenotype: NDArray[Any] = np.empty(0)
        self._fitness: NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
        self._indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._no_update_counter: int = 0
        self._reference_fitness: NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)

    def _update(
        self,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
        ranks: Optional[NDArray[np.int64]] = None,
        eps: float = 0.0,
    ) -> None:
        """Replace the snapshot using rows whose nondomination rank is 1.

        ``fitness`` must have shape (N, M), with M >= 1 and no NaN values.
        Both populations must have N rows. Optional ``ranks`` must contain
        N positive integers aligned with these exact population rows. When
        supplied, ranks are trusted and nondominated sorting is skipped.
        Otherwise, ranks are computed using ``fast_non_dominated_sort``.
        ``eps`` is a non-negative absolute tolerance for stability comparison,
        shared by all objectives. Zero requires exact set equality.
        """
        if not np.isfinite(eps) or eps < 0:
            raise ValueError("eps must be finite and non-negative.")
        fitness = np.asarray(fitness, dtype=np.float64)
        if fitness.ndim != 2 or fitness.shape[1] == 0:
            raise ValueError("fitness must be a matrix with at least one objective.")
        if np.isnan(fitness).any():
            raise ValueError("fitness must not contain NaN values.")
        if population_g.ndim == 0 or population_ph.ndim == 0:
            raise ValueError("populations must have a population axis.")
        if len(population_g) != len(fitness) or len(population_ph) != len(fitness):
            raise ValueError("populations and fitness must have the same number of rows.")

        if ranks is None:
            ranks = fast_non_dominated_sort(fitness)
        else:
            ranks = np.asarray(ranks)
            if ranks.ndim != 1 or len(ranks) != len(fitness):
                raise ValueError("ranks must be a vector with one entry per individual.")
            if not np.issubdtype(ranks.dtype, np.integer) or np.any(ranks < 1):
                raise ValueError("ranks must contain positive integers starting at 1.")
            if len(ranks) and not np.any(ranks == 1):
                raise ValueError("a nonempty population must contain a rank-1 solution.")

        indices = np.flatnonzero(ranks == 1)
        genotype = deepcopy(population_g[indices])
        phenotype = deepcopy(population_ph[indices])
        front_fitness = fitness[indices].copy()
        previous_values = self._reference_fitness
        current_values = np.unique(front_fitness, axis=0)
        unchanged = np.array_equal(previous_values, current_values)
        if not unchanged and eps > 0 and previous_values.shape[1] == current_values.shape[1]:
            unchanged = True
            for source, target in (
                (previous_values, current_values),
                (current_values, previous_values),
            ):
                for value in source:
                    close = np.isclose(target, value, atol=eps, rtol=0.0)
                    if not np.any(np.all(close, axis=1)):
                        unchanged = False
                        break
                if not unchanged:
                    break
        if unchanged:
            self._no_update_counter += 1
        else:
            self._no_update_counter = 0
            self._reference_fitness = current_values
        self._genotype = genotype
        self._phenotype = phenotype
        self._fitness = front_fitness
        self._indices = indices

    def get(self) -> Dict[str, Any]:
        """Return copies of genotype, phenotype, fitness, and population indices.

        Indices refer to the population passed to the most recent update,
        with the original row order preserved. Before any update, arrays
        are empty and fitness has shape (0, 0). An empty-population update
        preserves the supplied arrays' trailing dimensions.
        """
        return {
            "genotype": deepcopy(self._genotype),
            "phenotype": deepcopy(self._phenotype),
            "fitness": self._fitness.copy(),
            "indices": self._indices.copy(),
        }


class Statistics(dict):
    def _update(self, arg: Dict[str, Any]) -> None:
        for key, value in arg.items():
            try:
                value_to_append = value.copy()
            except AttributeError:
                value_to_append = value
            if key not in self.keys():
                self[key] = [value_to_append]
            else:
                self[key].append(value_to_append)


class EvolutionaryAlgorithm:
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        elitism: bool = True,
        init_population: Optional[
            Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        ] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        optimal_value: Optional[Union[float, int]] = None,
        termination_error_value: Union[float, int] = 0.0,
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
        use_fitness_cache: bool = False,
        fitness_cache_size: Optional[int] = 1000,
    ):
        self._fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]] = fitness_function
        self._iters: int = iters
        self._pop_size: int = pop_size
        self._elitism: bool = elitism
        self._init_population: Optional[
            Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        ] = init_population
        self._genotype_to_phenotype: Optional[Callable] = genotype_to_phenotype
        self._no_increase_num: Optional[int] = no_increase_num
        self._show_progress_each: Optional[int] = show_progress_each
        self._keep_history: bool = keep_history
        self._n_jobs: int = self._get_n_jobs(n_jobs)
        self._fitness_update_eps: float = fitness_update_eps
        self._use_fitness_cache: bool = use_fitness_cache
        if fitness_cache_size is not None and fitness_cache_size < 0:
            raise ValueError("Parameter fitness_cache_size must be non-negative or None.")
        self._fitness_cache_size: Optional[int] = fitness_cache_size
        self._fitness_cache: OrderedDict[Any, Tuple[Any, Union[float, NDArray[np.float64]]]] = (
            OrderedDict()
        )
        self._fitness_shape: Tuple[int, ...] = ()

        if fitness_function_args is not None:
            self._fitness_function_args = fitness_function_args
        else:
            self._fitness_function_args = {}

        if genotype_to_phenotype_args is not None:
            self._genotype_to_phenotype_args = genotype_to_phenotype_args
        else:
            self._genotype_to_phenotype_args = {}

        self._sign: int = -1 if minimization else 1
        self._aim: Union[float, int] = self._get_aim(optimal_value, termination_error_value)
        self._calls: int = 0

        self._thefittest: TheFittest = TheFittest()
        self._stats: Statistics = Statistics()

        self._population_g_i: Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        self._population_ph_i: NDArray
        self._fitness_i: NDArray[np.float64]

        if self._n_jobs > 1:
            self._parallel = Parallel(self._n_jobs)

        self._random_state = random_state
        self._on_generation = on_generation

    def _first_generation(self: EvolutionaryAlgorithm) -> None:
        return None

    def _get_init_population(self: EvolutionaryAlgorithm) -> None:
        self._first_generation()

    def _get_aim(
        self: EvolutionaryAlgorithm,
        optimal_value: Optional[Union[float, int]],
        termination_error_value: Union[float, int],
    ) -> Union[float, int]:
        if optimal_value is not None:
            return self._sign * optimal_value - termination_error_value
        else:
            return np.inf

    def _show_progress(self: EvolutionaryAlgorithm, current_iter: int) -> None:
        if self._show_progress_each is not None:
            cond_show_now = current_iter % self._show_progress_each == 0
            if cond_show_now:
                current_best = self._sign * self._thefittest._fitness
                print(f"{current_iter}-th iteration with the best fitness = {current_best}")

    def _termitation_check(self: EvolutionaryAlgorithm) -> bool:
        cond_aim = self._thefittest._fitness >= self._aim
        cond_no_increase = self._thefittest._no_update_counter == self._no_increase_num
        return bool(cond_aim or cond_no_increase)

    def _update_fittest(
        self: EvolutionaryAlgorithm,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
    ) -> None:
        self._thefittest._update(
            population_g=population_g,
            population_ph=population_ph,
            fitness=fitness,
            eps=self._fitness_update_eps,
        )

    def _update_stats(self: EvolutionaryAlgorithm, **kwargs: Any) -> None:
        if self._keep_history:
            self._stats._update(kwargs)

    def _get_phenotype(self, population_g: NDArray[Any]) -> NDArray[Any]:
        if self._genotype_to_phenotype is not None:
            if self._n_jobs > 1:
                populations_g = self._split_population(population_g)
                populations_ph = self._parallel(
                    delayed(self._genotype_to_phenotype)(
                        populations_g_i, **self._genotype_to_phenotype_args
                    )
                    for populations_g_i in populations_g
                )
                population_ph = np.concatenate(populations_ph, axis=0)
            else:
                population_ph = self._genotype_to_phenotype(
                    population_g, **self._genotype_to_phenotype_args
                )
        else:
            population_ph = population_g
        return population_ph

    def _get_fitness(
        self: EvolutionaryAlgorithm, population_ph: NDArray[Any]
    ) -> NDArray[np.float64]:
        if self._n_jobs > 1:
            populations_ph = self._split_population(population_ph)
            values = self._parallel(
                delayed(self._fitness_function)(populations_ph_i, **self._fitness_function_args)
                for populations_ph_i in populations_ph
            )
            value = np.concatenate(values, axis=0)
        else:
            value = self._fitness_function(population_ph, **self._fitness_function_args)

        self._calls += len(value)
        fitness = self._sign * value
        return fitness

    def _get_genotype_cache_key(self: EvolutionaryAlgorithm, genotype: Any) -> Any:
        if hasattr(genotype, "signature"):
            return ("signature", genotype.signature())

        if isinstance(genotype, np.ndarray):
            if genotype.dtype == object:
                return (
                    "object_array",
                    genotype.shape,
                    tuple(self._get_genotype_cache_key(item) for item in genotype.flat),
                )

            contiguous = np.ascontiguousarray(genotype)
            return ("ndarray", contiguous.dtype.str, contiguous.shape, contiguous.tobytes())

        try:
            array = np.asarray(genotype)
            contiguous = np.ascontiguousarray(array)
            return ("array_like", contiguous.dtype.str, contiguous.shape, contiguous.tobytes())
        except (TypeError, ValueError):
            return ("repr", repr(genotype))

    def _save_fitness_cache(
        self: EvolutionaryAlgorithm,
        key: Any,
        phenotype: Any,
        fitness: Union[float, NDArray[np.float64]],
    ) -> None:
        try:
            cached_phenotype = phenotype.copy()
        except AttributeError:
            cached_phenotype = phenotype

        cached_fitness = fitness.copy() if isinstance(fitness, np.ndarray) else float(fitness)
        self._fitness_cache[key] = (cached_phenotype, cached_fitness)
        self._fitness_cache.move_to_end(key)

        if self._fitness_cache_size is not None:
            while len(self._fitness_cache) > self._fitness_cache_size:
                self._fitness_cache.popitem(last=False)

    def _evaluate_population(
        self: EvolutionaryAlgorithm, population_g: NDArray[Any]
    ) -> Tuple[NDArray[Any], NDArray[np.float64]]:
        if not self._use_fitness_cache:
            population_ph = self._get_phenotype(population_g)
            fitness = self._get_fitness(population_ph)
            return population_ph, fitness

        population_size = len(population_g)
        population_ph_list: List[Any] = [None] * population_size
        fitness = np.empty((population_size,) + self._fitness_shape, dtype=np.float64)
        missing_indexes: List[int] = []
        missing_keys: List[Any] = []
        pending_indexes: Dict[Any, List[int]] = {}

        for i, genotype in enumerate(population_g):
            key = self._get_genotype_cache_key(genotype)
            cached_value = self._fitness_cache.get(key)
            if cached_value is None:
                if key in pending_indexes:
                    pending_indexes[key].append(i)
                else:
                    pending_indexes[key] = [i]
                    missing_indexes.append(i)
                    missing_keys.append(key)
            else:
                phenotype, cached_fitness = cached_value
                try:
                    population_ph_list[i] = phenotype.copy()
                except AttributeError:
                    population_ph_list[i] = phenotype
                fitness[i] = cached_fitness
                self._fitness_cache.move_to_end(key)

        if missing_indexes:
            missing_population_g = population_g[missing_indexes]
            missing_population_ph = self._get_phenotype(missing_population_g)
            missing_fitness = self._get_fitness(missing_population_ph)

            for local_i, population_i in enumerate(missing_indexes):
                phenotype = missing_population_ph[local_i]
                fitness_value = missing_fitness[local_i]
                for duplicate_i in pending_indexes[missing_keys[local_i]]:
                    try:
                        population_ph_list[duplicate_i] = phenotype.copy()
                    except AttributeError:
                        population_ph_list[duplicate_i] = phenotype
                    fitness[duplicate_i] = fitness_value
                self._save_fitness_cache(
                    missing_keys[local_i],
                    phenotype,
                    fitness_value,
                )

        population_ph = np.array(population_ph_list)
        return population_ph, fitness

    def get_remains_calls(self: EvolutionaryAlgorithm) -> int:
        """
        Get the number of remaining fitness function calls.

        Returns the number of fitness evaluations that can still be performed
        based on the configured population size and number of iterations.

        Returns
        -------
        int
            Number of remaining fitness function calls.

        Examples
        --------
        >>> optimizer.fit()
        >>> remaining = optimizer.get_remains_calls()
        >>> print(f"Remaining calls: {remaining}")
        """
        return (self._pop_size * self._iters) - self._calls

    def get_calls(self: EvolutionaryAlgorithm) -> int:
        """
        Get the number of fitness function calls already performed.

        Returns the total number of times the fitness function has been
        evaluated since the start of the optimization process. This includes
        all evaluations across generations and parallel executions.

        Returns
        -------
        int
            Number of fitness function calls performed so far.

        See Also
        --------
        get_remains_calls : Returns the number of remaining fitness evaluations.

        Examples
        --------
        >>> optimizer.fit()
        >>> calls = optimizer.get_calls()
        >>> print(f"Fitness function was called {calls} times")
        """
        return self._calls

    def get_fittest(self: EvolutionaryAlgorithm) -> Dict:
        """
        Get the best solution found during optimization.

        Returns a dictionary containing the genotype, phenotype, and fitness
        value of the best individual found during the evolutionary process.

        Returns
        -------
        Dict
            Dictionary with keys:

            - 'genotype': array-like
                Internal representation of the best solution.
            - 'phenotype': array-like
                Decoded representation of the best solution.
            - 'fitness': float
                Fitness value of the best solution.

        Examples
        --------
        >>> optimizer.fit()
        >>> fittest = optimizer.get_fittest()
        >>> print('Best solution:', fittest['phenotype'])
        >>> print('Best fitness:', fittest['fitness'])
        """
        return self._thefittest.get()

    def get_stats(self: EvolutionaryAlgorithm) -> Statistics:
        """
        Get statistics collected during the optimization process.

        Returns a Statistics object (dictionary subclass) containing history
        of various metrics collected during evolution if `keep_history=True`
        was set during initialization.

        Returns
        -------
        Statistics
            Dictionary containing evolution statistics with keys such as:

            - 'fitness': list of arrays
                Fitness values of populations at each iteration.
            - 'population_g': list of arrays
                Genotype populations at each iteration.
            - 'population_ph': list of arrays
                Phenotype populations at each iteration.
            - 'max_fitness': list of float
                Maximum fitness at each iteration.

            Additional keys may be present depending on the specific algorithm.

        Notes
        -----
        Statistics are only collected if `keep_history=True` was specified
        during optimizer initialization. Otherwise, returns an empty dictionary.

        Examples
        --------
        >>> optimizer.fit()
        >>> stats = optimizer.get_stats()
        >>> print(f"Number of generations: {len(stats['max_fitness'])}")
        >>> print(f"Final max fitness: {stats['max_fitness'][-1]}")
        """
        return self._stats

    def _update_data(self: EvolutionaryAlgorithm) -> None:
        max_fitness_id = np.argmax(self._fitness_i)
        self._update_fittest(self._population_g_i, self._population_ph_i, self._fitness_i)
        self._update_stats(
            fitness=self._fitness_i,
            population_g=self._population_g_i,
            population_ph=self._population_ph_i,
            max_fitness=self._fitness_i[max_fitness_id],
            max_g=self._population_g_i[max_fitness_id],
            max_ph=self._population_ph_i[max_fitness_id],
        )

    def _adapt(self: EvolutionaryAlgorithm) -> None:
        return None

    def _from_population_g_to_fitness(self: EvolutionaryAlgorithm) -> None:
        self._population_ph_i, self._fitness_i = self._evaluate_population(self._population_g_i)

        self._update_data()

        if self._elitism:
            (
                self._population_g_i[-1],
                self._population_ph_i[-1],
                self._fitness_i[-1],
            ) = self._thefittest.get().values()

    def _get_new_population(self: EvolutionaryAlgorithm) -> None:
        return None

    def _get_n_jobs(self: EvolutionaryAlgorithm, n_jobs: int) -> int:
        if n_jobs < 0:
            return max(cpu_count() + 1 + n_jobs, 1)
        elif n_jobs == 0:
            raise ValueError("Parameter n_jobs == 0 has no meaning.")
        elif n_jobs > self._pop_size:
            return self._pop_size
        else:
            return n_jobs

    def _split_population(self: EvolutionaryAlgorithm, population: NDArray) -> List:
        jobs = min(self._n_jobs, len(population))
        indexes = np.linspace(start=0, stop=len(population), num=jobs + 1, dtype=np.int64)
        indexes = indexes[1:-1]

        population_split = np.split(population, indexes)
        return population_split

    def fit(self: EvolutionaryAlgorithm) -> EvolutionaryAlgorithm:
        """
        Execute the evolutionary optimization process.

        Runs the evolutionary algorithm for the specified number of iterations,
        evolving the population to optimize the fitness function. The method
        handles initialization, population evolution, fitness evaluation, and
        termination conditions.

        Returns
        -------
        EvolutionaryAlgorithm
            Returns self to allow method chaining.

        Notes
        -----
        The optimization process includes:

        1. Random state initialization
        2. Initial population generation
        3. Iterative evolution:

           - Selection and variation operations
           - Fitness evaluation
           - Population update
           - Progress tracking (if enabled)
           - Termination checking

        The process terminates when:

        - Maximum iterations reached
        - Optimal value achieved (if specified)
        - No improvement for specified number of iterations (if configured)

        Examples
        --------
        >>> optimizer.fit()
        >>> fittest = optimizer.get_fittest()
        >>> print('Best solution found:', fittest['phenotype'])
        """

        check_random_state(self._random_state)
        self._get_init_population()
        self._from_population_g_to_fitness()

        for i in range(self._iters - 1):
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                self._get_new_population()
                self._from_population_g_to_fitness()
                if self._on_generation is not None:
                    self._on_generation(self)

        return self


class MultiObjectiveEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    """Shared configuration and state for multiobjective optimizers.

    Each entry in ``minimization`` specifies whether the corresponding objective
    is minimized. Objective signs encode conversion to the internal maximization
    convention.

    ``no_increase_num`` optionally stops evolution after that many consecutive
    updates with an unchanged set of first-front fitness vectors, using the
    absolute tolerance ``fitness_update_eps``. ``optimal_value`` specifies
    targets in the original objective directions. A single solution must
    reach every target within ``termination_error_value`` to stop evolution.
    """

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        minimization: Tuple[bool, ...],
        no_increase_num: Optional[int] = None,
        init_population: Optional[
            Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        ] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        on_generation: Optional[Callable] = None,
        optimal_value: Optional[Tuple[float, ...]] = None,
        termination_error_value: float = 0.0,
        fitness_update_eps: float = 0.0,
        use_fitness_cache: bool = False,
        fitness_cache_size: Optional[int] = 1000,
    ) -> None:
        minimization = tuple(minimization)
        if not np.isfinite(fitness_update_eps) or fitness_update_eps < 0:
            raise ValueError("fitness_update_eps must be finite and non-negative.")
        if not np.isfinite(termination_error_value) or termination_error_value < 0:
            raise ValueError("termination_error_value must be finite and non-negative.")
        if no_increase_num is not None and (
            isinstance(no_increase_num, (bool, np.bool_))
            or not isinstance(no_increase_num, (int, np.integer))
            or no_increase_num < 1
        ):
            raise ValueError("no_increase_num must be a positive integer or None.")
        if not minimization or any(
            not isinstance(value, (bool, np.bool_)) for value in minimization
        ):
            raise ValueError("minimization must contain at least one boolean entry.")

        super().__init__(
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            elitism=False,
            no_increase_num=no_increase_num,
            init_population=init_population,
            genotype_to_phenotype=genotype_to_phenotype,
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
        self._minimization: Tuple[bool, ...] = minimization
        self._n_objectives: int = len(minimization)
        self._fitness_shape = (self._n_objectives,)
        self._objective_signs: NDArray[np.float64] = np.array(
            [-1.0 if value else 1.0 for value in minimization],
            dtype=np.float64,
        )
        self._objective_aim: Optional[NDArray[np.float64]] = None
        if optimal_value is not None:
            targets = np.asarray(optimal_value, dtype=np.float64)
            if targets.shape != (self._n_objectives,) or not np.all(np.isfinite(targets)):
                raise ValueError("optimal_value must contain one finite target per objective.")
            self._objective_aim = targets * self._objective_signs - termination_error_value
        self._pareto_front: ParetoFront = ParetoFront()
        self._pareto_rank_i: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._crowding_distance_i: NDArray[np.float64] = np.empty(0, dtype=np.float64)

    def _get_fitness(self, population_ph: NDArray[Any]) -> NDArray[np.float64]:
        """Evaluate an (N, M) objective matrix and convert it to maximization.

        Rows must correspond to the supplied population and columns to
        ``minimization``. The evaluation counter counts individuals, not
        objective components, after the result shape has been validated.
        """
        if self._n_jobs > 1:
            populations_ph = self._split_population(population_ph)
            values = self._parallel(
                delayed(self._fitness_function)(populations_ph_i, **self._fitness_function_args)
                for populations_ph_i in populations_ph
            )
            value = np.concatenate(values, axis=0)
        else:
            value = self._fitness_function(population_ph, **self._fitness_function_args)

        objectives = np.asarray(value, dtype=np.float64)
        expected_shape = (len(population_ph), self._n_objectives)
        if objectives.shape != expected_shape:
            raise ValueError(
                f"fitness_function must return shape {expected_shape}, got {objectives.shape}."
            )

        self._calls += len(population_ph)
        return objectives * self._objective_signs

    def _update_data(self) -> None:
        self._pareto_rank_i = fast_non_dominated_sort(self._fitness_i)
        self._crowding_distance_i = np.empty(len(self._fitness_i), dtype=np.float64)
        for rank in np.unique(self._pareto_rank_i):
            indices = np.flatnonzero(self._pareto_rank_i == rank)
            self._crowding_distance_i[indices] = crowding_distance_assignment(
                self._fitness_i[indices]
            )

        self._pareto_front._update(
            self._population_g_i,
            self._population_ph_i,
            self._fitness_i,
            ranks=self._pareto_rank_i,
            eps=self._fitness_update_eps,
        )
        self._update_stats(
            fitness=self._fitness_i,
            population_g=self._population_g_i,
            population_ph=self._population_ph_i,
            pareto_rank=self._pareto_rank_i,
            crowding_distance=self._crowding_distance_i,
        )

    def _termitation_check(self) -> bool:
        cond_aim = self._objective_aim is not None and np.any(
            np.all(self._fitness_i >= self._objective_aim, axis=1)
        )
        cond_no_increase = (
            self._no_increase_num is not None
            and self._pareto_front._no_update_counter >= self._no_increase_num
        )
        return bool(cond_aim or cond_no_increase)

    def _show_progress(self, current_iter: int) -> None:
        if self._show_progress_each is not None:
            if current_iter % self._show_progress_each == 0:
                front_size = np.count_nonzero(self._pareto_rank_i == 1)
                best_values = self._fitness_i.max(axis=0) * self._objective_signs
                objectives = ", ".join(
                    f"f{i + 1}={value:.6g}" for i, value in enumerate(best_values)
                )
                print(
                    f"{current_iter}-th iteration with Pareto front size = {front_size}; "
                    f"best objectives = [{objectives}]"
                )

    def get_pareto_front(self) -> Dict[str, Any]:
        """Return a copy of the current first front in original objective directions.

        The dictionary contains ``genotype``, ``phenotype``, ``fitness``, and
        ``indices``. Fitness has one column per entry in ``minimization``;
        indices refer to rows of the current population. Before an evaluated
        population is recorded, fitness has shape (0, n_objectives).
        """
        front = self._pareto_front.get()
        front["fitness"] = front["fitness"].reshape(-1, self._n_objectives) * self._objective_signs
        return front

    def get_fittest(self) -> Dict:
        """Reject single-solution retrieval without a rule for choosing a compromise."""
        raise NotImplementedError(
            "Multiobjective optimization has no single best solution. Use get_pareto_front()."
        )
