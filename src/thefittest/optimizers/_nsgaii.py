from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base._ea import MultiObjectiveEvolutionaryAlgorithm
from ..utils.crossovers import empty_crossover
from ..utils.crossovers import one_point_crossover
from ..utils.crossovers import two_point_crossover
from ..utils.crossovers import uniform_crossover
from ..utils.crossovers import uniform_rank_crowding_crossover
from ..utils.mutations import flip_mutation
from ..utils.multiobjective import fast_non_dominated_sort
from ..utils.multiobjective import crowding_distance_assignment
from ..utils.random import random_sample
from ..utils.random import flip_coin
from ..utils.selections import rank_crowding_tournament_selection
from ..utils.selections import dominance_crowding_tournament_selection
from ._geneticalgorithm import GeneticAlgorithm


class NSGAII(MultiObjectiveEvolutionaryAlgorithm):
    """Configuration for binary-coded NSGA-II.

    ``selection`` chooses either "rank_crowding" (nondomination rank followed
    by crowding distance) or "dominance_crowding" (direct dominance followed
    by crowding distance). Both use binary tournaments.

    Binary crossover and mutation names follow GeneticAlgorithm: "empty",
    "one_point", "two_point", "uniform_2", "uniform_7", "uniform_k" for
    crossover; "weak", "average", "strong", "custom_rate" for mutation.
    "uniform_tour_3" and "uniform_tour_7" use three and seven selected
    parents, respectively, with a binary rank-and-crowding tournament for
    each gene, independently of the parent selection mode.
    ``parents_num`` controls "uniform_k". ``mutation_rate`` is the per-bit
    probability for "custom_rate"; other rates are scaled by chromosome length.
    ``crossover_rate`` is the probability of applying crossover; otherwise
    the first selected parent is copied before mutation.
    """

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        str_len: int,
        minimization: Tuple[bool, ...],
        selection: str = "dominance_crowding",
        crossover: str = "uniform_2",
        mutation: str = "weak",
        mutation_rate: float = 0.05,
        parents_num: int = 2,
        init_population: Optional[NDArray[np.byte]] = None,
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
    ) -> None:
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if selection not in ("rank_crowding", "dominance_crowding"):
            raise ValueError('selection must be "rank_crowding" or "dominance_crowding".')

        super().__init__(
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            minimization=minimization,
            init_population=init_population,
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
        )
        self._str_len: int = str_len
        self._specified_selection: str = selection
        self._specified_crossover: str = crossover
        self._specified_mutation: str = mutation
        self._mutation_rate: float = mutation_rate
        self._parents_num: int = parents_num
        self._crossover_rate: float = crossover_rate

        self._selection_pool: Dict[str, Tuple[Callable, int]] = {
            "rank_crowding": (rank_crowding_tournament_selection, 2),
            "dominance_crowding": (dominance_crowding_tournament_selection, 2),
        }
        self._crossover_pool: Dict[str, Tuple[Callable, int]] = {
            "empty": (empty_crossover, 1),
            "one_point": (one_point_crossover, 2),
            "two_point": (two_point_crossover, 2),
            "uniform_2": (uniform_crossover, 2),
            "uniform_7": (uniform_crossover, 7),
            "uniform_k": (uniform_crossover, self._parents_num),
            "uniform_tour_3": (uniform_rank_crowding_crossover, 3),
            "uniform_tour_7": (uniform_rank_crowding_crossover, 7),
        }
        self._mutation_pool: Dict[str, Tuple[Callable, Union[float, int], bool]] = {
            "weak": (flip_mutation, 1 / 3, False),
            "average": (flip_mutation, 1, False),
            "strong": (flip_mutation, 3, False),
            "custom_rate": (flip_mutation, self._mutation_rate, True),
        }
        if crossover not in self._crossover_pool:
            raise ValueError(f"Unknown binary crossover: {crossover}.")
        if mutation not in self._mutation_pool:
            raise ValueError(f"Unknown binary mutation: {mutation}.")

    def _first_generation(self) -> None:
        if self._init_population is None:
            self._population_g_i = GeneticAlgorithm.binary_string_population(
                self._pop_size, self._str_len
            )
        else:
            self._population_g_i = self._init_population.copy()

    def _get_init_population(self) -> None:
        super()._get_init_population()
        self._population_ph_i, self._fitness_i = self._evaluate_population(self._population_g_i)
        self._pareto_rank_i, self._crowding_distance_i = self._rank_and_crowding(self._fitness_i)

    def _from_population_g_to_fitness(self) -> None:
        self._update_data()

    @staticmethod
    def _rank_and_crowding(
        fitness: NDArray[np.float64],
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        ranks = fast_non_dominated_sort(fitness)
        distances = np.empty(len(fitness), dtype=np.float64)
        for rank in np.unique(ranks):
            indices = np.flatnonzero(ranks == rank)
            distances[indices] = crowding_distance_assignment(fitness[indices])
        return ranks, distances

    def _update_data(self) -> None:
        # Preserve crowding distances calculated before truncating the combined population.
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

    def _get_new_individ_g(
        self,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
    ) -> NDArray[np.byte]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func, proba, is_constant_rate = self._mutation_pool[specified_mutation]

        if specified_selection == "rank_crowding":
            selected_id = selection_func(
                self._pareto_rank_i, self._crowding_distance_i, tour_size, quantity
            )
        else:
            selected_id = selection_func(
                self._fitness_i, self._crowding_distance_i, tour_size, quantity
            )

        if self._crossover_rate == 1.0 or flip_coin(self._crossover_rate):
            ranks = self._pareto_rank_i[selected_id]
            if specified_crossover not in ("uniform_tour_3", "uniform_tour_7"):
                # Ordinary crossovers accept scalar metadata but do not use its values.
                ranks = ranks.astype(np.float64)
            offspring_no_mutated = crossover_func(
                self._population_g_i[selected_id],
                self._crowding_distance_i[selected_id],
                ranks,
            )
        else:
            offspring_no_mutated = self._population_g_i[selected_id[0]].copy()
        if not is_constant_rate:
            proba = proba / len(offspring_no_mutated)
        offspring = mutation_func(offspring_no_mutated, np.float64(proba))
        return offspring

    def _get_offspring_population(self) -> NDArray[np.byte]:
        offspring = np.array(
            [
                self._get_new_individ_g(
                    self._specified_selection,
                    self._specified_crossover,
                    self._specified_mutation,
                )
                for _ in range(self._pop_size)
            ],
            dtype=np.byte,
        )
        return offspring

    def _get_new_population(self) -> None:
        offspring_g = self._get_offspring_population()
        offspring_ph, offspring_fitness = self._evaluate_population(offspring_g)
        population_g = np.concatenate((self._population_g_i, offspring_g), axis=0)
        population_ph = np.concatenate((self._population_ph_i, offspring_ph), axis=0)
        fitness = np.concatenate((self._fitness_i, offspring_fitness), axis=0)
        ranks, distances = self._rank_and_crowding(fitness)
        survivors = self._get_survivor_indices(ranks, distances)
        self._replace_population(population_g, population_ph, fitness, ranks, distances, survivors)

    def _get_survivor_indices(
        self, ranks: NDArray[np.int64], distances: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        survivors = np.empty(self._pop_size, dtype=np.int64)
        filled = 0
        for rank in np.unique(ranks):
            indices = np.flatnonzero(ranks == rank)
            remaining = self._pop_size - filled
            if len(indices) > remaining:
                # Randomize equal distances without changing preference for larger ones.
                permutation = random_sample(len(indices), len(indices), False)
                indices = indices[permutation]
                order = np.argsort(-distances[indices], kind="mergesort")
                indices = indices[order[:remaining]]
            survivors[filled : filled + len(indices)] = indices
            filled += len(indices)
            if filled == self._pop_size:
                break
        return survivors

    def _replace_population(
        self,
        population_g: NDArray[np.byte],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
        ranks: NDArray[np.int64],
        distances: NDArray[np.float64],
        survivors: NDArray[np.int64],
    ) -> None:
        self._population_g_i = population_g[survivors]
        self._population_ph_i = population_ph[survivors]
        self._fitness_i = fitness[survivors]
        self._pareto_rank_i = ranks[survivors]
        self._crowding_distance_i = distances[survivors]
