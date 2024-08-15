from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import numpy as np
from numpy.typing import NDArray

from ..base import Tree
from ._geneticalgorithm import GeneticAlgorithm
from ..tools import donothing
from ..tools.operators import flip_mutation


class MyAdaptGAVar2(GeneticAlgorithm):
    """Semenkin, E.S., Semenkina, M.E. Self-configuring Genetic Algorithm with Modified Uniform
    Crossover Operator. LNCS, 7331, 2012, pp. 414-421. https://doi.org/10.1007/978-3-642-30976-2_50
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
        init_population: Optional[NDArray[np.byte]] = None,
        adaptation_operator: str = "rank",
        adaptation_tour_size: int = 2,
        mutate_operator_proba: float = 0.1,
        genotype_to_phenotype: Callable[[NDArray[np.byte]], NDArray[Any]] = donothing,
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
        GeneticAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
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
        )

        self._adaptation_operator = self._selection_pool[adaptation_operator]
        self._adaptation_tour_size = adaptation_tour_size
        self._mutate_thresholds_proba = mutate_operator_proba

        self._selection_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._crossover_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}

        for operator_name in selections:
            self._selection_set[operator_name] = self._selection_pool[operator_name]
        self._selection_set = dict(sorted(self._selection_set.items()))

        for operator_name in crossovers:
            self._crossover_set[operator_name] = self._crossover_pool[operator_name]
        self._crossover_set = dict(sorted(self._crossover_set.items()))

        self._selection_operators: NDArray = self._random_choice_operators(
            list(self._selection_set.keys()), self._pop_size
        )
        self._crossover_operators: NDArray = self._random_choice_operators(
            list(self._crossover_set.keys()), self._pop_size
        )

        self._min_mutation_rate = (1/3)/self._str_len
        self._max_mutation_rate = 3/self._str_len

        self._mutation_probas: NDArray = self._random_probas(
            self._min_mutation_rate, self._max_mutation_rate, self._pop_size
        )
        self.i = 0

    def _random_probas(
        self: MyAdaptGAVar2, left: float, right: float, size: int
    ) -> NDArray[np.float64]:
        probas = np.random.uniform(left, right, size)
        return probas

    def _mutate_probas(
        self: MyAdaptGAVar2, mutation_probas: NDArray[np.float64], mutate_thresholds_proba
    ) -> NDArray[np.float64]:
        new_probas = mutation_probas.copy()
        roll = np.random.random(size=len(mutation_probas)) < mutate_thresholds_proba
        n_replaces = np.sum(roll, dtype=int)
        if n_replaces > 0:
            new_probas[roll] = self._random_probas(
                self._min_mutation_rate, self._max_mutation_rate, size=n_replaces
            )
        return new_probas

    def _random_choice_operators(
        self: MyAdaptGAVar2, operators_set: List[str], size: int
    ) -> NDArray:
        chosen_operator = np.random.choice(operators_set, size)
        return chosen_operator

    def _mutate_operators(
        self: MyAdaptGAVar2, operators: NDArray, operators_set: List[str], mutate_thresholds_proba
    ) -> NDArray:
        new_operators = operators.copy()
        roll = np.random.random(size=len(operators)) < mutate_thresholds_proba
        n_replaces = np.sum(roll, dtype=int)

        if n_replaces > 0:
            new_operators[roll] = self._random_choice_operators(operators_set, size=n_replaces)
        return new_operators

    def _choice_operators_or_proba_by_selection(
        self: MyAdaptGAVar2,
        operators: NDArray,
        fitness: NDArray[np.float64],
        fitness_rank: NDArray[np.float64],
    ) -> NDArray:
        selection, _ = self._adaptation_operator
        tour_size = self._adaptation_tour_size

        quantity = len(operators)
        selected_id = selection(fitness, fitness_rank, np.int64(tour_size), np.int64(quantity))
        new_operators = operators[selected_id]
        return new_operators

    def _update_data(self: MyAdaptGAVar2) -> None:
        super()._update_data()

        keys, values = np.unique(self._selection_operators, return_counts=True)
        selection_used = dict(zip(keys, values))
        selection_mean_fitness = {}

        for key in self._selection_set.keys():
            selection_mean_fitness[key] = np.mean(self._fitness_i[self._selection_operators == key])
            if key not in selection_used:
                selection_used[key] = 0

        keys, values = np.unique(self._crossover_operators, return_counts=True)
        crossover_used = dict(zip(keys, values))
        crossover_mean_fitness = {}

        for key in self._crossover_set.keys():
            crossover_mean_fitness[key] = np.mean(self._fitness_i[self._crossover_operators == key])
            if key not in crossover_used:
                crossover_used[key] = 0

        self._update_stats(
            s_mean_fit = selection_mean_fitness,
            c_mean_fit = crossover_mean_fitness,
            s_used=selection_used,
            c_used=crossover_used,
            m_probas=self._mutation_probas,
        )

    def _adapt(self: MyAdaptGAVar2) -> None:
        if self.i % 1000 == 0:
            proba = 1.0
        else:
            proba = self._mutate_thresholds_proba
            

        self._selection_operators = self._mutate_operators(
                self._choice_operators_or_proba_by_selection(
                    self._selection_operators, self._fitness_i, self._fitness_rank_i
                ),
                list(self._selection_set.keys()), proba
            )

        self._crossover_operators = self._mutate_operators(
                self._choice_operators_or_proba_by_selection(
                    self._crossover_operators, self._fitness_i, self._fitness_rank_i
                ),
                list(self._crossover_set.keys()), proba
            )

        self._mutation_probas = self._mutate_probas(
                self._choice_operators_or_proba_by_selection(
                    operators=self._mutation_probas,
                    fitness=self._fitness_i,
                    fitness_rank=self._fitness_rank_i
                ), proba
            )

    def _get_new_population(self: MyAdaptGAVar2) -> None:
        self._population_g_i = np.array(
            [
                self._get_new_individ_g(
                    self._selection_operators[i],
                    self._crossover_operators[i],
                    self._mutation_probas[i],
                )
                for i in range(self._pop_size)
            ],
            dtype=self._population_g_i.dtype,
        )
        self.i = self.i + 1

    def _get_new_individ_g(
        self: GeneticAlgorithm,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation_proba: np.float64,
    ) -> Union[Tree, np.byte]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]

        selected_id = selection_func(
            self._fitness_scale_i, self._fitness_rank_i, np.int64(tour_size), np.int64(quantity)
        )

        offspring_no_mutated = crossover_func(
            self._population_g_i[selected_id],
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
        )

        offspring = flip_mutation(offspring_no_mutated, np.float64(specified_mutation_proba))
        return offspring
