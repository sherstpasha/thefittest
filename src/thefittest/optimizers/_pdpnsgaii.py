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

from ._selfcnsgaii import SelfCNSGAII
from ..utils import numpy_group_by
from ..utils.random import random_sample
from ..utils.random import flip_coin


class PDPNSGAII(SelfCNSGAII):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        str_len: int,
        minimization: Tuple[bool, ...],
        mutation_rate: float = 0.05,
        parents_num: int = 2,
        selections: Tuple[str, ...] = ("rank_crowding", "dominance_crowding"),
        crossovers: Tuple[str, ...] = (
            "empty",
            "one_point",
            "two_point",
            "uniform_2",
            "uniform_7",
            "uniform_tour_3",
            "uniform_tour_7",
        ),
        mutations: Tuple[str, ...] = ("weak", "average", "strong"),
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
        use_fitness_cache: bool = False,
        fitness_cache_size: Optional[int] = 1000,
    ) -> None:
        for kind, names in (
            ("selection", selections),
            ("crossover", crossovers),
            ("mutation", mutations),
        ):
            if not names:
                raise ValueError(f"{kind} operators must be non-empty and unique.")
        super().__init__(
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            minimization=minimization,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
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
        self._previous_fitness_i: List[NDArray[np.float64]] = []
        self._success_i: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)

    def _choice_parent(self, fitness_i_selected: NDArray[np.float64]) -> NDArray[np.float64]:
        index = random_sample(len(fitness_i_selected), 1, True)[0]
        return fitness_i_selected[index].copy()

    def _culc_r_i(
        self,
        success: NDArray[np.bool_],
        operator: NDArray[np.str_],
        operator_keys: List[str],
    ) -> Tuple[NDArray[np.str_], NDArray[np.float64]]:
        keys, groups = numpy_group_by(group=success, by=operator)
        r_values = np.array(
            [(np.sum(group) ** 2 + 1) / (len(group) + 1) for group in groups],
            dtype=np.float64,
        )
        for key in operator_keys:
            if key not in keys:
                keys = np.append(keys, key)
                r_values = np.append(r_values, 0.0)
        return keys, r_values

    def _get_new_proba_pdp(
        self,
        proba_dict: Dict[str, float],
        operators: NDArray[np.str_],
        threshold: float,
    ) -> Dict[str, float]:
        keys, r_values = self._culc_r_i(self._success_i, operators, list(proba_dict))
        probabilities = threshold + r_values * (
            (1.0 - len(proba_dict) * threshold) / r_values.sum()
        )
        return dict(sorted(zip(keys, probabilities)))

    def _adapt(self) -> None:
        self._selection_proba = self._get_new_proba_pdp(
            self._selection_proba, self._selection_operators, self._thresholds["selection"]
        )
        self._crossover_proba = self._get_new_proba_pdp(
            self._crossover_proba, self._crossover_operators, self._thresholds["crossover"]
        )
        self._mutation_proba = self._get_new_proba_pdp(
            self._mutation_proba, self._mutation_operators, self._thresholds["mutation"]
        )

    def _get_new_individ_g(
        self,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
    ) -> NDArray[np.byte]:
        if specified_selection == "rank_crowding":
            selection_data = self._pareto_rank_i
        elif specified_selection == "dominance_crowding":
            selection_data = self._fitness_i
        else:
            raise ValueError(f"Unsupported selection: {specified_selection}.")

        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func, proba, is_constant_rate = self._mutation_pool[specified_mutation]

        selected_id = selection_func(selection_data, self._crowding_distance_i, tour_size, quantity)
        previous_fitness = self._choice_parent(self._fitness_i[selected_id])
        self._previous_fitness_i.append(previous_fitness)

        if self._crossover_rate == 1.0 or flip_coin(self._crossover_rate):
            ranks = self._pareto_rank_i[selected_id]
            if specified_crossover not in ("uniform_tour_3", "uniform_tour_7"):
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
        return mutation_func(offspring_no_mutated, np.float64(proba))

    def _replace_population(
        self,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
        ranks: NDArray[np.int64],
        distances: NDArray[np.float64],
        survivors: NDArray[np.int64],
    ) -> None:
        parents_fitness = np.asarray(self._previous_fitness_i, dtype=np.float64)
        offspring_fitness = fitness[self._pop_size :]
        if parents_fitness.shape != offspring_fitness.shape:
            raise ValueError("Each offspring must have one saved parent fitness vector.")
        self._success_i = np.all(offspring_fitness >= parents_fitness, axis=1) & np.any(
            offspring_fitness > parents_fitness, axis=1
        )
        super()._replace_population(
            population_g, population_ph, fitness, ranks, distances, survivors
        )
        self._previous_fitness_i.clear()
