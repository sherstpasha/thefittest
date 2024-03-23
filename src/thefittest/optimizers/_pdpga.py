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
    """Niehaus, J., Banzhaf, W. (2001). Adaption of Operator Probabilities in
    Genetic Programming. In: Miller, J., Tomassini, M., Lanzi, P.L., Ryan, C.,
    Tettamanzi, A.G.B., Langdon, W.B. (eds) Genetic Programming. EuroGP 2001.
    Lecture Notes in Computer Science, vol 2038. Springer, Berlin, Heidelberg.
    https://doi.org/10.1007/3-540-45355-5_26
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
