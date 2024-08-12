from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._geneticalgorithm import GeneticAlgorithm
from ..tools import donothing
from ..tools.transformations import numpy_group_by

import torch
import torch.nn as nn
import torch.optim as optim


# Задаем нейронную сеть для RL агента
def create_policy_network(input_size, output_sizes):
    class PolicyNetwork(nn.Module):
        def __init__(self, input_size, output_sizes):
            super(PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, 5)

            # Три выходных слоя для трех типов операторов
            self.fc2_crossover = nn.Linear(5, output_sizes[0])
            self.fc2_mutation = nn.Linear(5, output_sizes[1])
            self.fc2_selection = nn.Linear(5, output_sizes[2])

            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))

            # Три выхода, по одному на каждый тип операторов
            crossover_probs = self.softmax(self.fc2_crossover(x))
            mutation_probs = self.softmax(self.fc2_mutation(x))
            selection_probs = self.softmax(self.fc2_selection(x))

            return crossover_probs, mutation_probs, selection_probs

    return PolicyNetwork(input_size, output_sizes)


def create_optimizer(network, learning_rate=0.01):
    return optim.Adam(network.parameters(), lr=learning_rate)


def process_population_state(fitness_vector, population):
    # Средний фитнесс популяции
    avg_fitness = np.mean(fitness_vector)

    # Стандартное отклонение фитнесс-функции
    std_fitness = np.std(fitness_vector)

    # Разнообразие популяции (например, на основе уникальных битовых строк)
    unique_individuals = np.unique(population, axis=0)
    diversity = len(unique_individuals) / len(population)

    # Формирование вектора состояния для нейросети
    state = torch.tensor([[avg_fitness, std_fitness, diversity]], dtype=torch.float32)

    return state, avg_fitness


def get_operator_probabilities(state, network):
    # Получаем предсказания вероятностей для каждого типа операторов
    selection_probs, crossover_probs, mutation_probs = network(state)

    # Преобразуем их в numpy массивы для удобства
    selection_probs = selection_probs.detach().numpy().flatten()
    crossover_probs = crossover_probs.detach().numpy().flatten()
    mutation_probs = mutation_probs.detach().numpy().flatten()

    # Нормализуем вероятности, чтобы сумма была равна 1, независимо от небольших отклонений
    selection_probs /= np.sum(selection_probs)
    crossover_probs /= np.sum(crossover_probs)
    mutation_probs /= np.sum(mutation_probs)

    # В случае небольших отклонений, подгоняем суммы к 1
    selection_probs *= 1 / np.sum(selection_probs)
    crossover_probs *= 1 / np.sum(crossover_probs)
    mutation_probs *= 1 / np.sum(mutation_probs)

    # Возвращаем вероятности для всех типов операторов в заданном порядке
    return selection_probs, crossover_probs, mutation_probs


def train_policy_network(optimizer, action_dist, action_index, reward):
    optimizer.zero_grad()
    loss = -action_dist.log_prob(torch.tensor(action_index)) * reward
    loss.backward()
    optimizer.step()


class SelfCGANet(GeneticAlgorithm):
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
        mutations: Tuple[str, ...] = ("weak", "average", "strong"),
        init_population: Optional[NDArray[np.byte]] = None,
        K: float = 2,
        selection_threshold_proba: float = 0.05,
        crossover_threshold_proba: float = 0.05,
        mutation_threshold_proba: float = 0.05,
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

        self._K: float = K
        self._thresholds: Dict[str, float] = {
            "selection": selection_threshold_proba,
            "crossover": crossover_threshold_proba,
            "mutation": mutation_threshold_proba,
        }

        self._selection_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._crossover_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._mutation_set: Dict[str, Tuple[Callable, Union[float, int], bool]] = {}

        self._selection_proba: Dict[str, float]
        self._crossover_proba: Dict[str, float]
        self._mutation_proba: Dict[str, float]

        for operator_name in selections:
            self._selection_set[operator_name] = self._selection_pool[operator_name]
        self._selection_set = dict(sorted(self._selection_set.items()))

        for operator_name in crossovers:
            self._crossover_set[operator_name] = self._crossover_pool[operator_name]
        self._crossover_set = dict(sorted(self._crossover_set.items()))

        for operator_name in mutations:
            self._mutation_set[operator_name] = self._mutation_pool[operator_name]
        self._mutation_set = dict(sorted(self._mutation_set.items()))

        self._z_selection = len(self._selection_set)
        self._z_crossover = len(self._crossover_set)
        self._z_mutation = len(self._mutation_set)

        self.input_lag = 5
        self.output_window = 10
        self._selection_adapt_net = create_policy_network(
            input_size=3,
            output_sizes=[self._z_selection, self._z_crossover, self._z_mutation],
        )
        self._selection_optimizer = create_optimizer(self._selection_adapt_net, 0.01)
        self.i = 1

        self._selection_proba = dict(
            zip(list(self._selection_set.keys()), np.full(self._z_selection, 1 / self._z_selection))
        )
        if "empty" in self._crossover_set.keys():
            self._crossover_proba = dict(
                zip(
                    list(self._crossover_set.keys()),
                    np.full(self._z_crossover, 0.9 / (self._z_crossover - 1)),
                )
            )
            self._crossover_proba["empty"] = 0.1
        else:
            self._crossover_proba = dict(
                zip(
                    list(self._crossover_set.keys()),
                    np.full(self._z_crossover, 1 / self._z_crossover),
                )
            )
        self._mutation_proba = dict(
            zip(list(self._mutation_set.keys()), np.full(self._z_mutation, 1 / self._z_mutation))
        )

        self._selection_operators: NDArray = self._choice_operators(
            proba_dict=self._selection_proba
        )
        self._crossover_operators: NDArray = self._choice_operators(
            proba_dict=self._crossover_proba
        )
        self._mutation_operators: NDArray = self._choice_operators(proba_dict=self._mutation_proba)

    def _choice_operators(self: SelfCGANet, proba_dict: Dict["str", float]) -> NDArray:
        operators = list(proba_dict.keys())
        proba = list(proba_dict.values())
        print(proba)
        chosen_operator = np.random.choice(operators, self._pop_size, p=proba)
        return chosen_operator

    def _get_new_proba(
        self: SelfCGANet,
        proba_dict: Dict["str", float],
        operator: str,
        threshold: float,
    ) -> Dict["str", float]:
        proba_dict[operator] += self._K / self._iters
        proba_value = np.array(list(proba_dict.values()))
        proba_value -= self._K / (len(proba_dict) * self._iters)
        proba_value = proba_value.clip(threshold, 1)
        proba_value = proba_value / proba_value.sum()
        new_proba_dict = dict(zip(proba_dict.keys(), proba_value))
        return new_proba_dict

    def _find_fittest_operator(
        self: SelfCGANet, operators: NDArray, fitness: NDArray[np.float64]
    ) -> str:
        keys, groups = numpy_group_by(group=fitness, by=operators)
        mean_fit = np.array(list(map(np.mean, groups)))
        fittest_operator = keys[np.argmax(mean_fit)]
        return fittest_operator, mean_fit

    def _update_data(self: SelfCGANet) -> None:
        super()._update_data()
        self._update_stats(
            s_proba=self._selection_proba,
            c_proba=self._crossover_proba,
            m_proba=self._mutation_proba,
        )

    def _adapt(self: SelfCGANet) -> None:

        state, avg_fitness = process_population_state(self._fitness_i, self._population_g_i)

        selection_probs, crossover_probs, mutation_probs = get_operator_probabilities(
            state, self._selection_adapt_net
        )

        self._selection_proba = dict(zip(self._selection_proba.keys(), selection_probs))
        self._crossover_proba = dict(zip(self._crossover_proba.keys(), crossover_probs))
        self._mutation_proba = dict(zip(self._mutation_proba.keys(), mutation_probs))

        self._selection_operators = np.random.choice(
            list(self._selection_proba.keys()), self._pop_size, p=selection_probs
        )
        self._crossover_operators = np.random.choice(
            list(self._crossover_proba.keys()), self._pop_size, p=crossover_probs
        )
        self._mutation_operators = np.random.choice(
            list(self._mutation_proba.keys()), self._pop_size, p=mutation_probs
        )

    def _get_new_population(self: SelfCGANet) -> None:
        self._population_g_i = np.array(
            [
                self._get_new_individ_g(
                    self._selection_operators[i],
                    self._crossover_operators[i],
                    self._mutation_operators[i],
                )
                for i in range(self._pop_size)
            ],
            dtype=self._population_g_i.dtype,
        )
        self.i = self.i + 1
