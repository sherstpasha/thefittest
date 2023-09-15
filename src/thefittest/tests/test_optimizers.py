import numpy as np

from ..base import EphemeralNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import UniversalSet
from ..base._ea import TheFittest
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import JADE
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SaDE2005
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..tools.metrics import coefficient_determination
from ..tools.operators import Add
from ..tools.operators import Div
from ..tools.operators import Mul
from ..tools.operators import Neg
from ..tools.operators import best_1
from ..tools.operators import current_to_best_1
from ..tools.operators import growing_mutation
from ..tools.operators import one_point_crossover
from ..tools.operators import one_point_crossoverGP
from ..tools.operators import point_mutation
from ..tools.operators import proportional_selection
from ..tools.operators import rand_1
from ..tools.operators import rank_selection
from ..tools.operators import tournament_selection
from ..tools.operators import uniform_crossover
from ..tools.operators import uniform_crossoverGP
from ..tools.random import binary_string_population
from ..tools.random import float_population
from ..tools.random import half_and_half


# def test_GeneticAlgorithm_start_settings():
#     def fitness_function(x):
#         return np.sum(x, axis=1, dtype=np.float64)

#     iters = 100
#     pop_size = 50
#     str_len = 200

#     # simple start
#     optimizer = GeneticAlgorithm(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=True,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()

#     fittest = optimizer.get_fittest()
#     assert isinstance(fittest, TheFittest)

#     stats = optimizer.get_stats()

#     assert optimizer.get_remains_calls() == 0
#     assert len(stats["fitness_max"]) == iters
#     for i in range(len(stats["fitness_max"][:-1])):
#         assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
#     assert optimizer._sign == -1

#     # start with the no_increase_num is equal 15
#     def fitness_function(x):
#         return np.ones(len(x), dtype=np.float64)

#     no_increase_num = 15
#     optimizer = GeneticAlgorithm(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=no_increase_num,
#         minimization=False,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()

#     assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
#     assert optimizer._sign == 1

#     # start with the optimal_value is equal 1
#     optimizer = GeneticAlgorithm(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=1,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=False,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()
#     assert optimizer.get_remains_calls() == pop_size * (iters - 1)


# def test_GeneticAlgorithm_set_strategy():
#     def fitness_function(x):
#         return np.sum(x, axis=1, dtype=np.float64)

#     iters = 10
#     pop_size = 10
#     str_len = 5

#     optimizer = GeneticAlgorithm(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=False,
#         show_progress_each=None,
#         keep_history=True,
#     )

#     initial_population = binary_string_population(pop_size=pop_size, str_len=str_len)
#     optimizer.set_strategy(
#         selection_oper="proportional",
#         crossover_oper="uniform2",
#         mutation_oper="weak",
#         tour_size_param=6,
#         initial_population=initial_population,
#         elitism_param=True,
#         parents_num_param=7,
#         mutation_rate_param=0.1,
#     )

#     assert optimizer._specified_selection == (proportional_selection, 0)
#     assert optimizer._specified_crossover == (uniform_crossover, 2)
#     assert optimizer._specified_mutation[1]() == 1 / (3 * str_len)

#     optimizer.set_strategy(
#         selection_oper="tournament_k",
#         crossover_oper="uniformk",
#         mutation_oper="custom_rate",
#         tour_size_param=6,
#         initial_population=initial_population,
#         elitism_param=True,
#         parents_num_param=7,
#         mutation_rate_param=0.1,
#     )

#     assert optimizer._specified_selection == (tournament_selection, optimizer._tour_size)
#     assert optimizer._specified_crossover == (uniform_crossover, optimizer._parents_num)
#     assert optimizer._specified_mutation[1] == 0.1

#     optimizer.fit()

#     stats = optimizer.get_stats()

#     assert np.all(stats["population_g"][0] == initial_population)


def test_DifferentialEvolution_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    # simple start
    optimizer = DifferentialEvolution(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, TheFittest)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["fitness_max"]) == iters
    for i in range(len(stats["fitness_max"][:-1])):
        assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = DifferentialEvolution(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=no_increase_num,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = DifferentialEvolution(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_DifferentialEvolution_set_strategy():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 10
    pop_size = 10
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    optimizer = DifferentialEvolution(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    initial_population = float_population(pop_size=pop_size, left=left, right=right)

    optimizer.set_strategy(
        mutation_oper="current_to_best_1",
        F_param=0.35,
        CR_param=0.7,
        elitism_param=False,
        initial_population=initial_population,
    )

    assert optimizer._specified_mutation == current_to_best_1
    assert optimizer._F == 0.35
    assert optimizer._CR == 0.7
    assert optimizer._elitism is False

    optimizer.set_strategy(
        mutation_oper="best_1",
        F_param=0.11,
        CR_param=0.3,
        elitism_param=True,
        initial_population=initial_population,
    )

    assert optimizer._specified_mutation == best_1
    assert optimizer._F == 0.11
    assert optimizer._CR == 0.3
    assert optimizer._elitism is True

    optimizer.fit()

    stats = optimizer.get_stats()

    assert np.all(stats["population_g"][0] == initial_population)


def test_JADE_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    # simple start
    optimizer = JADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, TheFittest)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["fitness_max"]) == iters
    for i in range(len(stats["fitness_max"][:-1])):
        assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = JADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=no_increase_num,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = JADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_JADE_set_strategy():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 10
    pop_size = 10
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    optimizer = JADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    initial_population = float_population(pop_size=pop_size, left=left, right=right)

    optimizer.set_strategy(
        c_param=0.2, p_param=0.1, elitism_param=False, initial_population=initial_population
    )

    assert optimizer._c == 0.2
    assert optimizer._p == 0.1
    assert optimizer._elitism is False

    optimizer.set_strategy(
        c_param=0.33, p_param=0.22, elitism_param=True, initial_population=initial_population
    )

    assert optimizer._c == 0.33
    assert optimizer._p == 0.22
    assert optimizer._elitism is True

    optimizer.fit()

    stats = optimizer.get_stats()

    assert np.all(stats["population_g"][0] == initial_population)


def test_jDE_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    # simple start
    optimizer = jDE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, TheFittest)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["fitness_max"]) == iters
    for i in range(len(stats["fitness_max"][:-1])):
        assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = jDE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=no_increase_num,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = jDE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_jDE_set_strategy():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 10
    pop_size = 10
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    optimizer = jDE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    initial_population = float_population(pop_size=pop_size, left=left, right=right)

    optimizer.set_strategy(
        mutation_oper="rand_1",
        F_left_param=0.22,
        F_right_param=0.33,
        t_f_param=0.44,
        t_cr_param=0.55,
        elitism_param=False,
        initial_population=initial_population,
    )

    assert optimizer._specified_mutation == rand_1
    assert optimizer._F_left == 0.22
    assert optimizer._F_right == 0.33
    assert optimizer._t_f == 0.44
    assert optimizer._t_cr == 0.55
    assert optimizer._elitism is False

    optimizer.set_strategy(
        mutation_oper="best_1",
        F_left_param=0.222,
        F_right_param=0.333,
        t_f_param=0.444,
        t_cr_param=0.555,
        elitism_param=True,
        initial_population=initial_population,
    )

    assert optimizer._specified_mutation == best_1
    assert optimizer._F_left == 0.222
    assert optimizer._F_right == 0.333
    assert optimizer._t_f == 0.444
    assert optimizer._t_cr == 0.555
    assert optimizer._elitism is True

    optimizer.fit()

    stats = optimizer.get_stats()

    assert np.all(stats["population_g"][0] == initial_population)


def test_SaDE2005_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    # simple start
    optimizer = SaDE2005(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, TheFittest)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["fitness_max"]) == iters
    for i in range(len(stats["fitness_max"][:-1])):
        assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = SaDE2005(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=no_increase_num,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = SaDE2005(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_SaDE2005_set_strategy():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 100
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    optimizer = SaDE2005(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    initial_population = float_population(pop_size=pop_size, left=left, right=right)

    optimizer.set_strategy(
        m_learning_period_param=15,
        CR_update_timer_param=5,
        CR_m_learning_period_param=25,
        threshold_params=0.11,
        Fm_param=0.55,
        F_sigma_param=0.33,
        CR_sigma_param=0.12,
        elitism_param=False,
        initial_population=initial_population,
    )

    assert optimizer._m_learning_period == 15
    assert optimizer._CR_update_timer == 5
    assert optimizer._CR_m_learning_period == 25
    assert optimizer._threshold == 0.11
    assert optimizer._Fm == 0.55
    assert optimizer._F_sigma == 0.33
    assert optimizer._CR_sigma == 0.12
    assert optimizer._elitism is False

    optimizer.set_strategy(
        m_learning_period_param=16,
        CR_update_timer_param=6,
        CR_m_learning_period_param=26,
        threshold_params=0.111,
        Fm_param=0.555,
        F_sigma_param=0.333,
        CR_sigma_param=0.122,
        elitism_param=True,
        initial_population=initial_population,
    )

    assert optimizer._m_learning_period == 16
    assert optimizer._CR_update_timer == 6
    assert optimizer._CR_m_learning_period == 26
    assert optimizer._threshold == 0.111
    assert optimizer._Fm == 0.555
    assert optimizer._F_sigma == 0.333
    assert optimizer._CR_sigma == 0.122
    assert optimizer._elitism is True
    optimizer.fit()

    stats = optimizer.get_stats()

    assert np.all(stats["population_g"][0] == initial_population)


# def test_SelfCGA_start_settings():
#     def fitness_function(x):
#         return np.sum(x, axis=1, dtype=np.float64)

#     iters = 100
#     pop_size = 50
#     str_len = 200

#     # simple start
#     optimizer = SelfCGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=True,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()

#     fittest = optimizer.get_fittest()
#     assert isinstance(fittest, TheFittest)

#     stats = optimizer.get_stats()

#     assert optimizer.get_remains_calls() == 0
#     assert len(stats["fitness_max"]) == iters
#     for i in range(len(stats["fitness_max"][:-1])):
#         assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
#     assert optimizer._sign == -1

#     # start with the no_increase_num is equal 15
#     def fitness_function(x):
#         return np.ones(len(x), dtype=np.float64)

#     no_increase_num = 15
#     optimizer = SelfCGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=no_increase_num,
#         minimization=False,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()

#     assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
#     assert optimizer._sign == 1

#     # start with the optimal_value is equal 1
#     optimizer = SelfCGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=1,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=False,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()
#     assert optimizer.get_remains_calls() == pop_size * (iters - 1)


# def test_SelfCGA_set_strategy():
#     def fitness_function(x):
#         return np.sum(x, axis=1, dtype=np.float64)

#     iters = 10
#     pop_size = 10
#     str_len = 5

#     optimizer = SelfCGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=False,
#         show_progress_each=None,
#         keep_history=True,
#     )

#     initial_population = binary_string_population(pop_size=pop_size, str_len=str_len)
#     optimizer.set_strategy(
#         selection_opers=("proportional", "tournament_3"),
#         crossover_opers=("uniform2", "one_point"),
#         mutation_opers=("weak", "average"),
#         tour_size_param=6,
#         initial_population=initial_population,
#         elitism_param=False,
#         parents_num_param=7,
#         mutation_rate_param=0.1,
#         K_param=0.2,
#         threshold_param=0.001,
#     )

#     selection_set = {
#         "proportional": (proportional_selection, 0),
#         "tournament_3": (tournament_selection, 3),
#     }
#     crossover_set = {"uniform2": (uniform_crossover, 2), "one_point": (one_point_crossover, 2)}
#     assert optimizer._selection_set == selection_set
#     assert optimizer._crossover_set == crossover_set
#     assert optimizer._mutation_set["weak"][1]() == 1 / (3 * str_len)
#     assert optimizer._mutation_set["average"][1]() == 1 / (str_len)
#     assert optimizer._tour_size == 6
#     assert optimizer._elitism is False
#     assert optimizer._parents_num == 7
#     assert optimizer._K == 0.2
#     assert optimizer._threshold == 0.001

#     optimizer.set_strategy(
#         selection_opers=("rank", "tournament_k"),
#         crossover_opers=("uniform7", "uniformk"),
#         mutation_opers=("weak", "custom_rate"),
#         tour_size_param=4,
#         initial_population=initial_population,
#         elitism_param=True,
#         parents_num_param=5,
#         mutation_rate_param=0.12,
#         K_param=0.3,
#         threshold_param=0.0011,
#     )

#     selection_set = {"rank": (rank_selection, 0), "tournament_k": (tournament_selection, 4)}
#     crossover_set = {"uniform7": (uniform_crossover, 7), "uniformk": (uniform_crossover, 5)}
#     assert optimizer._selection_set == selection_set
#     assert optimizer._crossover_set == crossover_set
#     assert optimizer._mutation_set["weak"][1]() == 1 / (3 * str_len)
#     assert optimizer._mutation_set["custom_rate"][1] == 0.12
#     assert optimizer._tour_size == 4
#     assert optimizer._elitism is True
#     assert optimizer._parents_num == 5
#     assert optimizer._K == 0.3
#     assert optimizer._threshold == 0.0011

#     optimizer.fit()

#     stats = optimizer.get_stats()

#     assert np.all(stats["population_g"][0] == initial_population)


def test_SHADE_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    # simple start
    optimizer = SHADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, TheFittest)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["fitness_max"]) == iters
    for i in range(len(stats["fitness_max"][:-1])):
        assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = SHADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=no_increase_num,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = SHADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_SHADE_set_strategy():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 10
    pop_size = 10
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    optimizer = SHADE(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        left=left,
        right=right,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    initial_population = float_population(pop_size=pop_size, left=left, right=right)

    optimizer.set_strategy(elitism_param=False, initial_population=initial_population)

    assert optimizer._elitism is False

    optimizer.set_strategy(elitism_param=True, initial_population=initial_population)

    assert optimizer._elitism is True

    optimizer.fit()

    stats = optimizer.get_stats()

    assert np.all(stats["population_g"][0] == initial_population)


# def test_SHAGA_start_settings():
#     def fitness_function(x):
#         return np.sum(x, axis=1, dtype=np.float64)

#     iters = 100
#     pop_size = 50
#     str_len = 200

#     # simple start
#     optimizer = SHAGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=True,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()

#     fittest = optimizer.get_fittest()
#     assert isinstance(fittest, TheFittest)

#     stats = optimizer.get_stats()

#     assert optimizer.get_remains_calls() == 0
#     assert len(stats["fitness_max"]) == iters
#     for i in range(len(stats["fitness_max"][:-1])):
#         assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
#     assert optimizer._sign == -1

#     # start with the no_increase_num is equal 15
#     def fitness_function(x):
#         return np.ones(len(x), dtype=np.float64)

#     no_increase_num = 15
#     optimizer = SHAGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=no_increase_num,
#         minimization=False,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()

#     assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
#     assert optimizer._sign == 1

#     # start with the optimal_value is equal 1
#     optimizer = SHAGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=1,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=False,
#         show_progress_each=1,
#         keep_history=True,
#     )

#     optimizer.fit()
#     assert optimizer.get_remains_calls() == pop_size * (iters - 1)


# def test_SHAGA_set_strategy():
#     def fitness_function(x):
#         return np.sum(x, axis=1, dtype=np.float64)

#     iters = 10
#     pop_size = 10
#     str_len = 5

#     optimizer = SHAGA(
#         fitness_function,
#         iters=iters,
#         pop_size=pop_size,
#         str_len=str_len,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         minimization=False,
#         show_progress_each=None,
#         keep_history=True,
#     )

#     initial_population = binary_string_population(pop_size=pop_size, str_len=str_len)
#     optimizer.set_strategy(initial_population=initial_population, elitism_param=False)

#     assert optimizer._elitism is False

#     optimizer.set_strategy(initial_population=initial_population, elitism_param=True)

#     assert optimizer._elitism is True

#     optimizer.fit()

#     stats = optimizer.get_stats()

#     assert np.all(stats["population_g"][0] == initial_population)


# def test_GeneticProgramming_start_settings():
#     def generator1():
#         return np.round(np.random.uniform(0, 10), 4)

#     def generator2():
#         return np.random.randint(0, 10)

#     def problem(x):
#         return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

#     function = problem
#     left_border = -4.5
#     right_border = 4.5
#     sample_size = 300
#     n_dimension = 1

#     iters = 20
#     pop_size = 15

#     X = np.array(
#         [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
#     ).T
#     y = function(X)

#     functional_set = (
#         FunctionalNode(Add()),
#         FunctionalNode(Mul()),
#         FunctionalNode(Neg()),
#         FunctionalNode(Div()),
#     )

#     terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
#     terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
#     uniset = UniversalSet(functional_set, terminal_set)

#     def fitness_function(trees):
#         fitness = []
#         for tree in trees:
#             y_pred = tree() * np.ones(len(y))
#             fitness.append(-coefficient_determination(y, y_pred))
#         return np.array(fitness)

#     # simple start
#     optimizer = GeneticProgramming(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         show_progress_each=1,
#         minimization=True,
#         keep_history=True,
#     )

#     optimizer.fit()

#     fittest = optimizer.get_fittest()
#     assert isinstance(fittest, TheFittest)

#     stats = optimizer.get_stats()

#     assert optimizer.get_remains_calls() == 0
#     assert len(stats["fitness_max"]) == iters
#     for i in range(len(stats["fitness_max"][:-1])):
#         assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
#     assert optimizer._sign == -1

#     # start with the no_increase_num is equal 15
#     def fitness_function(x):
#         return np.ones(len(x), dtype=np.float64)

#     no_increase_num = 15

#     optimizer = GeneticProgramming(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=no_increase_num,
#         show_progress_each=1,
#         minimization=False,
#         keep_history=True,
#     )

#     optimizer.fit()

#     assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
#     assert optimizer._sign == 1

#     # start with the optimal_value is equal 1
#     optimizer = GeneticProgramming(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=1,
#         termination_error_value=0,
#         no_increase_num=None,
#         show_progress_each=1,
#         minimization=False,
#         keep_history=True,
#     )

#     optimizer.fit()
#     assert optimizer.get_remains_calls() == pop_size * (iters - 1)


# def test_GeneticProgramming_set_strategy():
#     def generator1():
#         return np.round(np.random.uniform(0, 10), 4)

#     def generator2():
#         return np.random.randint(0, 10)

#     def problem(x):
#         return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

#     function = problem
#     left_border = -4.5
#     right_border = 4.5
#     sample_size = 300
#     n_dimension = 1

#     iters = 20
#     pop_size = 15

#     X = np.array(
#         [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
#     ).T
#     y = function(X)

#     functional_set = (
#         FunctionalNode(Add()),
#         FunctionalNode(Mul()),
#         FunctionalNode(Neg()),
#         FunctionalNode(Div()),
#     )

#     terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
#     terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
#     uniset = UniversalSet(functional_set, terminal_set)

#     def fitness_function(trees):
#         fitness = []
#         for tree in trees:
#             y_pred = tree() * np.ones(len(y))
#             fitness.append(-coefficient_determination(y, y_pred))
#         return np.array(fitness)

#     # simple start
#     optimizer = GeneticProgramming(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         show_progress_each=1,
#         minimization=True,
#         keep_history=True,
#     )

#     initial_population = half_and_half(pop_size=pop_size, uniset=uniset, max_level=14)

#     optimizer.set_strategy(
#         selection_oper="rank",
#         crossover_oper="one_point",
#         mutation_oper="average_grow",
#         initial_population=initial_population,
#         max_level_param=14,
#         elitism_param=False,
#         init_level_param=3,
#         mutation_rate_param=0.22,
#     )

#     assert optimizer._specified_selection == (rank_selection, 0)
#     assert optimizer._specified_crossover == (one_point_crossoverGP, 2)
#     assert optimizer._specified_mutation == (growing_mutation, 1, True)
#     assert optimizer._max_level == 14
#     assert optimizer._init_level == 3
#     assert optimizer._mutation_rate == 0.22
#     assert optimizer._elitism is False
#     assert np.all(optimizer._initial_population == initial_population)

#     optimizer.set_strategy(
#         selection_oper="tournament_k",
#         crossover_oper="uniformk",
#         mutation_oper="custom_rate_point",
#         initial_population=initial_population,
#         max_level_param=13,
#         elitism_param=True,
#         init_level_param=2,
#         parents_num_param=8,
#         tour_size_param=6,
#         mutation_rate_param=0.223,
#     )

#     assert optimizer._specified_selection == (tournament_selection, 6)
#     assert optimizer._specified_crossover == (uniform_crossoverGP, 8)
#     assert optimizer._specified_mutation == (point_mutation, 0.223, False)
#     assert optimizer._max_level == 13
#     assert optimizer._init_level == 2
#     assert optimizer._mutation_rate == 0.223
#     assert optimizer._elitism is True
#     assert np.all(optimizer._initial_population == initial_population)

#     optimizer.fit()

#     stats = optimizer.get_stats()

#     assert np.all(stats["population_g"][0] == initial_population)


# def test_SelfCGP_start_settings():
#     def generator1():
#         return np.round(np.random.uniform(0, 10), 4)

#     def generator2():
#         return np.random.randint(0, 10)

#     def problem(x):
#         return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

#     function = problem
#     left_border = -4.5
#     right_border = 4.5
#     sample_size = 300
#     n_dimension = 1

#     iters = 20
#     pop_size = 15

#     X = np.array(
#         [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
#     ).T
#     y = function(X)

#     functional_set = (
#         FunctionalNode(Add()),
#         FunctionalNode(Mul()),
#         FunctionalNode(Neg()),
#         FunctionalNode(Div()),
#     )

#     terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
#     terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
#     uniset = UniversalSet(functional_set, terminal_set)

#     def fitness_function(trees):
#         fitness = []
#         for tree in trees:
#             y_pred = tree() * np.ones(len(y))
#             fitness.append(-coefficient_determination(y, y_pred))
#         return np.array(fitness)

#     # simple start
#     optimizer = SelfCGP(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         show_progress_each=1,
#         minimization=True,
#         keep_history=True,
#     )

#     optimizer.fit()

#     fittest = optimizer.get_fittest()
#     assert isinstance(fittest, TheFittest)

#     stats = optimizer.get_stats()

#     assert optimizer.get_remains_calls() == 0
#     assert len(stats["fitness_max"]) == iters
#     for i in range(len(stats["fitness_max"][:-1])):
#         assert stats["fitness_max"][i] <= stats["fitness_max"][i + 1]
#     assert optimizer._sign == -1

#     # start with the no_increase_num is equal 15
#     def fitness_function(x):
#         return np.ones(len(x), dtype=np.float64)

#     no_increase_num = 15

#     optimizer = SelfCGP(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=no_increase_num,
#         show_progress_each=1,
#         minimization=False,
#         keep_history=True,
#     )

#     optimizer.fit()

#     assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
#     assert optimizer._sign == 1

#     # start with the optimal_value is equal 1
#     optimizer = SelfCGP(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=1,
#         termination_error_value=0,
#         no_increase_num=None,
#         show_progress_each=1,
#         minimization=False,
#         keep_history=True,
#     )

#     optimizer.fit()
#     assert optimizer.get_remains_calls() == pop_size * (iters - 1)


# def test_SelfCGP_set_strategy():
#     def generator1():
#         return np.round(np.random.uniform(0, 10), 4)

#     def generator2():
#         return np.random.randint(0, 10)

#     def problem(x):
#         return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

#     function = problem
#     left_border = -4.5
#     right_border = 4.5
#     sample_size = 300
#     n_dimension = 1

#     iters = 20
#     pop_size = 15

#     X = np.array(
#         [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
#     ).T
#     y = function(X)

#     functional_set = (
#         FunctionalNode(Add()),
#         FunctionalNode(Mul()),
#         FunctionalNode(Neg()),
#         FunctionalNode(Div()),
#     )

#     terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
#     terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
#     uniset = UniversalSet(functional_set, terminal_set)

#     def fitness_function(trees):
#         fitness = []
#         for tree in trees:
#             y_pred = tree() * np.ones(len(y))
#             fitness.append(-coefficient_determination(y, y_pred))
#         return np.array(fitness)

#     # simple start
#     optimizer = SelfCGP(
#         fitness_function=fitness_function,
#         uniset=uniset,
#         pop_size=pop_size,
#         iters=iters,
#         optimal_value=None,
#         termination_error_value=0,
#         no_increase_num=None,
#         show_progress_each=1,
#         minimization=True,
#         keep_history=True,
#     )

#     initial_population = half_and_half(pop_size=pop_size, uniset=uniset, max_level=14)

#     optimizer.set_strategy(
#         selection_opers=("tournament_3", "tournament_k"),
#         crossover_opers=("one_point", "uniformk"),
#         mutation_opers=("average_grow", "weak_point"),
#         tour_size_param=5,
#         initial_population=initial_population,
#         elitism_param=False,
#         parents_num_param=4,
#         mutation_rate_param=0.1,
#         threshold_param=0.02,
#         max_level_param=13,
#         init_level_param=3,
#     )

#     selection_set = {
#         "tournament_3": (tournament_selection, 3),
#         "tournament_k": (tournament_selection, 5),
#     }
#     crossover_set = {"one_point": (one_point_crossoverGP, 2), "uniformk": (uniform_crossoverGP, 4)}
#     mutation_set = {
#         "average_grow": (growing_mutation, 1, True),
#         "weak_point": (point_mutation, 0.25, True),
#     }

#     assert optimizer._selection_set == selection_set
#     assert optimizer._crossover_set == crossover_set
#     assert optimizer._mutation_set == mutation_set
#     assert optimizer._tour_size == 5
#     assert np.all(optimizer._initial_population == initial_population)
#     assert optimizer._elitism is False
#     assert optimizer._parents_num == 4
#     assert optimizer._mutation_rate == 0.1
#     assert optimizer._threshold == 0.02
#     assert optimizer._max_level == 13
#     assert optimizer._init_level == 3

#     optimizer.set_strategy(
#         selection_opers=("tournament_5", "rank"),
#         crossover_opers=("uniform2",),
#         mutation_opers=("weak_grow", "custom_rate_point"),
#         tour_size_param=4,
#         initial_population=initial_population,
#         elitism_param=True,
#         parents_num_param=9,
#         mutation_rate_param=0.11,
#         threshold_param=0.022,
#         max_level_param=11,
#         init_level_param=9,
#     )

#     selection_set = {"tournament_5": (tournament_selection, 5), "rank": (rank_selection, 0)}
#     crossover_set = {"uniform2": (uniform_crossoverGP, 2)}
#     mutation_set = {
#         "weak_grow": (growing_mutation, 0.25, True),
#         "custom_rate_point": (point_mutation, 0.11, False),
#     }

#     assert optimizer._selection_set == selection_set
#     assert optimizer._crossover_set == crossover_set
#     assert optimizer._mutation_set == mutation_set
#     assert optimizer._tour_size == 4
#     assert np.all(optimizer._initial_population == initial_population)
#     assert optimizer._elitism is True
#     assert optimizer._parents_num == 9
#     assert optimizer._mutation_rate == 0.11
#     assert optimizer._threshold == 0.022
#     assert optimizer._max_level == 11
#     assert optimizer._init_level == 9

#     optimizer.fit()

#     stats = optimizer.get_stats()

#     assert np.all(stats["population_g"][0] == initial_population)
