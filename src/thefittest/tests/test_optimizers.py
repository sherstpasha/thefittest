import numpy as np

from ..base._ea import TheFittest
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
# from ..optimizers import GeneticProgramming
from ..tools.operators import best_1
from ..tools.operators import current_to_best_1
from ..tools.operators import flip_mutation
from ..tools.operators import proportional_selection
from ..tools.operators import tournament_selection
from ..tools.operators import uniform_crossover
from ..tools.random import binary_string_population
from ..tools.random import float_population


def test_GeneticAlgorithm_start_settings():

    def fitness_function(x):
        return np.sum(x, axis=1)

    iters = 100
    pop_size = 50
    str_len = 200

    # simple start
    optimizer = GeneticAlgorithm(fitness_function,
                                 iters=iters,
                                 pop_size=pop_size,
                                 str_len=str_len,
                                 optimal_value=None,
                                 termination_error_value=0,
                                 no_increase_num=None,
                                 minimization=True,
                                 show_progress_each=1,
                                 keep_history=True)

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, TheFittest)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats['fitness_max']) == iters
    for i, fitness_max_i in enumerate(stats['fitness_max'][:-1]):
        assert stats['fitness_max'][i] <= stats['fitness_max'][i+1]
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.int64)
    no_increase_num = 15
    optimizer = GeneticAlgorithm(fitness_function,
                                 iters=iters,
                                 pop_size=pop_size,
                                 str_len=str_len,
                                 optimal_value=None,
                                 termination_error_value=0,
                                 no_increase_num=no_increase_num,
                                 minimization=False,
                                 show_progress_each=1,
                                 keep_history=True)

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size*(iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = GeneticAlgorithm(fitness_function,
                                 iters=iters,
                                 pop_size=pop_size,
                                 str_len=str_len,
                                 optimal_value=1,
                                 termination_error_value=0,
                                 no_increase_num=None,
                                 minimization=False,
                                 show_progress_each=1,
                                 keep_history=True)

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size*(iters - 1)


def test_GeneticAlgorithm_set_strategy():

    def fitness_function(x):
        return np.sum(x, axis=1)

    iters = 10
    pop_size = 10
    str_len = 5

    optimizer = GeneticAlgorithm(fitness_function,
                                 iters=iters,
                                 pop_size=pop_size,
                                 str_len=str_len,
                                 optimal_value=None,
                                 termination_error_value=0,
                                 no_increase_num=None,
                                 minimization=False,
                                 show_progress_each=None,
                                 keep_history=True)

    initial_population = binary_string_population(pop_size=pop_size,
                                                  str_len=str_len)
    optimizer.set_strategy(selection_oper='proportional',
                           crossover_oper='uniform2',
                           mutation_oper='weak',
                           tour_size_param=6,
                           initial_population=initial_population,
                           elitism_param=True,
                           parents_num_param=7,
                           mutation_rate_param=0.1)

    assert optimizer._specified_selection == (proportional_selection, 0)
    assert optimizer._specified_crossover == (uniform_crossover, 2)
    assert optimizer._specified_mutation == (flip_mutation, 1/(3*str_len))

    optimizer.set_strategy(selection_oper='tournament_k',
                           crossover_oper='uniformk',
                           mutation_oper='custom_rate',
                           tour_size_param=6,
                           initial_population=initial_population,
                           elitism_param=True,
                           parents_num_param=7,
                           mutation_rate_param=0.1)

    assert optimizer._specified_selection == (
        tournament_selection, optimizer._tour_size)
    assert optimizer._specified_crossover == (
        uniform_crossover, optimizer._parents_num)
    assert optimizer._specified_mutation == (
        flip_mutation, optimizer._mutation_rate)

    optimizer.fit()

    stats = optimizer.get_stats()

    assert np.all(stats['population_g'][0] == initial_population)


def test_DifferentialEvolution_start_settings():

    def fitness_function(x):
        return np.sum(x, axis=1)

    iters = 100
    pop_size = 50
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    # simple start
    optimizer = DifferentialEvolution(fitness_function,
                                      iters=iters,
                                      pop_size=pop_size,
                                      left=left,
                                      right=right,
                                      optimal_value=None,
                                      termination_error_value=0,
                                      no_increase_num=None,
                                      minimization=True,
                                      show_progress_each=1,
                                      keep_history=True)

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, TheFittest)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats['fitness_max']) == iters
    for i, fitness_max_i in enumerate(stats['fitness_max'][:-1]):
        assert stats['fitness_max'][i] <= stats['fitness_max'][i+1]
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.int64)
    no_increase_num = 15
    optimizer = DifferentialEvolution(fitness_function,
                                      iters=iters,
                                      pop_size=pop_size,
                                      left=left,
                                      right=right,
                                      optimal_value=None,
                                      termination_error_value=0,
                                      no_increase_num=no_increase_num,
                                      minimization=False,
                                      show_progress_each=1,
                                      keep_history=True)

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size*(iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = DifferentialEvolution(fitness_function,
                                      iters=iters,
                                      pop_size=pop_size,
                                      left=left,
                                      right=right,
                                      optimal_value=1,
                                      termination_error_value=0,
                                      no_increase_num=None,
                                      minimization=False,
                                      show_progress_each=1,
                                      keep_history=True)

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size*(iters - 1)


def test_DifferentialEvolution_set_strategy():

    def fitness_function(x):
        return np.sum(x, axis=1)

    iters = 10
    pop_size = 10
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    optimizer = DifferentialEvolution(fitness_function,
                                      iters=iters,
                                      pop_size=pop_size,
                                      left=left,
                                      right=right,
                                      optimal_value=None,
                                      termination_error_value=0,
                                      no_increase_num=None,
                                      minimization=True,
                                      show_progress_each=1,
                                      keep_history=True)

    initial_population = float_population(pop_size=pop_size, left=left, right=right)

    optimizer.set_strategy(mutation_oper='current_to_best_1',
                           F_param=0.35,
                           CR_param=0.7,
                           elitism_param=False,
                           initial_population=initial_population)

    assert optimizer._specified_mutation == current_to_best_1
    assert optimizer._F == 0.35
    assert optimizer._CR == 0.7
    assert optimizer._elitism is False

    optimizer.set_strategy(mutation_oper='best_1',
                           F_param=0.11,
                           CR_param=0.3,
                           elitism_param=True,
                           initial_population=initial_population)

    assert optimizer._specified_mutation == best_1
    assert optimizer._F == 0.11
    assert optimizer._CR == 0.3
    assert optimizer._elitism is True

    optimizer.fit()

    stats = optimizer.get_stats()

    assert np.all(stats['population_g'][0] == initial_population)


# def test_GeneticProgramming_start_settings():

#     def fitness_function(x):
#         return np.sum(x, axis=1)

#     iters = 100
#     pop_size = 50
#     str_len = 200

    # # simple start
    # optimizer = GeneticAlgorithm(fitness_function,
    #                              iters=iters,
    #                              pop_size=pop_size,
    #                              str_len=str_len,
    #                              optimal_value=None,
    #                              termination_error_value=0,
    #                              no_increase_num=None,
    #                              minimization=True,
    #                              show_progress_each=1,
    #                              keep_history=True)

    # optimizer.fit()

    # fittest = optimizer.get_fittest()
    # assert isinstance(fittest, TheFittest)

    # stats = optimizer.get_stats()

    # assert optimizer.get_remains_calls() == 0
    # assert len(stats['fitness_max']) == iters
    # for i, fitness_max_i in enumerate(stats['fitness_max'][:-1]):
    #     assert stats['fitness_max'][i] <= stats['fitness_max'][i+1]
    # assert optimizer._sign == -1

    # # start with the no_increase_num is equal 15
    # def fitness_function(x):
    #     return np.ones(len(x), dtype=np.int64)
    # no_increase_num = 15
    # optimizer = GeneticAlgorithm(fitness_function,
    #                              iters=iters,
    #                              pop_size=pop_size,
    #                              str_len=str_len,
    #                              optimal_value=None,
    #                              termination_error_value=0,
    #                              no_increase_num=no_increase_num,
    #                              minimization=False,
    #                              show_progress_each=1,
    #                              keep_history=True)

    # optimizer.fit()

    # assert optimizer.get_remains_calls() == pop_size*(iters - no_increase_num - 1)
    # assert optimizer._sign == 1

    # # start with the optimal_value is equal 1
    # optimizer = GeneticAlgorithm(fitness_function,
    #                              iters=iters,
    #                              pop_size=pop_size,
    #                              str_len=str_len,
    #                              optimal_value=1,
    #                              termination_error_value=0,
    #                              no_increase_num=None,
    #                              minimization=False,
    #                              show_progress_each=1,
    #                              keep_history=True)

    # optimizer.fit()
    # assert optimizer.get_remains_calls() == pop_size*(iters - 1)
