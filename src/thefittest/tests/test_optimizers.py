import numpy as np

from ..base import EphemeralNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import UniversalSet
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..utils._metrics import coefficient_determination
from ..base._tree import Add
from ..base._tree import Div
from ..base._tree import Mul
from ..base._tree import Neg


def test_GeneticAlgorithm_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    str_len = 200

    # simple start
    optimizer = GeneticAlgorithm(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = GeneticAlgorithm(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
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
    optimizer = GeneticAlgorithm(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_GeneticAlgorithm_set_strategy():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 10
    pop_size = 8
    str_len = 30

    selections = (
        "proportional",
        "rank",
        "tournament_k",
        "tournament_3",
        "tournament_5",
        "tournament_7",
    )
    crossover = (
        "empty",
        "one_point",
        "two_point",
        "uniform_2",
        "uniform_7",
        "uniform_k",
        "uniform_prop_2",
        "uniform_prop_7",
        "uniform_prop_k",
        "uniform_rank_2",
        "uniform_rank_7",
        "uniform_rank_k",
        "uniform_tour_3",
        "uniform_tour_7",
        "uniform_tour_k",
    )

    mutation = ("weak", "average", "strong", "custom_rate")

    tour_size = (3, 4, 5)
    parents_num = (7, 8, 9)
    mutation_rate = (0.11, 0.22)

    for selections_i in selections:
        for mutation_rate_i in mutation_rate:
            for crossover_i in crossover:
                for mutation_i in mutation:
                    for tour_size_i in tour_size:
                        for parents_num_i in parents_num:
                            random_state = np.random.randint(0, 100)
                            initial_population = GeneticAlgorithm.binary_string_population(
                                pop_size=pop_size, str_len=str_len
                            )
                            optimizer = GeneticAlgorithm(
                                fitness_function,
                                iters=iters,
                                pop_size=pop_size,
                                str_len=str_len,
                                optimal_value=None,
                                termination_error_value=0,
                                no_increase_num=None,
                                minimization=False,
                                show_progress_each=None,
                                keep_history=True,
                                init_population=initial_population,
                                selection=selections_i,
                                crossover=crossover_i,
                                mutation=mutation_i,
                                tour_size=tour_size_i,
                                elitism=True,
                                parents_num=parents_num_i,
                                mutation_rate=mutation_rate_i,
                                random_state=random_state,
                            )

                            assert optimizer._specified_selection == selections_i
                            assert optimizer._specified_crossover == crossover_i
                            assert optimizer._specified_mutation == mutation_i
                            assert optimizer._tour_size == tour_size_i
                            assert optimizer._parents_num == parents_num_i
                            assert optimizer._mutation_rate == mutation_rate_i

                            if selections_i[-2:] == "_k":
                                assert (
                                    optimizer._selection_pool[optimizer._specified_selection][1]
                                    == tour_size_i
                                )
                            if crossover_i[-2:] == "_k":
                                assert (
                                    optimizer._crossover_pool[optimizer._specified_crossover][1]
                                    == parents_num_i
                                )
                            if mutation_i == "custom_rate":
                                assert (
                                    optimizer._mutation_pool[optimizer._specified_mutation][1]
                                    == mutation_rate_i
                                )

                            optimizer.fit()

                            stats = optimizer.get_stats()
                            assert np.all(stats["population_g"][0] == initial_population)

                            genotype_1 = optimizer.get_fittest()["genotype"]

                            optimizer = GeneticAlgorithm(
                                fitness_function,
                                iters=iters,
                                pop_size=pop_size,
                                str_len=str_len,
                                optimal_value=None,
                                termination_error_value=0,
                                no_increase_num=None,
                                minimization=False,
                                show_progress_each=None,
                                keep_history=True,
                                init_population=initial_population,
                                selection=selections_i,
                                crossover=crossover_i,
                                mutation=mutation_i,
                                tour_size=tour_size_i,
                                elitism=True,
                                parents_num=parents_num_i,
                                mutation_rate=mutation_rate_i,
                                random_state=random_state,
                            )

                            optimizer.fit()

                            genotype_2 = optimizer.get_fittest()["genotype"]

                            assert np.all(genotype_1 == genotype_2)


def test_SelfCGA_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    str_len = 200

    # simple start
    optimizer = SelfCGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = SelfCGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
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
    optimizer = SelfCGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_SelfCGA_set_strategy():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 10
    pop_size = 10
    str_len = 5
    initial_population = SelfCGA.binary_string_population(pop_size=pop_size, str_len=str_len)
    tour_size = 8
    parents_num = 8
    mutation_rate = 0.5

    optimizer = SelfCGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=None,
        keep_history=True,
        init_population=initial_population,
        selections=(
            "proportional",
            "rank",
            "tournament_k",
            "tournament_3",
        ),
        crossovers=(
            "empty",
            "one_point",
            "two_point",
            "uniform_2",
            "uniform_k",
            "uniform_prop_2",
            "uniform_prop_k",
            "uniform_rank_2",
            "uniform_rank_k",
            "uniform_tour_3",
            "uniform_tour_k",
        ),
        mutations=("weak", "custom_rate"),
        tour_size=tour_size,
        elitism=True,
        parents_num=parents_num,
        mutation_rate=mutation_rate,
        random_state=42,
    )
    optimizer.fit()
    stats = optimizer.get_stats()
    assert np.all(stats["population_g"][0] == initial_population)

    genotype_1 = optimizer.get_fittest()["genotype"]

    optimizer = SelfCGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=None,
        keep_history=True,
        init_population=initial_population,
        selections=(
            "proportional",
            "rank",
            "tournament_k",
            "tournament_3",
        ),
        crossovers=(
            "empty",
            "one_point",
            "two_point",
            "uniform_2",
            "uniform_k",
            "uniform_prop_2",
            "uniform_prop_k",
            "uniform_rank_2",
            "uniform_rank_k",
            "uniform_tour_3",
            "uniform_tour_k",
        ),
        mutations=("weak", "custom_rate"),
        tour_size=tour_size,
        elitism=True,
        parents_num=parents_num,
        mutation_rate=mutation_rate,
        random_state=42,
    )

    optimizer.fit()

    genotype_2 = optimizer.get_fittest()["genotype"]

    assert np.all(genotype_1 == genotype_2)


def test_GeneticProgramming_start_settings():
    def generator1():
        return np.round(np.random.uniform(0, 10), 4)

    def generator2():
        return np.random.randint(0, 10)

    def problem(x):
        return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

    function = problem
    left_border = -4.5
    right_border = 4.5
    sample_size = 300
    n_dimension = 1

    iters = 20
    pop_size = 15

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T
    y = function(X)

    functional_set = (
        FunctionalNode(Add()),
        FunctionalNode(Mul()),
        FunctionalNode(Neg()),
        FunctionalNode(Div()),
    )

    terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
    terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
    uniset = UniversalSet(functional_set, terminal_set)

    def fitness_function(trees):
        fitness = []
        for tree in trees:
            y_pred = tree() * np.ones(len(y))
            fitness.append(-coefficient_determination(y, y_pred))
        return np.array(fitness)

    # simple start
    optimizer = GeneticProgramming(
        fitness_function=fitness_function,
        uniset=uniset,
        pop_size=pop_size,
        iters=iters,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        show_progress_each=1,
        minimization=True,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15

    optimizer = GeneticProgramming(
        fitness_function=fitness_function,
        uniset=uniset,
        pop_size=pop_size,
        iters=iters,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=no_increase_num,
        show_progress_each=1,
        minimization=False,
        keep_history=True,
    )

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = GeneticProgramming(
        fitness_function=fitness_function,
        uniset=uniset,
        pop_size=pop_size,
        iters=iters,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        show_progress_each=1,
        minimization=False,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_GeneticProgramming_set_strategy():
    def generator1():
        return np.round(np.random.uniform(0, 10), 4)

    def generator2():
        return np.random.randint(0, 10)

    def problem(x):
        return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

    function = problem
    left_border = -4.5
    right_border = 4.5
    sample_size = 30
    n_dimension = 1

    iters = 2
    pop_size = 10

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T
    y = function(X)

    functional_set = (
        FunctionalNode(Add()),
        FunctionalNode(Mul()),
        FunctionalNode(Neg()),
        FunctionalNode(Div()),
    )

    terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
    terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
    uniset = UniversalSet(functional_set, terminal_set)

    def fitness_function(trees):
        fitness = []
        for tree in trees:
            y_pred = tree() * np.ones(len(y))
            fitness.append(-coefficient_determination(y, y_pred))
        return np.array(fitness)

    initial_population = GeneticProgramming.half_and_half(
        pop_size=pop_size, uniset=uniset, max_level=14
    )
    selections = (
        "proportional",
        "rank",
        "tournament_k",
        "tournament_3",
    )
    crossover = (
        "gp_empty",
        "gp_standard",
        "gp_one_point",
        "gp_uniform_2",
        "gp_uniform_k",
        "gp_uniform_prop_2",
        "gp_uniform_prop_k",
        "gp_uniform_rank_2",
        "gp_uniform_rank_k",
        "gp_uniform_tour_3",
        "gp_uniform_tour_k",
    )

    mutation = (
        "gp_weak_point",
        "gp_custom_rate_point",
        "gp_weak_grow",
        "gp_custom_rate_grow",
        "gp_weak_swap",
        "gp_custom_rate_swap",
        "gp_weak_shrink",
        "gp_custom_rate_shrink",
    )

    for selections_i in selections:
        for crossover_i in crossover:
            for mutation_i in mutation:
                initial_population = GeneticProgramming.half_and_half(
                    pop_size=pop_size, uniset=uniset, max_level=14
                )
                random_state = np.random.randint(0, 100)
                optimizer = GeneticProgramming(
                    fitness_function,
                    iters=iters,
                    pop_size=pop_size,
                    uniset=uniset,
                    optimal_value=None,
                    termination_error_value=0,
                    no_increase_num=None,
                    minimization=False,
                    show_progress_each=None,
                    keep_history=True,
                    init_population=initial_population,
                    selection=selections_i,
                    crossover=crossover_i,
                    mutation=mutation_i,
                    tour_size=3,
                    elitism=True,
                    parents_num=3,
                    mutation_rate=0.33,
                    random_state=random_state,
                )

                assert optimizer._specified_selection == selections_i
                assert optimizer._specified_crossover == crossover_i
                assert optimizer._specified_mutation == mutation_i
                assert optimizer._tour_size == 3
                assert optimizer._parents_num == 3
                assert optimizer._mutation_rate == 0.33

                if selections_i[-2:] == "_k":
                    assert optimizer._selection_pool[optimizer._specified_selection][1] == 3
                if crossover_i[-2:] == "_k":
                    assert optimizer._crossover_pool[optimizer._specified_crossover][1] == 3
                if mutation_i[:9] == "gp_custom":
                    assert optimizer._mutation_pool[optimizer._specified_mutation][1] == 0.33

                optimizer.fit()

                stats = optimizer.get_stats()
                assert np.all(stats["population_g"][0] == initial_population)
                genotype_1 = optimizer.get_fittest()["genotype"]

                optimizer = GeneticProgramming(
                    fitness_function,
                    iters=iters,
                    pop_size=pop_size,
                    uniset=uniset,
                    optimal_value=None,
                    termination_error_value=0,
                    no_increase_num=None,
                    minimization=False,
                    show_progress_each=None,
                    keep_history=True,
                    init_population=initial_population,
                    selection=selections_i,
                    crossover=crossover_i,
                    mutation=mutation_i,
                    tour_size=3,
                    elitism=True,
                    parents_num=3,
                    mutation_rate=0.33,
                    random_state=random_state,
                )
                optimizer.fit()

                genotype_2 = optimizer.get_fittest()["genotype"]

                assert np.all(genotype_1 == genotype_2)


def test_SelfCGP_start_settings():
    def generator1():
        return np.round(np.random.uniform(0, 10), 4)

    def generator2():
        return np.random.randint(0, 10)

    def problem(x):
        return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

    function = problem
    left_border = -4.5
    right_border = 4.5
    sample_size = 300
    n_dimension = 1

    iters = 20
    pop_size = 15

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T
    y = function(X)

    functional_set = (
        FunctionalNode(Add()),
        FunctionalNode(Mul()),
        FunctionalNode(Neg()),
        FunctionalNode(Div()),
    )

    terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
    terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
    uniset = UniversalSet(functional_set, terminal_set)

    def fitness_function(trees):
        fitness = []
        for tree in trees:
            y_pred = tree() * np.ones(len(y))
            fitness.append(-coefficient_determination(y, y_pred))
        return np.array(fitness)

    # simple start
    optimizer = SelfCGP(
        fitness_function=fitness_function,
        uniset=uniset,
        pop_size=pop_size,
        iters=iters,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        show_progress_each=1,
        minimization=True,
        keep_history=True,
    )

    optimizer.fit()

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15

    optimizer = SelfCGP(
        fitness_function=fitness_function,
        uniset=uniset,
        pop_size=pop_size,
        iters=iters,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=no_increase_num,
        show_progress_each=1,
        minimization=False,
        keep_history=True,
    )

    optimizer.fit()

    assert optimizer.get_remains_calls() == pop_size * (iters - no_increase_num - 1)
    assert optimizer._sign == 1

    # start with the optimal_value is equal 1
    optimizer = SelfCGP(
        fitness_function=fitness_function,
        uniset=uniset,
        pop_size=pop_size,
        iters=iters,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        show_progress_each=1,
        minimization=False,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)


def test_SelfCGP_set_strategy():
    def generator1():
        return np.round(np.random.uniform(0, 10), 4)

    def generator2():
        return np.random.randint(0, 10)

    def problem(x):
        return 3 * x[:, 0] ** 2 + 2 * x[:, 0] + 5

    function = problem
    left_border = -4.5
    right_border = 4.5
    sample_size = 300
    n_dimension = 1

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T
    y = function(X)

    functional_set = (
        FunctionalNode(Add()),
        FunctionalNode(Mul()),
        FunctionalNode(Neg()),
        FunctionalNode(Div()),
    )

    terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
    terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
    uniset = UniversalSet(functional_set, terminal_set)

    def fitness_function(trees):
        fitness = []
        for tree in trees:
            y_pred = tree() * np.ones(len(y))
            fitness.append(-coefficient_determination(y, y_pred))
        return np.array(fitness)

    iters = 10
    pop_size = 10
    initial_population = SelfCGP.half_and_half(pop_size=pop_size, uniset=uniset, max_level=14)
    tour_size = 8
    parents_num = 8
    mutation_rate = 0.5

    optimizer = SelfCGP(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        uniset=uniset,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=None,
        keep_history=True,
        init_population=initial_population,
        selections=(
            "proportional",
            "rank",
            "tournament_k",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers=(
            "gp_empty",
            "gp_standard",
            "gp_one_point",
            "gp_uniform_2",
            "gp_uniform_7",
            "gp_uniform_k",
            "gp_uniform_prop_2",
            "gp_uniform_prop_7",
            "gp_uniform_prop_k",
            "gp_uniform_rank_2",
            "gp_uniform_rank_7",
            "gp_uniform_rank_k",
            "gp_uniform_tour_3",
            "gp_uniform_tour_7",
            "gp_uniform_tour_k",
        ),
        mutations=(
            "gp_weak_point",
            "gp_average_point",
            "gp_strong_point",
            "gp_custom_rate_point",
            "gp_weak_grow",
            "gp_average_grow",
            "gp_strong_grow",
            "gp_custom_rate_grow",
            "gp_weak_swap",
            "gp_average_swap",
            "gp_strong_swap",
            "gp_custom_rate_swap",
            "gp_weak_shrink",
            "gp_average_shrink",
            "gp_strong_shrink",
            "gp_custom_rate_shrink",
        ),
        tour_size=tour_size,
        elitism=True,
        parents_num=parents_num,
        mutation_rate=mutation_rate,
        random_state=42,
    )
    optimizer.fit()
    stats = optimizer.get_stats()
    assert np.all(stats["population_g"][0] == initial_population)
    genotype_1 = optimizer.get_fittest()["genotype"]

    optimizer = SelfCGP(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        uniset=uniset,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=None,
        keep_history=True,
        init_population=initial_population,
        selections=(
            "proportional",
            "rank",
            "tournament_k",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers=(
            "gp_empty",
            "gp_standard",
            "gp_one_point",
            "gp_uniform_2",
            "gp_uniform_7",
            "gp_uniform_k",
            "gp_uniform_prop_2",
            "gp_uniform_prop_7",
            "gp_uniform_prop_k",
            "gp_uniform_rank_2",
            "gp_uniform_rank_7",
            "gp_uniform_rank_k",
            "gp_uniform_tour_3",
            "gp_uniform_tour_7",
            "gp_uniform_tour_k",
        ),
        mutations=(
            "gp_weak_point",
            "gp_average_point",
            "gp_strong_point",
            "gp_custom_rate_point",
            "gp_weak_grow",
            "gp_average_grow",
            "gp_strong_grow",
            "gp_custom_rate_grow",
            "gp_weak_swap",
            "gp_average_swap",
            "gp_strong_swap",
            "gp_custom_rate_swap",
            "gp_weak_shrink",
            "gp_average_shrink",
            "gp_strong_shrink",
            "gp_custom_rate_shrink",
        ),
        tour_size=tour_size,
        elitism=True,
        parents_num=parents_num,
        mutation_rate=mutation_rate,
        random_state=42,
    )

    optimizer.fit()

    genotype_2 = optimizer.get_fittest()["genotype"]

    assert np.all(genotype_1 == genotype_2)


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
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
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

    iters = 2
    pop_size = 10
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    mutations = ("best_1", "rand_1", "current_to_best_1", "rand_to_best1", "best_2", "rand_2")
    F = (0.5, 0.3)
    CR = (0.31, 0.33)

    for mutation_i in mutations:
        for F_i in F:
            for CR_i in CR:
                initial_population = DifferentialEvolution.float_population(
                    pop_size=pop_size, left=left, right=right
                )
                random_state = np.random.randint(0, 100)
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
                    mutation=mutation_i,
                    init_population=initial_population,
                    F=F_i,
                    CR=CR_i,
                    random_state=random_state,
                )
                optimizer.fit()
                stats = optimizer.get_stats()

                assert optimizer._specified_mutation == mutation_i
                assert optimizer._F == F_i
                assert optimizer._CR == CR_i

                assert np.all(stats["population_g"][0] == initial_population)

                genotype_1 = optimizer.get_fittest()["genotype"]

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
                    mutation=mutation_i,
                    init_population=initial_population,
                    F=F_i,
                    CR=CR_i,
                    random_state=random_state,
                )

                optimizer.fit()

                genotype_2 = optimizer.get_fittest()["genotype"]

                assert np.all(genotype_1 == genotype_2)


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
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
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

    iters = 2
    pop_size = 10
    n_vars = 10
    left = np.full(n_vars, -1, dtype=np.float64)
    right = np.full(n_vars, 1, dtype=np.float64)

    mutations = ("best_1", "rand_1", "current_to_best_1", "rand_to_best1", "best_2", "rand_2")
    F_min = (0.05, 0.1)
    F_max = (0.9, 0.95)
    t_F = (0.051, 0.21)
    t_CR = (0.05, 0.1)

    for mutation_i in mutations:
        for F_min_i in F_min:
            for F_max_i in F_max:
                for t_F_i in t_F:
                    for t_CR_i in t_CR:
                        initial_population = jDE.float_population(
                            pop_size=pop_size, left=left, right=right
                        )
                        random_state = np.random.randint(0, 100)
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
                            mutation=mutation_i,
                            F_min=F_min_i,
                            F_max=F_max_i,
                            t_CR=t_CR_i,
                            t_F=t_F_i,
                            init_population=initial_population,
                            random_state=random_state,
                        )
                        optimizer.fit()
                        stats = optimizer.get_stats()

                        assert optimizer._specified_mutation == mutation_i
                        assert optimizer._F_min == F_min_i
                        assert optimizer._F_max == F_max_i
                        assert optimizer._t_CR == t_CR_i
                        assert optimizer._t_F == t_F_i
                        print(stats.keys())
                        assert np.all(stats["population_g"][0] == initial_population)

                        genotype_1 = optimizer.get_fittest()["genotype"]

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
                            mutation=mutation_i,
                            F_min=F_min_i,
                            F_max=F_max_i,
                            t_CR=t_CR_i,
                            t_F=t_F_i,
                            init_population=initial_population,
                            random_state=random_state,
                        )

                        optimizer.fit()

                        genotype_2 = optimizer.get_fittest()["genotype"]

                        assert np.all(genotype_1 == genotype_2)


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
        random_state=42,
    )

    optimizer.fit()

    genotype_1 = optimizer.get_fittest()["genotype"]

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
        random_state=42,
    )
    optimizer.fit()

    genotype_2 = optimizer.get_fittest()["genotype"]

    assert np.all(genotype_1 == genotype_2)

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
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


def test_SHAGA_start_settings():
    def fitness_function(x):
        return np.sum(x, axis=1, dtype=np.float64)

    iters = 100
    pop_size = 50
    str_len = 200

    # simple start
    optimizer = SHAGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
        random_state=42,
    )

    optimizer.fit()

    genotype_1 = optimizer.get_fittest()["genotype"]

    optimizer = SHAGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=None,
        termination_error_value=0,
        no_increase_num=None,
        minimization=True,
        show_progress_each=1,
        keep_history=True,
        random_state=42,
    )

    optimizer.fit()

    genotype_2 = optimizer.get_fittest()["genotype"]

    assert np.all(genotype_1 == genotype_2)

    fittest = optimizer.get_fittest()
    assert isinstance(fittest, dict)

    stats = optimizer.get_stats()

    assert optimizer.get_remains_calls() == 0
    assert len(stats["max_fitness"]) == iters
    assert optimizer._sign == -1

    # start with the no_increase_num is equal 15
    def fitness_function(x):
        return np.ones(len(x), dtype=np.float64)

    no_increase_num = 15
    optimizer = SHAGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
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
    optimizer = SHAGA(
        fitness_function,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        optimal_value=1,
        termination_error_value=0,
        no_increase_num=None,
        minimization=False,
        show_progress_each=1,
        keep_history=True,
    )

    optimizer.fit()
    assert optimizer.get_remains_calls() == pop_size * (iters - 1)
