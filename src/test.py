import numpy as np

from thefittest.base import EphemeralNode
from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import UniversalSet
from thefittest.base._tree import Add
from thefittest.base._tree import Div
from thefittest.base._tree import Mul
from thefittest.base._tree import Neg
from thefittest.optimizers import DifferentialEvolution
from thefittest.optimizers import GeneticAlgorithm
from thefittest.optimizers import GeneticProgramming
from thefittest.optimizers import SHADE
from thefittest.optimizers import SHAGA
from thefittest.optimizers import SelfCGA
from thefittest.optimizers import SelfCGP
from thefittest.optimizers import jDE
from thefittest.utils._metrics import coefficient_determination


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

    iters = 3
    pop_size = 5

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
        pop_size=pop_size, uniset=uniset, max_level=5
    )
    selections = (
        # "proportional",
        "rank",
        # "tournament_k",
        # "tournament_3",
    )
    crossover = (
        "gp_empty",
        # "gp_standard",
        # "gp_one_point",
        # "gp_uniform_2",
        # "gp_uniform_k",
        # "gp_uniform_prop_2",
        # "gp_uniform_prop_k",
        # "gp_uniform_rank_2",
        # "gp_uniform_rank_k",
        # "gp_uniform_tour_3",
        # "gp_uniform_tour_k",
    )

    mutation = (
        # "gp_weak_point",
        "gp_custom_rate_point",
        # "gp_weak_grow",
        # "gp_custom_rate_grow",
        # "gp_weak_swap",
        # "gp_custom_rate_swap",
        # "gp_weak_shrink",
        # "gp_custom_rate_shrink",
    )
    for ttt in range(1):
        for selections_i in selections:
            for crossover_i in crossover:
                for mutation_i in mutation:
                    initial_population = GeneticProgramming.half_and_half(
                        pop_size=pop_size, uniset=uniset, max_level=5
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
                        # show_progress_each=1,
                        keep_history=True,
                        init_population=initial_population,
                        selection=selections_i,
                        crossover=crossover_i,
                        mutation=mutation_i,
                        tour_size=3,
                        elitism=True,
                        parents_num=3,
                        mutation_rate=1,
                        random_state=random_state,
                        max_level=5
                    )

                    assert optimizer._specified_selection == selections_i
                    assert optimizer._specified_crossover == crossover_i
                    assert optimizer._specified_mutation == mutation_i
                    assert optimizer._tour_size == 3
                    assert optimizer._parents_num == 3
                    # assert optimizer._mutation_rate == 0.33

                    if selections_i[-2:] == "_k":
                        assert optimizer._selection_pool[optimizer._specified_selection][1] == 3
                    if crossover_i[-2:] == "_k":
                        assert optimizer._crossover_pool[optimizer._specified_crossover][1] == 3
                    # if mutation_i[:9] == "gp_custom":
                    #     assert optimizer._mutation_pool[optimizer._specified_mutation][1] == 0.33

                    optimizer.fit()

                    stats = optimizer.get_stats()
                    # assert np.all(stats["population_g"][0] == initial_population)
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
                        # show_progress_each=1,
                        keep_history=True,
                        init_population=initial_population,
                        selection=selections_i,
                        crossover=crossover_i,
                        mutation=mutation_i,
                        tour_size=3,
                        elitism=True,
                        parents_num=3,
                        mutation_rate=1,
                        random_state=random_state,
                        max_level=5
                    )

                    print('--------------------------')
                    optimizer.fit()

                    genotype_2 = optimizer.get_fittest()["genotype"]
                    # print(mutation_i, crossover_i, selections_i)
                    # print(genotype_1, len(genotype_1), genotype_1.get_max_level())
                    # print(genotype_2, len(genotype_2), genotype_2.get_max_level())
                    # print(genotype_1 != genotype_2)
                    # assert np.all(genotype_1 == genotype_2)
                    genotype_1 == genotype_2


test_GeneticProgramming_set_strategy()
