# from thefittest.base._ea import EvolutionaryAlgorithm

# def function(x):
#     return np.sum(x)


# model = EvolutionaryAlgorithm(fitness_function = function,
#         iters = 100,
#         pop_size = 100,
#         elitism = True,
#         init_population = None,
#         genotype_to_phenotype = None,
#         optimal_value = None,
#         termination_error_value = 0.0,
#         no_increase_num = None,
#         minimization = False,
#         show_progress_each = None,
#         keep_history = False,
#         n_jobs = 1,
#         fitness_function_args = None,
#         genotype_to_phenotype_args = None,
#         random_state = None)

# model.fit()


# from thefittest.optimizers import GeneticAlgorithm, SHAGA, SelfCGA, PDPGA
# from thefittest.benchmarks import OneMax

# number_of_iterations = 10
# population_size = 10
# string_length = 50

# optimizer = PDPGA(
#     fitness_function=OneMax(),
#     iters=number_of_iterations,
#     pop_size=population_size,
#     str_len=string_length,
#     show_progress_each=None,
#     random_state=18,
# )

# optimizer.fit()

# print(optimizer.get_fittest())

# optimizer = PDPGA(
#     fitness_function=OneMax(),
#     iters=number_of_iterations,
#     pop_size=population_size,
#     str_len=string_length,
#     show_progress_each=None,
#     random_state=18,
# )

# optimizer.fit()

# print(optimizer.get_fittest())

# optimizer.fit()

# fittest = optimizer.get_fittest()


# import numpy as np
# import matplotlib.pyplot as plt

# from thefittest.benchmarks import Griewank
# from thefittest.optimizers import DifferentialEvolution, SHADE


# n_dimension = 10
# left_border = -100.0
# right_border = 100.0
# number_of_iterations = 50
# population_size = 50


# left_border_array = np.full(shape=n_dimension, fill_value=left_border, dtype=np.float64)
# right_border_array = np.full(shape=n_dimension, fill_value=right_border, dtype=np.float64)

# optimizer = SHADE(
#     fitness_function=Griewank(),
#     iters=number_of_iterations,
#     pop_size=population_size,
#     left=left_border_array,
#     right=right_border_array,
#     # show_progress_each=10,
#     minimization=True,
#     keep_history=True,
#     random_state=18,
# )

# optimizer.fit()

# print(optimizer.get_fittest())

# optimizer = SHADE(
#     fitness_function=Griewank(),
#     iters=number_of_iterations,
#     pop_size=population_size,
#     left=left_border_array,
#     right=right_border_array,
#     # show_progress_each=10,
#     minimization=True,
#     keep_history=True,
#     random_state=18,
# )

# optimizer.fit()

# print(optimizer.get_fittest())


import numpy as np
import matplotlib.pyplot as plt

from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.optimizers import GeneticProgramming
from thefittest.base._tree import Mul
from thefittest.base._tree import Add
from thefittest.base._tree import Div
from thefittest.base._tree import Neg
from thefittest.utils._metrics import coefficient_determination


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


def problem(x):
    return np.sin(x[:, 0])


function = problem
left_border = -4.5
right_border = 4.5
sample_size = 300
n_dimension = 1

number_of_iterations = 100
population_size = 500

X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
y = function(X)


functional_set = (
    FunctionalNode(Add()),
    FunctionalNode(Mul()),
    FunctionalNode(Neg()),
    FunctionalNode(Div()),
)


terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset = UniversalSet(functional_set, tuple(terminal_set))


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree() * np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    return np.array(fitness)


optimizer = GeneticProgramming(
    fitness_function=fitness_function,
    uniset=uniset,
    pop_size=population_size,
    iters=number_of_iterations,
    # show_progress_each=1,
    minimization=False,
    keep_history=False,
    selection="tournament_k",
    mutation="gp_weak_grow",
    tour_size=5,
    max_level=7,
    random_state=18,
)

optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

predict = fittest["phenotype"]()

print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])


optimizer = GeneticProgramming(
    fitness_function=fitness_function,
    uniset=uniset,
    pop_size=population_size,
    iters=number_of_iterations,
    # show_progress_each=10,
    minimization=False,
    keep_history=False,
    selection="tournament_k",
    mutation="gp_weak_grow",
    tour_size=5,
    max_level=7,
    random_state=18,
)

optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

predict = fittest["phenotype"]()

print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])
