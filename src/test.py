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
# from thefittest.tools.print import print_tree


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


def problem(x):
    return np.sin(x[:,0])


function = problem
left_border = -4.5
right_border = 4.5
sample_size = 300
n_dimension = 1

number_of_iterations = 100
population_size = 500

X = np.array([np.linspace(left_border, right_border, sample_size)
              for _ in range(n_dimension)]).T
y = function(X)


functional_set = (FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Neg()),
                  FunctionalNode(Div()))


terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_dimension)]
terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset = UniversalSet(functional_set, tuple(terminal_set))


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree()*np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    return np.array(fitness)


optimizer = GeneticProgramming(fitness_function=fitness_function,
                               uniset=uniset,
                               pop_size=population_size,
                               iters=number_of_iterations,
                               show_progress_each=10,
                               minimization=False,
                               keep_history=False, 
                               selection="tournament_k",
                               mutation="gp_weak_grow", 
                               tour_size=5,
                               max_level=7)

optimizer.fit()

# fittest = optimizer.get_fittest()
# stats = optimizer.get_stats()

# predict = fittest['phenotype']()

# print('The fittest individ:', fittest['phenotype'])
# print('with fitness', fittest['fitness'])

# fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=1)

# ax[0].plot(X[:,0], y, label = "True y")
# ax[0].plot(X[:,0], predict, label = "Predict y")
# ax[0].legend()

# # print_tree(tree=fittest['phenotype'], ax=ax[1])

# plt.tight_layout()
# plt.show()