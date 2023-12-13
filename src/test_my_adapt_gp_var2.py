from collections import defaultdict
from thefittest.optimizers import SelfCGP
from thefittest.optimizers._my_adapt_gp import MyAdaptGP
from thefittest.optimizers._my_adapt_gp_var2 import MyAdaptGPVar2
import numpy as np
import matplotlib.pyplot as plt

from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.optimizers import GeneticProgramming
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Div
from thefittest.tools.operators import Neg
from thefittest.tools.metrics import coefficient_determination
from thefittest.tools.print import print_tree


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

number_of_iterations = 300
population_size = 1000

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


function = problem
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


optimizer = MyAdaptGPVar2(
    fitness_function=fitness_function,
    uniset=uniset,
    pop_size=population_size,
    iters=number_of_iterations,
    show_progress_each=10,
    minimization=False,
    keep_history=True,
    n_jobs=1,
    # max_level=8,
    adaptation_operator="rank",
    mutation="grow",
)


optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

predict = fittest["phenotype"]()

print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=2)


ax[0][0].plot(X[:, 0], y, label="True y")
ax[0][0].plot(X[:, 0], predict, label="Predict y")
ax[0][0].legend()

selectiom_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["s_used"][i].items():
        selectiom_proba[key].append(value)

for key, value in selectiom_proba.items():
    ax[0][1].plot(range(number_of_iterations), value, label=key)
ax[0][1].legend()

crossover_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["c_used"][i].items():
        crossover_proba[key].append(value)

for key, value in crossover_proba.items():
    ax[1][0].plot(range(number_of_iterations), value, label=key)
ax[1][0].legend()

mutation_proba = np.array(stats["m_probas"])

ax[1][1].plot(range(number_of_iterations), mutation_proba.mean(axis=1), label=key)

plt.tight_layout()
plt.savefig("testgp.png")
