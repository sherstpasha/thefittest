import numpy as np
import matplotlib.pyplot as plt

from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.classifiers._gpnneclassifier import TwoTreeGeneticProgramming
from thefittest.classifiers._gpnneclassifier import TwoTreeSelfCGP
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Div
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.metrics import coefficient_determination
from thefittest.tools.print import print_tree
from collections import defaultdict


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


def problem2(x):
    return np.sin(x[:, 0])


def problem(x):
    return np.sin(x[:, 0])


function = problem
left_border = -4.5
right_border = 4.5
sample_size = 300
n_dimension = 1

number_of_iterations = 500
population_size = 1000

X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
y = function(X)
y2 = problem2(X)

functional_set = (
    FunctionalNode(Add()),
    FunctionalNode(Mul()),
    FunctionalNode(Neg()),
    FunctionalNode(Div()),
)


terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset_1 = UniversalSet(functional_set, tuple(terminal_set))

functional_set2 = (
    FunctionalNode(Add()),
    FunctionalNode(Mul()),
    FunctionalNode(Neg()),
    FunctionalNode(Div()),
)


terminal_set2 = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
terminal_set2.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset_2 = UniversalSet(functional_set2, tuple(terminal_set2))


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred_1 = tree[0]() * np.ones(len(y))
        y_pred_2 = tree[1]() * np.ones(len(y))
        fitness.append(
            coefficient_determination(y, y_pred_1) + coefficient_determination(y2, y_pred_2)
        )
    return np.array(fitness, dtype=np.float64)


# optimizer = TwoTreeGeneticProgramming(
#     fitness_function=fitness_function,
#     uniset_1=uniset_1,
#     uniset_2=uniset_2,
#     pop_size=population_size,
#     iters=number_of_iterations,
#     show_progress_each=10,
#     minimization=False,
#     keep_history=False,
#     selection="tournament_k",
#     mutation="gp_weak_grow",
#     tour_size=5,
#     max_level=10,
#     elitism=True,
# )

optimizer = TwoTreeSelfCGP(
    fitness_function=fitness_function,
    uniset_1=uniset_1,
    uniset_2=uniset_2,
    pop_size=population_size,
    iters=number_of_iterations,
    show_progress_each=1,
    minimization=False,
    keep_history=True,
    max_level=10,
    elitism=True,
)


# optimizer._first_generation()
# print(optimizer._population_g_i)
optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

predict_1 = fittest["phenotype"][0]() * np.ones(len(y))
predict_2 = fittest["phenotype"][1]() * np.ones(len(y))

print("The fittest individ:", fittest["phenotype"][0], fittest["phenotype"][1])
print("with fitness", fittest["fitness"])

fig, ax = plt.subplots(figsize=(14, 7), ncols=4, nrows=2)

ax[0][0].plot(X[:, 0], y, label="True y")
ax[0][0].plot(X[:, 0], predict_1, label="Predict y")
ax[0][0].legend()

ax[0][1].plot(X[:, 0], y2, label="True y")
ax[0][1].plot(X[:, 0], predict_2, label="Predict y")
ax[0][1].legend()

# ax[2].plot(X[:, 0], y, label="True y")
# ax[2].plot(X[:, 0], predict, label="Predict y")
# ax[2].legend()
selectiom_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["s_proba"][i].items():
        selectiom_proba[key].append(value)

for key, value in selectiom_proba.items():
    ax[1][0].plot(range(number_of_iterations), value, label=key)
ax[1][0].legend()

crossover_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["c_proba"][i].items():
        crossover_proba[key].append(value)

for key, value in crossover_proba.items():
    ax[1][1].plot(range(number_of_iterations), value, label=key)
ax[1][1].legend()

mutation_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["m_proba"][i].items():
        mutation_proba[key].append(value)

for key, value in mutation_proba.items():
    ax[1][2].plot(range(number_of_iterations), value, label=key)
ax[1][2].legend()

print_tree(tree=fittest["phenotype"][0], ax=ax[0][2])
print_tree(tree=fittest["phenotype"][1], ax=ax[0][3])

plt.tight_layout()
plt.show()
