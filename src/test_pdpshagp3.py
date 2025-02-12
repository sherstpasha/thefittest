from collections import defaultdict
from thefittest.optimizers._pdpshagp import PDPSHAGP, SelfCSHAGP
import numpy as np
import matplotlib.pyplot as plt

from thefittest.base import FunctionalNode, TerminalNode, EphemeralNode, UniversalSet
from thefittest.tools.operators import (
    Mul,
    Add,
    Div,
    Sin,
    Exp,
    Sub,
    SqrtAbs,
    LogAbs,
    Cos,
    Arcsin,
    Tanh,
    Neg,
)
from thefittest.tools.metrics import coefficient_determination, root_mean_square_error
from thefittest.tools.print import print_tree
from thefittest.benchmarks.symbolicregression17 import problems_dict

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# # Выбираем задачу
# F = "F3"


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


def generator3():
    return np.e


def generator4():
    return np.pi


number_of_iterations = 1000
population_size = 300

# Формируем выборку
data = np.loadtxt(
    r"C:\Users\pasha\OneDrive\Рабочий стол\Feynman_with_units\Feynman_with_units\I.12.5",
)

print(data.shape)

X = data[:1000, :-1]
y = data[:1000, -1]

print(X.shape, y.shape)


n_dimension = X.shape[1]

# X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
# y = function(X)


functional_set = (
    FunctionalNode(Add()),
    FunctionalNode(Sub()),
    FunctionalNode(Mul()),
    FunctionalNode(Div()),
    FunctionalNode(SqrtAbs()),
    FunctionalNode(Exp()),
    FunctionalNode(LogAbs()),
    FunctionalNode(Sin()),
    FunctionalNode(Cos()),
)


terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
terminal_set.extend(
    [
        EphemeralNode(generator1),
        EphemeralNode(generator2),
        EphemeralNode(generator3),
        EphemeralNode(generator4),
    ]
)
uniset = UniversalSet(functional_set, tuple(terminal_set))


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree() * np.ones(len(y))
        fitness.append(root_mean_square_error(y.astype(np.float32), y_pred.astype(np.float32)))
    return np.array(fitness, dtype=np.float32)


optimizer = PDPSHAGP(
    fitness_function=fitness_function,
    uniset=uniset,
    pop_size=population_size,
    iters=number_of_iterations,
    show_progress_each=10,
    minimization=True,
    keep_history=True,
    # optimal_value=0,
)

print(optimizer._first_generation())
print(optimizer._population_g_i)
optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

predict = fittest["phenotype"]() * np.ones(len(y))

print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])

print(stats.keys())

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=3)

calls_done = number_of_iterations - (optimizer.get_remains_calls() / population_size)

ax[0][0].plot(range(number_of_iterations), stats["max_fitness"])
ax[0][0].set_title("Fitness")
ax[0][0].set_ylabel("Fitness value")
ax[0][0].set_xlabel("Iterations")

selectiom_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["s_proba"][i].items():
        selectiom_proba[key].append(value)

for key, value in selectiom_proba.items():
    ax[0][1].plot(range(number_of_iterations), value, label=key)
ax[0][1].legend()

crossover_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["c_proba"][i].items():
        crossover_proba[key].append(value)

for key, value in crossover_proba.items():
    ax[1][0].plot(range(number_of_iterations), value, label=key)
ax[1][0].legend()

mutation_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["m_proba"][i].items():
        mutation_proba[key].append(value)

for key, value in mutation_proba.items():
    ax[1][1].plot(range(number_of_iterations), value, label=key)
ax[1][1].legend()

ax[2][0].plot(range(number_of_iterations), np.array(stats["H_MR"]).mean(axis=1))

ax[2][1].plot(range(number_of_iterations), np.array(stats["H_CR"]).mean(axis=1))

plt.tight_layout()
plt.savefig("pdpshagp.png")
