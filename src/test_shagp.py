from collections import defaultdict
from thefittest.optimizers import SelfCGP
from thefittest.optimizers._shagp import SHAGP
import numpy as np
import matplotlib.pyplot as plt

from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.optimizers import GeneticProgramming, PDPGP
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Div
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Sin
from thefittest.tools.operators import Exp
from thefittest.tools.metrics import coefficient_determination
from thefittest.tools.print import print_tree
from thefittest.benchmarks.symbolicregression17 import problems_dict


F = "F2"


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


def problem(x):
    # return np.sin(X[:, 0])
    return problems_dict[F]["function"](x)


function = problem
left_border = problems_dict[F]["bounds"][0]
right_border = problems_dict[F]["bounds"][1]
sample_size = 300
n_dimension = problems_dict[F]["n_vars"]

number_of_iterations = 300
population_size = 300

X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
y = function(X)


functional_set = (
    FunctionalNode(Add()),
    FunctionalNode(Mul()),
    FunctionalNode(Neg()),
    FunctionalNode(Div()),
    FunctionalNode(Sin()),
    # FunctionalNode(Exp()),
)


terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset = UniversalSet(functional_set, tuple(terminal_set))


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree() * np.ones(len(y))
        fitness.append(coefficient_determination(y.astype(np.float32), y_pred.astype(np.float32)))
    return np.array(fitness, dtype=np.float32)


optimizer = SHAGP(
    fitness_function=fitness_function,
    uniset=uniset,
    pop_size=population_size,
    iters=number_of_iterations,
    show_progress_each=10,
    minimization=False,
    keep_history=True,
    selection="rank",
    crossover="gp_standart",
    mutation="shrink",
    # tour_size=10,
    # max_level=10,
)

# optimizer = PDPGP(
#     fitness_function=fitness_function,
#     uniset=uniset,
#     pop_size=population_size,
#     iters=number_of_iterations,
#     show_progress_each=10,
#     minimization=False,
#     keep_history=True,
#     # selection="rank",
#     # crossover="gp_standart",
#     # mutation="shrink",
#     # tour_size=10,
#     # max_level=10,
# )

print(optimizer._first_generation())
print(optimizer._population_g_i)
optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

predict = fittest["phenotype"]() * np.ones(len(y))

print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])

print(stats.keys())

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=2)


ax[0][0].plot(X[:, 0], y, label="True y")
ax[0][0].plot(X[:, 0], predict, label="Predict y")
ax[0][0].legend()

ax[0][1].plot(range(number_of_iterations), stats["max_fitness"])
ax[0][1].set_title("Fitness")
ax[0][1].set_ylabel("Fitness value")
ax[0][1].set_xlabel("Iterations")

ax[1][0].plot(range(number_of_iterations), np.array(stats["H_MR"]).mean(axis=1))
ax[1][0].set_title("Mean history of the H_MR")
ax[1][0].set_ylabel("Mean H_MR")
ax[1][0].set_xlabel("Iterations")

ax[1][1].plot(range(number_of_iterations), np.array(stats["H_CR"]).mean(axis=1))
ax[1][1].set_title("Mean history of the CR")
ax[1][1].set_ylabel("Mean CR")
ax[1][1].set_xlabel("Iterations")

plt.tight_layout()
plt.savefig("testgp.png")
