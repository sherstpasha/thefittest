import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._base import EphemeralNode
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Sub
from thefittest.tools.operators import Add
# from thefittest.tools.operators import Pow
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Div
from thefittest.tools.operators import Exp
from thefittest.tools.operators import SqrtAbs
from thefittest.tools.operators import Inv
from thefittest.tools.operators import Pow2
# from thefittest.tools.operators import LogAbs
# from thefittest.tools.operators import Pow3
# from thefittest.tools.operators import FloorDiv
from thefittest.tools.transformations import donothing
from thefittest.optimizers import SelfCGP, GeneticProgramming
from thefittest.tools.transformations import scale_data
from thefittest.tools.transformations import root_mean_square_error
from thefittest.tools.transformations import coefficient_determination
from thefittest.benchmarks.symbolicregression17 import problems_dict


min_value = np.finfo(np.float64).min
max_value = np.finfo(np.float64).max


def print_tree(some_tree, ax):
    graph = some_tree.get_graph(False)
    g = nx.Graph()
    g.add_nodes_from(graph['nodes'])
    g.add_edges_from(graph['edges'])

    nx.draw_networkx_nodes(g, graph['pos'], node_color=graph['colors'],
                           edgecolors='black', linewidths=0.5, ax=ax)
    nx.draw_networkx_edges(g, graph['pos'], ax=ax)
    nx.draw_networkx_labels(
        g, graph['pos'], graph['labels'], font_size=10, ax=ax)


def problem(x):
    return np.sin(x[:, 0])


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


F = 'F3'
function = problems_dict[F]['function']
left = problems_dict[F]['bounds'][0]
right = problems_dict[F]['bounds'][1]
# function = problem
# left = -4.75
# right = 4.75
size = 300
n_vars = problems_dict[F]['n_vars']
# n_vars = 1
X = np.array([np.linspace(left, right, size) for _ in range(n_vars)]).T


y = function(X)

functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                #   FunctionalNode(Div()),
                #   FunctionalNode(Sub()),
                  FunctionalNode(Pow2()),
                  FunctionalNode(Inv()),  
                  FunctionalNode(Neg()),
                #   FunctionalNode(Exp()),
                #   FunctionalNode(SqrtAbs()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin()),
                  ]


terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_vars)]

ephemeral_set = [EphemeralNode(generator1), EphemeralNode(generator2)]

uniset = UniversalSet(functional_set, terminal_set, ephemeral_set)


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree.compile()*np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    fitness = np.array(fitness)
    return np.clip(fitness, -1e32, 1)


model = SelfCGP(fitness_function=fitness_function,
                genotype_to_phenotype=donothing,
                uniset=uniset,
                pop_size=500, iters=300,
                show_progress_each=1,
                minimization=False,
                no_increase_num=300,
                keep_history='quick')
model.fit()

fittest = model.thefittest.phenotype

stats = model.stats
print(fittest)
y_pred = fittest.compile()
y_pred = np.ones_like(y)*y_pred

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=3)

ax[0][0].plot(range(len(X)), y, label='y_true', color='green')
ax[0][0].plot(range(len(X)), y_pred, label='y_pred', color='red')
ax[0][0].legend(loc="upper left")

ax[0][1].plot(range(len(stats.fitness)), stats.fitness)

for key, value in stats.m_proba.items():
    ax[1][0].plot(range(len(value)), value, label=key)
ax[1][0].legend(loc="upper left")

for key, value in stats.c_proba.items():
    ax[1][1].plot(range(len(value)), value, label=key)
ax[1][1].legend(loc="upper left")

for key, value in stats.s_proba.items():
    ax[2][0].plot(range(len(value)), value, label=key)
ax[2][0].legend(loc="upper left")

print_tree(fittest, ax[2][1])

plt.tight_layout()
plt.savefig('line1.png')
plt.close()
