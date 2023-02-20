import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from thefittest.optimizers._base import Tree
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._operators import Mul
from thefittest.optimizers._operators import Sub
from thefittest.optimizers._operators import Add
from thefittest.optimizers._operators import Pow
from thefittest.optimizers._operators import Cos
from thefittest.optimizers._operators import Sin
from thefittest.optimizers._operators import Neg
from thefittest.optimizers._operators import Div
from thefittest.tools import donothing
from sklearn.model_selection import train_test_split
from thefittest.optimizers import GeneticProgramming, SelfCGP


def graph(some_tree):
    reverse_nodes = some_tree.nodes[::-1].copy()
    pack = []
    edges = []
    nodes = []
    labels = {}

    for i, node in enumerate(reverse_nodes):
        labels[len(reverse_nodes) - i -
               #    1] = str(len(reverse_nodes) - i - 1) + '. ' + node.sign  # один раз развернуть или вообще не разворачивать а сразу считать так
               1] = node.sign[:7]

        nodes.append(len(reverse_nodes) - i - 1)

        for _ in range(node.n_args):
            edges.append((len(reverse_nodes) - i - 1,
                         len(reverse_nodes) - pack.pop() - 1))
        pack.append(i)

    edges.reverse()
    nodes.reverse()

    return edges, labels, nodes


def print_tree(some_tree, fig_name, underline_nodes=[]):
    edges, labels, nodes = graph(some_tree)
    levels = some_tree.levels

    colors = np.zeros(shape=(len(nodes), 4))
    pos = np.zeros(shape=(len(nodes), 2))
    for i, lvl_i in enumerate(levels):
        total = 0
        cond = lvl_i == np.array(levels)
        h = 1/(1 + np.sum(cond))
        arange = np.arange(len(pos))[cond]

        for j, a_j in enumerate(arange):
            total += h
            pos[a_j][0] = total

        pos[i][1] = -lvl_i

    for i, node in enumerate(some_tree.nodes):
        if i in underline_nodes:
            colors[i] = (0.5, 0.1, 0.1, 1)
        else:
            if type(node) is FunctionalNode:
                colors[i] = (1, 0.72, 0.43, 1)
            else:
                colors[i] = (0.21, 0.76, 0.56, 1)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    nx.draw_networkx_nodes(g, pos, node_color=colors,
                           edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=10)

    plt.savefig(fig_name)
    plt.close()


def root_mean_square_error(y_true, y_predict):
    return np.sqrt(np.mean((y_true - y_predict)**2))

# def problem(x):
#     return np.cos(x[:, 0]) + 5*x[:, 1]


def problem(x):
    return np.cos(x[:,0])
    # return 20*np.cos(2*x[:, 0]) + 0.3*x[:, 0]*x[:, 0]


left = -12
right = 12
size = 500
n_vars = 1
X = np.random.uniform(left, right, size=(size, n_vars))
y = problem(X)
# y = y + np.random.uniform(0, 10, len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

uniset = UniversalSet(functional_set=(Add(),
                                    #   Cos(),
                                    #   Sin(),
                                      Div(),
                                      Mul(),
                                      Sub(),
                                      ),
                      terminal_set={'x0': X_train[:, 0]},
                      constant_set=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
                      )


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree.compile()*np.ones(len(y_train))
        level = tree.get_max_level()
        if level > 49:
            fine = len(tree.nodes)
            # fine = 0
        else:
            fine = 0
        fitness.append(root_mean_square_error(y_train, y_pred) + fine)
    return 1/(1 + np.array(fitness))


model = SelfCGP(fitness_function=fitness_function,
                genotype_to_phenotype=donothing,
                uniset=uniset,
                pop_size=500, iters=1000,
                show_progress_each=10,
                minimization=False,
                no_increase_num=300,
                keep_history='full')
model.tour_size = 3
model.max_level = 15
model.fit()

fittest = model.thefittest.phenotype

stats = model.stats
print(fittest)

x_plot = np.linspace(left, right, size).reshape(-1, 1)
y_plot = problem(x_plot)

fittest_pred = fittest.change_terminals({'x0': x_plot})
y_pred = fittest_pred.compile()
if type(y_pred) is not np.ndarray:
    y_pred = np.full(len(x_plot), y_pred)

y_pred_train = fittest.compile()

plt.plot(x_plot[:,0], y_plot, label='true', color='green')
plt.scatter(X_train[:,0], y_train, label='train', color='black', alpha=0.3)
plt.scatter(X_train[:,0], y_pred_train, label='train', color='red', alpha=0.3)
plt.plot(x_plot[:,0], y_pred, color='red')
plt.legend()
plt.savefig('line1.png')
plt.close()

print_tree(fittest, 'fittest.png')

plt.plot(range(len(stats.fitness)), stats.fitness)
plt.savefig('fitness.png')
plt.close()


for key, value in stats.m_proba.items():
    plt.plot(range(len(value)), value, label = key)
plt.legend()
plt.savefig('m_proba.png')
plt.close()

for key, value in stats.c_proba.items():
    plt.plot(range(len(value)), value, label = key)
plt.legend()
plt.savefig('c_proba.png')
plt.close()

for key, value in stats.s_proba.items():
    plt.plot(range(len(value)), value, label = key)
plt.legend()
plt.savefig('s_proba.png')
plt.close()
