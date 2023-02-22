import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from thefittest.optimizers._base import Tree
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Sub
from thefittest.tools.operators import Add
from thefittest.tools.operators import Pow
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Div
from thefittest.tools.operators import Exp
from thefittest.tools.operators import Sqrt
from thefittest.tools.transformations import donothing
from sklearn.model_selection import train_test_split
from thefittest.optimizers import GeneticProgramming
from thefittest.optimizers import SelfCGP
from thefittest.tools.transformations import scale_data


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
    return np.sin(x[:,0])
    # return 20*np.cos(x[:, 0]) + x[:, 0]*x[:, 0]


def problem_1(x):
    return 0.05*((x[:, 0]-1)**2) + (3 - 2.9*np.exp(-2.77257*(x[:, 0]**2)))*(1 - np.cos(x[:, 0]*(4-50*np.exp(-2.77257*(x[:, 0]**2)))))


def problem_2(x):
    return 1 - 0.5*np.cos(1.5*(10*x[:, 0]-0.3))*np.cos(31.4*x[:, 0]) + 0.5*np.cos(np.sqrt(5)*10*x[:, 0])*np.cos(35*x[:, 0])


left = -4.5
right = 4.5
size = 100
n_vars = 1
X = np.linspace(left, right, size).reshape(-1, 1)
y = problem(X)
# y = scale_data(y)
# y = problem + np.random.uniform(0, 10, len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


def generator():
    return np.round(np.random.uniform(-5, 5), 4)


uniset = UniversalSet(functional_set=(
    Add(),
    Sub(),
    Mul(),
    Div(),
    # Sin(),
    # Cos(),
    # Pow(),
    # Exp(),
    # Sqrt(),
),
    terminal_set={'x0': X_train[:, 0]},
    constant_set={'e1': generator}
)

def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree.compile()*np.ones(len(y_train))
        fitness.append(root_mean_square_error(y_train, y_pred))
    return 1/(1 + np.array(fitness))


model = SelfCGP(fitness_function=fitness_function,
                genotype_to_phenotype=donothing,
                uniset=uniset,
                pop_size=10000, iters=101,
                show_progress_each=10,
                minimization=False,
                no_increase_num=300,
                keep_history='quick')
model.max_level = 16
model.fit()

fittest = model.thefittest.phenotype

stats = model.stats
print(fittest)

x_plot = np.linspace(left, right, size).reshape(-1, 1)
y_plot = problem(x_plot)
# y_plot = scale_data(y_plot)

fittest_pred = fittest.change_terminals({'x0': x_plot})
y_pred = fittest_pred.compile()
if type(y_pred) is not np.ndarray:
    y_pred = np.full(len(x_plot), y_pred)

y_pred_train = fittest.compile()
y_pred_train = y_pred_train

plt.plot(x_plot[:, 0], y_plot, label='true', color='green')
plt.scatter(X_train[:, 0], y_train, label='train', color='black', alpha=0.3)
# plt.scatter(X_train[:,0], y_pred_train, label='train', color='red', alpha=0.3)
plt.plot(x_plot[:, 0], y_pred, color='red')
plt.legend()
plt.savefig('line1.png')
plt.close()

print_tree(fittest, 'fittest.png')

plt.plot(range(len(stats.fitness)), stats.fitness)
plt.savefig('fitness.png')
plt.close()


for key, value in stats.m_proba.items():
    plt.plot(range(len(value)), value, label=key)
plt.legend()
plt.savefig('m_proba.png')
plt.close()

for key, value in stats.c_proba.items():
    plt.plot(range(len(value)), value, label=key)
plt.legend()
plt.savefig('c_proba.png')
plt.close()

for key, value in stats.s_proba.items():
    plt.plot(range(len(value)), value, label=key)
plt.legend()
plt.savefig('s_proba.png')
plt.close()
