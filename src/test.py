import numpy as np
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._base import EphemeralNode
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Inv
from thefittest.tools.operators import Pow2
from thefittest.tools.operators import SqrtAbs
from thefittest.optimizers import SelfCGP
from thefittest.tools.transformations import donothing
from thefittest.tools.transformations import coefficient_determination
from thefittest.benchmarks.symbolicregression17 import problems_dict
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)

F = 'F1'
function = problems_dict[F]['function']
left = problems_dict[F]['bounds'][0]
right = problems_dict[F]['bounds'][1]
size = 300
n_vars = problems_dict[F]['n_vars']
X = np.array([np.linspace(left, right, size) for _ in range(n_vars)]).T

y = function(X)

functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Pow2()),
                  FunctionalNode(Inv()),
                  FunctionalNode(Neg()),
                  FunctionalNode(SqrtAbs()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin())]
terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_vars)]
ephemeral_set = [EphemeralNode(generator1), EphemeralNode(generator2)]

uniset = UniversalSet(functional_set, terminal_set, ephemeral_set)


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree.compile()*np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    fitness = np.array(fitness)
    return fitness


model = SelfCGP(fitness_function=fitness_function,
                genotype_to_phenotype=donothing,
                uniset=uniset,
                pop_size=500, iters=1000,
                show_progress_each=1,
                minimization=False,
                optimal_value=1.0,
                no_increase_num=300,
                keep_history='full')

model.fit()

stats = model.stats


def get_plot(trees):
    x_array = []
    y_array = []
    t = []
    # print(len(trees))
    #
    for i, tree in enumerate(trees):

        y_pred = tree.compile()
        y_pred = np.ones_like(y)*y_pred
        if i > 0:
            if np.all(y_pred != last):
                y_array.extend(y_pred)
                x_array.extend(np.arange(len(y_pred)))
                last = y_pred
                t.append(1)
        else:
            y_array.extend(y_pred)
            x_array.extend(np.arange(len(y_pred)))
            last = y_pred
            t.append(1)

    y_array = np.array(y_array)
    x_array = np.array(x_array)
    y_lim = (np.min(y), np.max(y))

    index = np.repeat(np.arange(len(t)), y_pred.shape[0])
    print(index)

    data = pd.DataFrame({'x': x_array, 'y': y_array, 'i': index})
    fig = px.line(data, x='x', y='y', range_x=(0, len(y_pred)), range_y=y_lim,
                  animation_frame='i')

    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker_size=3)
    fig.add_traces(go.Scatter(x=np.arange(len(y_pred)),
                              y=y,
                              mode='lines',
                              name = 'y_true'))
    fig.show()


get_plot(stats.fittest)