import plotly.express as px
import plotly
import plotly.graph_objects as go


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thefittest.optimizers import SelfCGA
from thefittest.optimizers import GeneticAlgorithm
from thefittest.benchmarks._optproblems import *
from thefittest.benchmarks import CEC2005
from thefittest.tools.transformations import GrayCode
from thefittest.tools.transformations import donothing


def print_population_by_time(population_3d, grid_model, function):

    array3d = np.array(list(map(grid_model.transform, population_3d)))

    array2d = array3d.reshape(-1, 2)
    index = np.repeat(np.arange(array3d.shape[0]), array3d.shape[1])

    fx = function(array2d)
    data = pd.DataFrame({'x': array2d[:, 0],
                         'y': array2d[:, 1],
                         'z':  fx,
                         'i': index,
                         'inv_z': -fx})

    xlim = (grid_model.left[0], grid_model.right[0])
    ylim = (grid_model.left[1], grid_model.right[1])
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = np.linspace(ylim[0], ylim[1], 1000)
    z = function.build_grid(x, y)
    print(np.min(z))
    zlim = (np.min(z), np.max(z))

    fig = px.scatter_3d(data, x='x', y='y', z='z',
                        animation_frame='i',
                        range_x=xlim,
                        range_y=ylim,
                        range_z=zlim,
                        color='inv_z')
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker_size=3)
    fig.add_traces(go.Surface(x=x, y=y, z=z))

    fig.write_html("C:/Users/user/Desktop/file1.html")


n_variables = 30

left = np.full(n_variables, -100, dtype=np.float64)
right = np.full(n_variables, 100, dtype=np.float64)
# right = np.array([-100, -40], dtype = np.float64)
# left = np.array([-50, 0], dtype = np.float64)
parts = np.full(n_variables, 63, dtype=np.int64)

gray_code_to_float = GrayCode(fit_by='parts').fit(
    left=left, right=right, arg=parts)

# problem = CEC2005.ShiftedSphere()
problem = Sphere()
# problem = OneMax()

# problem = HighConditionedElliptic()
model = SelfCGA(fitness_function=problem,
                genotype_to_phenotype=gray_code_to_float.transform,
                # genotype_to_phenotype=donothing,
                iters=500,
                pop_size=100,
                str_len=np.sum(parts),
                # str_len=3000,
                show_progress_each=20,
                # optimal_value=0,
                # termination_error_value=0,
                keep_history='full',
                minimization=True)

# model.set_strategy(crossover_oper='uniform_rankk',
#                     mutation_oper= 'custom_rate',
#                       tour_size_param=7,
#                       parents_num_param=3,
#                       mutation_rate_param=0.1,
#                    elitism_param=True)

model.set_strategy()

# model.set_strategy(crossover_opers=['uniform2',
#                                     'uniform7',
#                                     'uniform_prop2',
#                                     'uniform_prop7',
#                                     'uniform_rank2',
#                                     'uniform_rank7',
#                                     'uniform_tour3',
#                                     'uniform_tour7'])
model.fit()
stats = model.stats
print(stats.fitness)
print(stats.population_g)
# print(stats.m_proba)
print(model.thefittest.fitness)


for key, value in stats.c_proba.items():
    plt.plot(range(len(value)), value, label=key)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig('line12.png')
plt.close()
# print(stats.population_g.shape)
# print(model.get_remains_calls())
# print_population_by_time(stats.population_g, gray_code_to_float, problem)