import plotly.express as px
import plotly
import plotly.graph_objects as go


import numpy as np
import pandas as pd


from thefittest.testfuncs._problems import *
from thefittest.testfuncs import CEC2005
from thefittest.optimizers import SHADE, JADE


def print_population_by_time(population_3d, grid_model, function, left, right):

    array3d = np.array(list(map(donothing, population_3d)))

    array2d = array3d.reshape(-1, 2)

    index = np.repeat(np.arange(array3d.shape[0]), array3d.shape[1])

    fx = function(array2d)
    data = pd.DataFrame({'x': array2d[:, 0],
                         'y': array2d[:, 1],
                         'z':  fx,
                         'i': index,
                         'inv_z': -fx})

    # xlim = (grid_model.left[0], grid_model.right[0])
    # ylim = (grid_model.left[1], grid_model.right[1])
    xlim = (left[0], right[0])
    ylim = (left[1], right[1])
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


# problem = CEC2005.problems_dict['F1']['function']()
problem = Sphere()
n_var = 30

left = np.full(n_var, -100)
right = np.full(n_var, 100)


def donothing(x):
    return x


model = SHADE(fitness_function=problem,
             genotype_to_phenotype=donothing,
             left=left,
             right=right,
             iters=1500,
             pop_size=100,
             minimization=True,
             show_progress_each=20,
             keep_history=True)


model.fit()
# print(model.sign*model.thefittest.fitness - CEC2005.problems_dict['F1']['optimum'])
# stats = model.stats

# print(stats.H_CR)
# print(stats.H_F)
# print_population_by_time(stats.population_g, donothing, problem, left=(
#     left[0], left[1]), right=(right[0], right[1]))