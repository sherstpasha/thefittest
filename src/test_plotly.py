import plotly.express as px
import plotly
import plotly.graph_objects as go


import numpy as np
import pandas as pd

from thefittest.optimizers import SelfCGA
from thefittest.testfuncs import Rastrigin
from thefittest.testfuncs import CEC2005
from thefittest.tools import GrayCode

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
    x = np.linspace(xlim[0], xlim[1], 500)
    y = np.linspace(ylim[0], ylim[1], 500)
    z = function.build_grid(x, y)
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

n_variables = 2

left = np.full(n_variables, -5, dtype=np.float64)
right = np.full(n_variables, 5, dtype=np.float64)
# right = np.array([-100, -40], dtype = np.float64)
# left = np.array([-50, 0], dtype = np.float64)
parts = np.full(n_variables, 16, dtype=np.int64)

gray_code_to_float = GrayCode(fit_by='parts').fit(left=left, right=right, arg=parts)

problem = CEC2005.RotatedHybridCompositionFunctionNarrowBasin()
model = SelfCGA(fitness_function = problem,
                genotype_to_phenotype = gray_code_to_float.transform,
                iters = 300,
                pop_size = 300,
                str_len = np.sum(parts),
                show_progress_each = 10,
                keep_history = True,
                minimization=True)

solution = model.fit()
stats = model.stats
print(solution.fitness)

print_population_by_time(stats.population, gray_code_to_float, problem)