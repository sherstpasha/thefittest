import numpy as np
from thefittest.optimizers import SelfCGA
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.transformations import GrayCode
from thefittest.benchmarks import Weierstrass

n_dimension = 30
left_border = -2.
right_border = 2.
n_bits_per_variable = 32

number_of_iterations = 100
population_size = 500

left_border_array = np.full(
    shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(
    shape=n_dimension, fill_value=right_border, dtype=np.float64)
parts = np.full(
    shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

genotype_to_phenotype = GrayCode(fit_by='parts').fit(left=left_border_array,
                                                     right=right_border_array,
                                                     arg=parts)
model = SelfCGA(fitness_function=Weierstrass(),
                genotype_to_phenotype=genotype_to_phenotype.transform,
                iters=number_of_iterations,
                pop_size=population_size,
                str_len=sum(parts),
                show_progress_each=1,
                minimization=True,
                keep_history=True)

model.fit()

solution = model.get_fittest()
stats = model.get_stats()

# print('The fittest individ:', model.thefittest.phenotype)
# print('with fitness', model.thefittest.fitness)

import plotly.graph_objects as go
import plotly.offline as pyo


data = {'index': list(range(population_size)),
        'fitness_max': stats['fitness_max']}
operators_types = ['selection', 'crossover', 'mutation']
total = 1 + len(model._selection_set) + len(model._crossover_set) + len(model._mutation_set)

for s_proba, c_proba, m_proba in zip(stats['s_proba'], stats['c_proba'], stats['m_proba']):
    for proba, oper_type in zip([s_proba, c_proba, m_proba],
                                operators_types):
        if oper_type not in data.keys():
            data[oper_type] = {}
        for key, value in proba.items():
            if key not in data[oper_type].keys():
                data[oper_type][key] = [value]
            else:
                data[oper_type][key].append(value)

# tracers and visible parametr
tracers = [go.Scatter(x=data['index'],
                      y=data['fitness_max'], name='fitness_max')]
visible_condition = {'fitness_max': [True] + [False]*(total - 1)}
tracer_id = 1
for oper_type in operators_types:
    visible_condition[oper_type] = [False]*total
    for key, value in data[oper_type].items():
        tracers.append(go.Scatter(
            x=data['index'], y=data[oper_type][key], name=key, visible=False))
        visible_condition[oper_type][tracer_id] = True
        tracer_id += 1

# frames for animations
frames = []
for k in range(1, population_size-1):
    data_arg = [dict(type='scatter',
                     x=data['index'][:k+1],
                     y=data['fitness_max'][:k+1])]
    for oper_type in operators_types:
        for key, value in data[oper_type].items():
            data_arg.append(dict(type='scatter',
                                 x=data['index'][:k+1],
                                 y=value[:k+1]))
    traces_arg = list(range(total))
    frames.append(dict(data=data_arg, traces=traces_arg))

# buttons for layouts
buttons = [dict(label='Play',
                      method='animate',
                      args=[None,
                            dict(frame=dict(duration=3,
                                            redraw=False),
                                 transition=dict(duration=0),
                                 fromcurrent=True,
                                 mode='immediate')])]
buttons_names = ['fitness_max'] + operators_types
for button_name in buttons_names:
    button = dict(label=button_name,
                  method="update",
                  args=[{"visible": visible_condition[button_name]},
                        {"title": button_name,
                         "annotations": []}])
    buttons.append(button)

# layout
layout = go.Layout(updatemenus=[dict(type="buttons",
                                     direction="right",
                                     active=1,
                                     x=0.57,
                                     y=1.2,
                                     buttons=buttons)])

fig = go.Figure(data=tracers, frames=frames,  layout=layout)
fig.write_html('fig1.html')