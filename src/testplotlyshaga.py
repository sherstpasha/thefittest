import numpy as np
import pandas as pd

from thefittest.optimizers import GeneticAlgorithm, SelfCGA
from thefittest.tools.transformations import GrayCode
from thefittest.tools.transformations import donothing
from thefittest.benchmarks import CEC2005
from thefittest.benchmarks import Sphere
import matplotlib.pyplot as plt
import networkx as nx


n_variables = 10

left = np.full(n_variables, -100, dtype=np.float64)
right = np.full(n_variables, 100, dtype=np.float64)
parts = np.full(n_variables, 63, dtype=np.int64)

gray_code_to_float = GrayCode(fit_by='parts').fit(
    left=left, right=right, arg=parts)



def func(x):
    return np.sum(x, axis=1)


problem = Sphere()

model = SelfCGA(fitness_function=problem,
                genotype_to_phenotype=gray_code_to_float.transform,
                iters=1500,
                pop_size=100,
                str_len=np.sum(parts),
                show_progress_each=1,
                keep_history='quick',
                minimization=True)



model.fit()

stats = model.stats

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=3)


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



plt.tight_layout()
plt.savefig('line1.png')
plt.close()

