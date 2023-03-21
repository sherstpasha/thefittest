import numpy as np
from thefittest.optimizers import GeneticAlgorithm, SelfCGA
from thefittest.benchmarks._optproblems import *
from thefittest.tools.transformations import GrayCode
from thefittest.tools.transformations import donothing
import matplotlib.pyplot as plt


n_variables = 30

left = np.full(n_variables, -100, dtype=np.float64)
right = np.full(n_variables, 100, dtype=np.float64)
parts = np.full(n_variables, 63, dtype=np.int64)

gray_code_to_float = GrayCode(fit_by='parts').fit(
    left=left, right=right, arg=parts)


problem = Ackley()

model = SelfCGA(fitness_function=problem,
                genotype_to_phenotype=gray_code_to_float.transform,
                iters=1500,
                pop_size=100,
                str_len=np.sum(parts),
                show_progress_each=5,
                keep_history='quick',
                minimization=True)

model.set_strategy(mutation_opers=['weak',
                                   'average',
                                   'strong',
                                   'fitness_based_adaptation'])

model.fit()

stats = model.stats
fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=3)


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
