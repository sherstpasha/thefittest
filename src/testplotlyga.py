import numpy as np
import pandas as pd

from thefittest.optimizers import GeneticAlgorithm
from thefittest.benchmarks._optproblems import *
from thefittest.tools.transformations import GrayCode
from thefittest.tools.transformations import donothing

n_variables = 30

left = np.full(n_variables, -100, dtype=np.float64)
right = np.full(n_variables, 100, dtype=np.float64)
parts = np.full(n_variables, 16, dtype=np.int64)

gray_code_to_float = GrayCode(fit_by='parts').fit(
    left=left, right=right, arg=parts)


problem = Sphere()

model = GeneticAlgorithm(fitness_function=problem,
                genotype_to_phenotype=gray_code_to_float.transform,
                iters=100,
                pop_size=100,
                str_len=np.sum(parts),
                show_progress_each=5,
                keep_history='full',
                minimization=True)

model.fit()