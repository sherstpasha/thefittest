import numpy as np

from thefittest.regressors import GeneticProgrammingRegressor
from thefittest.optimizers import GeneticProgramming, SelfCGP

from sklearn.metrics import f1_score, r2_score

# from thefittest.benchmarks import BanknoteDataset
from collections import defaultdict

import matplotlib.pyplot as plt

from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_diabetes


def problem(x):
    return np.sin(x[:, 0])


data = load_diabetes()

X = data.data
y = data.target


number_of_iterations = 200

model = GeneticProgrammingRegressor(
    n_iter=number_of_iterations,
    pop_size=500,
    optimizer=SelfCGP,
    optimizer_args={
        "keep_history": True,
        "show_progress_each": 10,
        "elitism": True,
    },
)

check_estimator(model)

# model.fit(X, y)

# predict = model.predict(X)

# print(r2_score(y, predict))
