import numpy as np


def problem(x):
    return np.sin(x[:, 0] * 3) * x[:, 0] * 0.5 + np.cos(x[:, 1] * 2) * x[:, 1] * 0.3


import matplotlib.pyplot as plt

from thefittest.regressors import GeneticProgrammingRegressor
from thefittest.optimizers._cshagp import CSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import PDPGP


n_dimension = 2
left_border = -4.5
right_border = 4.5
sample_size = 100


X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
y = problem(X)

model = GeneticProgrammingRegressor(
    n_iter=500,
    pop_size=1000,
    optimizer=PDPSHAGP,
    optimizer_args={"show_progress_each": 10},
)

model.fit(X, y)
predict = model.predict(X)

tree = model.get_tree()
print("The fittest individ:", tree)

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=1)

ax[0].plot(X[:, 0], y, label="True y")
ax[0].plot(X[:, 0], predict, label="Predict y")
ax[0].legend()

tree.plot(ax=ax[1])

plt.tight_layout()
plt.show()
