import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset, IrisDataset
from thefittest.classifiers._gpnnclassifier import GeneticProgrammingNeuralNetClassifier2
from thefittest.regressors._gpnnregression import GeneticProgrammingNeuralNetRegressor2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, r2_score
from sklearn.utils.estimator_checks import check_estimator


# data = IrisDataset()
# X = data.get_X()
# y = data.get_y()

# X_scaled = minmax_scale(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# model = GeneticProgrammingNeuralNetClassifier2(
#     n_iter=10,
#     pop_size=50,
#     optimizer=SelfCGP,
#     optimizer_args={
#         "show_progress_each": 1,
#         #  "n_jobs": 1
#     },
#     weights_optimizer=SHADE,
#     weights_optimizer_args={
#         "iters": 100,
#         "pop_size": 100,
#     },
# )

# # check_estimator(model)


# import time


# begin = time.time()
# model.fit(X_train, y_train)
# print(time.time() - begin)


# predict = model.predict(X_test)


# print("confusion_matrix: \n", confusion_matrix(y_test, predict))
# print("f1_score: \n", f1_score(y_test, predict, average="macro"))


def problem(x):
    return np.sin(x[:, 0])


function = problem
left_border = -4.5
right_border = 4.5
sample_size = 300
n_dimension = 1

X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
y = function(X)
X_scaled = minmax_scale(X)
y_scaled = minmax_scale(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.33)

model = GeneticProgrammingNeuralNetRegressor2(
    n_iter=5,
    pop_size=15,
    optimizer=SelfCGP,
    optimizer_args={"show_progress_each": 1, "n_jobs": 2},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 100, "pop_size": 100},
)


# check_estimator(model)

model.fit(X_train, y_train)

predict = model.predict(X_test)

print("coefficient_determination: \n", r2_score(y_test, predict))
