import matplotlib.pyplot as plt
import numpy as np

from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import MLPEAClassifier

# from thefittest.tools.print import print_net

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from thefittest.regressors import MLPEARegressor
from thefittest.optimizers import SHAGA

from sklearn.metrics import r2_score


def problem(x):
    return np.sin(x[:, 0])


function = problem
left_border = -4.5
right_border = 4.5
sample_size = 200
n_dimension = 1

X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
y = function(X)

X_scaled = minmax_scale(X)
y_scaled = minmax_scale(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.33)

model = MLPEARegressor(
    n_iter=500,
    pop_size=250,
    hidden_layers=(5, 5),
    activation="sigma",
    weights_optimizer=SHAGA,
    weights_optimizer_args={"show_progress_each": 50},
)

model.fit(X_train, y_train)

predict = model.predict(X_test)
net = model.get_net()

print("coefficient_determination: \n", r2_score(y_test, predict))

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=1)

ax[0].plot(X_scaled[:, 0], y_scaled, label="True y")
ax[0].scatter(X_test[:, 0], predict, label="Predict y")
ax[0].legend()

# print_net(net = net, ax = ax[1])

plt.tight_layout()
plt.show()
