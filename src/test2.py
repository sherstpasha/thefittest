import numpy as np

from thefittest.optimizers import SHADE, SHAGA, SelfCGA
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import MLPEAClassifier
from thefittest.regressors import MLPEARegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, r2_score
from sklearn.neural_network import MLPRegressor

# from thefittest.utils._metrics import coefficient_determination
from sklearn.utils.estimator_checks import check_estimator


# data = BanknoteDataset()
X = np.loadtxt("src/testX.py")
y = np.loadtxt("src/testy.py")

#X_scaled = minmax_scale(X)
# y = scale(X)
# X = minmax_scale(X)
# y = minmax_scale(y)

print(X.shape)
print(y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

model = MLPEARegressor(
    iters=100,
    pop_size=500,
    hidden_layers=(0,),
    activation="sigma",
    weights_optimizer=SHADE,
    # weights_optimizer_args={"show_progress_each": 1},
)

# model.fit(X, y)
# print(model.score(X, y))

# predict = model.predict(X)
# print(coefficient_determination(y, predict))
# print(r2_score(y, predict))

# model = MLPRegressor()

# model.fit(X, y)

# print(model.score(X, y))

# predict = model.predict(X)
# print(coefficient_determination(y, predict))
# print(r2_score(y, predict))


check_estimator(model)
