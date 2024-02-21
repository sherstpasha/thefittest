from thefittest.optimizers import SHADE
from sklearn.utils.estimator_checks import check_estimator

import numpy as np

from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset, IrisDataset, DigitsDataset
from thefittest.classifiers import MLPEAClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

model = MLPEAClassifier(
    iters=100,
    pop_size=100,
    hidden_layers=(5,),
    activation="relu",
    weights_optimizer=SHADE,
    random_state=42,
    # weights_optimizer_args={"show_progress_each": 1},
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(y_pred)
print(f1_score(y_test, y_pred, average="macro"))

model = MLPEAClassifier(
    iters=100,
    pop_size=100,
    hidden_layers=(5,),
    activation="relu",
    weights_optimizer=SHADE,
    random_state=42,
    # weights_optimizer_args={"show_progress_each": 1},
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(y_pred)
print(f1_score(y_test, y_pred, average="macro"))


# print(y_test)


# import pickle

# pickle.dumps(model)

print(check_estimator(model))
