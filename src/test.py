import numpy as np

from thefittest.optimizers import SHADE
from thefittest.benchmarks import DigitsDataset, BreastCancerDataset, BanknoteDataset, IrisDataset
from thefittest.classifiers import MLPEAClassifier
from thefittest.tools.print import print_net

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

import time

data = IrisDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)


h_predict = []
h_predict2 = []

for _ in range(1):
    model = MLPEAClassifier(
        iters=500,
        pop_size=500,
        hidden_layers=(10,),
        activation="sigma",
        weights_optimizer=SHADE,
        weights_optimizer_args={
            "show_progress_each": None,
            "n_jobs": 1,
            "keep_history": True,
            "show_progress_each": 1,
        },
    )
    print(_)

    t1 = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(t2 - t1)

    predict = model.predict(X_test)

    # h_predict.append(f1_score(y_test, predict, average="macro"))
    print("confusion_matrix: \n", confusion_matrix(y_test, predict))
    print("f1_score: \n", f1_score(y_test, predict, average="macro"))

# print(np.mean(h_predict))
# print(np.mean(h_predict2))
# net = model.get_net()

# print("confusion_matrix: \n", confusion_matrix(y_test, predict))
# print("f1_score: \n", f1_score(y_test, predict, average="macro"))
# print("confusion_matrix: \n", confusion_matrix(y_test, predict2))
# print("f1_score: \n", f1_score(y_test, predict2, average="macro"))

# stats = model.get_optimizer().get_stats()

# print(stats.keys())

# plt.plot(range(len(stats["test_error"])), (-1) * np.array(stats["test_error"]), label="test_error")
# plt.plot(
#     range(len(stats["max_fitness"])), (-1) * np.array(stats["max_fitness"]), label="train_error"
# )
# plt.legend()
# plt.show()
# print_net(net=net)  # the "4" node is offset
