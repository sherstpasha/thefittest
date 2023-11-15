import numpy as np

from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset, DigitsDataset
from thefittest.classifiers import MLPEAClassifier
from thefittest.tools.print import print_net

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import time as time

import warnings

warnings.simplefilter("ignore")

data = DigitsDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

model = MLPEAClassifier(
    iters=200,
    pop_size=500,
    hidden_layers=(15,),
    activation="relu",
    weights_optimizer=SHADE,
    weights_optimizer_args={"show_progress_each": 1, "n_jobs": 4},
)

begin = time.time()
model.fit(X_train, y_train)
print(time.time() - begin)
predict = model.predict(X_test)
net = model.get_net()

print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))

print_net(net=net)  # the "4" node is offset
