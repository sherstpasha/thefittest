import numpy as np

from thefittest.optimizers import SHADE, SHAGA
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import MLPEAClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3)

model = MLPEAClassifier(iters=500,
                        pop_size=250,
                        hidden_layers=(5,),
                        activation="relu",
                        weights_optimizer=SHAGA,
                        weights_optimizer_args={"show_progress_each": 50})

model.fit(X_train, y_train)