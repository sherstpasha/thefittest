import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.tools.print import print_net
from thefittest.tools.print import print_tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from thefittest.tools.random import random_tree


data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
n_outputs = len(set(y))
eye = np.eye(n_outputs, dtype=np.float64)
proba_train = eye[y_train]
proba_test = eye[y_test]

counter = []

model = GeneticProgrammingNeuralNetClassifier(
    iters=15,
    pop_size=50,
    optimizer=SelfCGP,
    optimizer_args={"show_progress_each": 1},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 25, "pop_size": 25},
    cache=True,
)

tree = random_tree(model._get_uniset(X_train), max_level=5)
net = model._genotype_to_phenotype(X_train, proba_train, [tree], n_outputs)[0]
