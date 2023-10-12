import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE
from thefittest.benchmarks import DigitsDataset, IrisDataset, WineDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.classifiers._gpnneclassifier import GeneticProgrammingNeuralNetEnsemblesClassifier
from thefittest.tools.print import print_nets
from thefittest.tools.print import print_tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


data = DigitsDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33)

counter = []

for i in range(1):
    model = GeneticProgrammingNeuralNetEnsemblesClassifier(
        iters=10,
        pop_size=15,
        input_block_size=1,
        optimizer=SelfCGP,
        optimizer_args={"show_progress_each": 1},
        weights_optimizer=SHADE,
        weights_optimizer_args={"iters": 100, "pop_size": 100},
        cache=True,
    )

    model.fit(X_train, y_train)

optimizer = model.get_optimizer()

tree = optimizer.get_fittest()["genotype"]
ensemble = optimizer.get_fittest()["phenotype"]
nets = ensemble._nets

predict = model.predict(X_test)
print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))


print_nets(nets=nets)

plt.tight_layout()
plt.show()
