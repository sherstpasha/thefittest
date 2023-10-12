import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE
from thefittest.benchmarks import DigitsDataset
from thefittest.benchmarks import IrisDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.classifiers._gpnneclassifier import GeneticProgrammingNeuralNetEnsemblesClassifier
from thefittest.tools.print import print_nets
from thefittest.tools.print import print_tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from thefittest.tools.random import random_tree


data = IrisDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
n_outputs = len(set(y))
eye = np.eye(n_outputs, dtype=np.float64)
proba_train = eye[y_train]
proba_test = eye[y_test]

model = GeneticProgrammingNeuralNetEnsemblesClassifier(
    iters=15,
    pop_size=50,
    optimizer=SelfCGP,
    optimizer_args={"show_progress_each": 1},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 200, "pop_size": 200, "show_progress_each": 1},
    cache=True,
)

tree = random_tree(model._get_uniset(X_train), max_level=5)

pop = model._genotype_to_phenotype(X_train, proba_train, [tree], n_outputs)

# model._train_all_ensemble(pop[0], X_train, proba_train)

predict = pop[0].average_output_classifier(X_test)[0]

print(predict.shape)

print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))


print_nets(nets=pop[0]._nets)

plt.tight_layout()
plt.show()
