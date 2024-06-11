import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE, SelfCGA
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.classifiers._gpnneclassifier_one_tree import (
    GeneticProgrammingNeuralNetStackingClassifier,
)
from thefittest.tools.print import print_net
from thefittest.tools.print import print_tree
from thefittest.tools.print import print_nets
from thefittest.tools.print import print_trees
from thefittest.tools.print import print_ens


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

model = GeneticProgrammingNeuralNetStackingClassifier(
    iters=3,
    pop_size=10,
    optimizer=SelfCGP,
    optimizer_args={"show_progress_each": 1},
    weights_optimizer=SelfCGA,
    weights_optimizer_args={"iters": 25, "pop_size": 25, "no_increase_num": 1000},
)

model.fit(X_train, y_train)

predict = model.predict(X_test)
optimizer = model.get_optimizer()

common_tree = optimizer.get_fittest()["genotype"]
ens = optimizer.get_fittest()["phenotype"]
trees = ens._trees

model._optimizer = None

model.save_to_file("model.pkl")

# print("confusion_matrix: \n", confusion_matrix(y_test, predict))
# print("f1_score: \n", f1_score(y_test, predict, average="macro"))

# print_tree(common_tree)
# plt.savefig("1_common_tree.png")
# plt.close()

# print_trees(trees)
# plt.savefig("2_trees.png")
# plt.close()

# print_nets(ens._nets)
# plt.savefig("3_nets.png")
# plt.close()

# print_net(ens._meta_algorithm)
# plt.savefig("4_meta_net.png")
# plt.close()

# print_ens(ens)
# plt.savefig("5_ens.png")
# plt.close()

