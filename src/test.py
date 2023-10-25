import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE
from thefittest.optimizers import SelfCGA
from thefittest.benchmarks import BanknoteDataset
from thefittest.benchmarks import UserKnowladgeDataset
from thefittest.benchmarks import CreditRiskDataset
from thefittest.benchmarks import IrisDataset
from thefittest.benchmarks import DigitsDataset

# from thefittest.benchmarks import IrisDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.classifiers._gpnneclassifier import GeneticProgrammingNeuralNetStackingClassifier
from thefittest.tools.print import print_net, print_nets
from thefittest.tools.print import print_tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score


def print_ens(ens) -> None:
    nets = ens._nets
    n_nets = len(nets)
    print(n_nets)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    height = 1 / n_nets

    bottom_x = np.linspace(1 - height, 0, n_nets)

    for i, bottom in enumerate(bottom_x):
        ax = fig.add_axes([0.0, bottom, 0.5, height])
        ax.axis("off")
        print_net(nets[i], ax=ax)

    if ens._meta_algorithm is not None:
        ax = fig.add_axes([0.5, 0, 0.5, 1])
        ax.axis("off")
        print_net(ens._meta_algorithm, ax=ax)

    plt.show()


data = UserKnowladgeDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)

model = GeneticProgrammingNeuralNetStackingClassifier(
    iters=5,
    pop_size=100,
    optimizer=SelfCGP,
    input_block_size=1,
    optimizer_args={"show_progress_each": 1, "init_level": 4, "max_level": 16},
    weights_optimizer=SelfCGA,
    weights_optimizer_args={"iters": 300, "pop_size": 300, "no_increase_num": 50},
    test_sample_ratio=0.5,
)

model.fit(X_train, y_train)

predict = model.predict(X_test)
predict_train = model.predict(X_train)
optimizer = model.get_optimizer()

tree = optimizer.get_fittest()["genotype"]
ens = optimizer.get_fittest()["phenotype"]

print("confusion_matrix: \n", confusion_matrix(y_train, predict_train))
print("f1_score: \n", f1_score(y_train, predict_train))
print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict))

# fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=1)
# print_nets(nets=net._nets)
print_ens(ens)

# print_tree(tree=tree, ax=ax[1])

# plt.tight_layout()
# plt.show()
