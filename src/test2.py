import matplotlib.pyplot as plt
import numpy as np
from thefittest.tools.print import print_nets
import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset
from thefittest.benchmarks import UserKnowladgeDataset
from thefittest.benchmarks import CreditRiskDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.classifiers._gpnneclassifier import GeneticProgrammingNeuralNetStackingClassifier
from thefittest.tools.print import print_net, print_nets
from thefittest.tools.print import print_tree
from thefittest.tools.random import random_tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score


def print_nets(ens) -> None:
    nets = ens._nets
    n_nets = len(nets)
    print(n_nets)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    height = 1 / n_nets

    bottom_x = np.linspace(1 - height, 0, n_nets)
    print(bottom_x)

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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33)

model = GeneticProgrammingNeuralNetStackingClassifier(
    iters=15,
    pop_size=50,
    optimizer=SelfCGP,
    input_block_size=1,
    optimizer_args={"show_progress_each": 1},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 100, "pop_size": 100},
    test_sample_ratio=0.3,
)


tree = random_tree(model._get_uniset(X_train), max_level=5)

ens = model._genotype_to_phenotype_ensemble(X.shape[1], len(set(y)), tree)

print(ens)


print_nets(ens)
