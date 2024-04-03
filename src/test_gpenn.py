import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.classifiers._gpnneclassifier import GeneticProgrammingNeuralNetStackingClassifier
from thefittest.classifiers._gpnneclassifier import TwoTreeSelfCGP
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
    optimizer=TwoTreeSelfCGP,
    optimizer_args={"show_progress_each": 1},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 100, "pop_size": 100},
)

model.fit(X_train, y_train)

predict = model.predict(X_test)
optimizer = model.get_optimizer()

trees = optimizer.get_fittest()["genotype"]
nets = optimizer.get_fittest()["phenotype"]

print(nets._nets)
# print(trees._genotypes)
print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))

# fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=1)
print_nets(nets._nets)

plt.savefig("print_nets.png")
plt.close()
print_ens(nets)
# print_trees(trees._genotypes)
plt.savefig("print_ens.png")
plt.close()


# # После создания всех рисунков из print_nets, скомбинируем их с текущими ax
# for sub_ax in print_nets_fig.get_axes():
#     ax[1].figure.delaxes(sub_ax)
#     ax[1].figure.add_subplot(ax[1].get_subplotspec(), sharey=ax[1])
#     ax[1].figure.add_axes(sub_ax)

plt.tight_layout()
plt.show()
