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

model = GeneticProgrammingNeuralNetClassifier(
    iters=3,
    pop_size=10,
    optimizer=SelfCGP,
    optimizer_args={"show_progress_each": 1},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 30, "pop_size": 30},
)

model.fit(X_train, y_train)

predict = model.predict(X_test)
optimizer = model.get_optimizer()

tree = optimizer.get_fittest()["genotype"]
net = optimizer.get_fittest()["phenotype"]


print(net)

net.save_to_file("net.pkl")