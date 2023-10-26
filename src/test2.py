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
import time

data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

model = GeneticProgrammingNeuralNetClassifier(
    iters=5,
    pop_size=20,
    optimizer=SelfCGP,
    optimizer_args={"show_progress_each": 1},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 300, "pop_size": 300},
)

begin = time.time()
model.fit(X_train, y_train)
print(time.time() - begin)

predict = model.predict(X_test)
optimizer = model.get_optimizer()

tree = optimizer.get_fittest()["genotype"]
net = optimizer.get_fittest()["phenotype"]

print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=1)
print_net(net=net, ax=ax[0])

print_tree(tree=tree, ax=ax[1])

plt.tight_layout()
plt.show()
