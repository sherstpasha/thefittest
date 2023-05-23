
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from thefittest.tools.print import print_tree
from thefittest.tools.random import train_test_split_stratified
from thefittest.tools.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split_stratified(
    X, y, 0.3)

model = GeneticProgrammingNeuralNetClassifier(15, 50,
                                              show_progress_each=1,
                                              input_block_size=1,
                                              keep_history=True)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

fittest = model._optimizer.get_fittest()
genotype, phenotype, fitness = fittest.get()

fig, ax = plt.subplots(2)
print_tree(genotype, ax[0])
print_net(phenotype, ax[1])
fig.savefig('net.png')

stats = model._optimizer.get_stats()

for i in range(len(stats['fitness_max'])):
    print(stats['fitness_max'][i],  stats['individ_max'][i])
