
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.regressors import MLPRegressorrEA
from thefittest.tools.print import print_tree
from thefittest.tools.random import train_test_split
from thefittest.tools.metrics import confusion_matrix
from thefittest.tools.metrics import coefficient_determination
from thefittest.optimizers import SHADE
from thefittest.benchmarks import IrisDataset
from thefittest.tools.transformations import scale_data

from sklearn.datasets import load_diabetes

if __name__ == '__main__':
    data = load_diabetes()
    # print(data.feature_names)
    X = data.data
    y = data.target
    X = scale_data(X)
    y = scale_data(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 0.3)

    model = MLPRegressorrEA(iters=500, pop_size=500, activation='sigma', output_activation='sigma',
                            hidden_layers=(32,), offset=True,
                            show_progress_each=1, keep_history=True,
                            optimizer_weights=SHADE)
    import time
    begin = time.time()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('coefficient_determination', coefficient_determination(y_test, y_pred))

    # fittest = model._optimizer.get_fittest()
    # genotype, phenotype, fitness = fittest.get()

    # fig, ax = plt.subplots(2)
    # print_tree(genotype, ax[0])
    # print_net(phenotype, ax[1])
    # plt.tight_layout()
    # fig.savefig('net.png')

    # stats = model._optimizer.get_stats()

    # showed = set()
    # for i in range(len(stats['fitness_max'])):
    #     if stats['fitness_max'][i] not in showed:
    #         print(stats['fitness_max'][i],  stats['individ_max'][i])
    #         showed.add(stats['fitness_max'][i])

    # print(model._optimizer._calls)
