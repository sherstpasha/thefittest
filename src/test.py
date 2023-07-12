
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.tools.print import print_tree
from thefittest.tools.random import train_test_split_stratified
from thefittest.tools.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score
from thefittest.optimizers import SHADE
from thefittest.benchmarks import IrisDataset
from thefittest.tools.transformations import scale_data


if __name__ == '__main__':
    data = IrisDataset()
    # print(data.feature_names)
    X = data.get_X()
    y = data.get_y()
    X = scale_data(X)


    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, 0.3)

    model = GeneticProgrammingNeuralNetClassifier(50, 15,
                                                show_progress_each=1,
                                                input_block_size=1,
                                                max_hidden_block_size=5,
                                                keep_history=True,
                                                optimizer_weights=SHADE,
                                                optimizer_weights_eval_num=10000,
                                                cache=True)
    import time
    begin = time.time()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    fittest = model._optimizer.get_fittest()
    genotype, phenotype, fitness = fittest.get()

    fig, ax = plt.subplots(2)
    print_tree(genotype, ax[0])
    print_net(phenotype, ax[1])
    plt.tight_layout()
    fig.savefig('net.png')

    stats = model._optimizer.get_stats()

    showed = set()
    for i in range(len(stats['fitness_max'])):
        if stats['fitness_max'][i] not in showed:
            print(stats['fitness_max'][i],  stats['individ_max'][i])
            showed.add(stats['fitness_max'][i])

    print(model._optimizer._calls)