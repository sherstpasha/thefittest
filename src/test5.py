
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.tools.print import print_tree
from thefittest.tools.random import train_test_split_stratified
from thefittest.tools.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score
from thefittest.optimizers import SHADE
from thefittest.optimizers import SelfCGA
from thefittest.benchmarks import WineDataset
from thefittest.tools.transformations import scale_data
from thefittest.classifiers import MLPClassifierEA
from thefittest.tools.metrics import categorical_crossentropy3d


if __name__ == '__main__':
    data = WineDataset()
    # print(data.feature_names)
    X = data.get_X()
    y = data.get_y()
    X = scale_data(X)

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, 0.3)
    
    model = MLPClassifierEA(iters=100, pop_size=100, activation='tanh',
                        hidden_layers=(10, 10, 10), offset=True,
                        show_progress_each=1, keep_history=True,
                        optimizer_weights=SHADE)
    
    model.fit(X_train, y_train)

    net = model._net


    optimizer = SelfCGA
    net.train_weights(, )
    
    
