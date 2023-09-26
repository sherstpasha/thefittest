from ..benchmarks import IrisDataset
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..classifiers import MLPEAClassifier
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGA
from ..tools.transformations import scale_data
from ..base._net import Net


def test_GeneticProgrammingNeuralNetClassifier():
    data = IrisDataset()
    X = data.get_X()
    y = data.get_y()

    data = IrisDataset()
    X = scale_data(data.get_X())
    y = data.get_y()

    optimizer = GeneticProgramming

    optimizer_args = {"tour_size": 15, "show_progress_each": 1}

    iters = 3
    pop_size = 10

    weights_optimizer = DifferentialEvolution
    weights_optimizer_args = {"iters": 25, "pop_size": 25, "CR": 0.9}

    model = GeneticProgrammingNeuralNetClassifier(
        iters=iters,
        pop_size=pop_size,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    weights_optimizer = SelfCGA
    weights_optimizer_args = {"iters": 25, "pop_size": 25, "K": 0.33}

    model = GeneticProgrammingNeuralNetClassifier(
        iters=iters,
        pop_size=pop_size,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    model = GeneticProgrammingNeuralNetClassifier(
        iters=iters,
        pop_size=pop_size,
    )

    model.fit(X, y)

    model.predict(X)

    optimizer = model.get_optimizer()

    net = model.get_net()

    assert isinstance(net, Net)
    assert isinstance(optimizer, model._optimizer_class)


def test_MLPEAClassifier():
    data = IrisDataset()
    X = data.get_X()
    y = data.get_y()

    data = IrisDataset()
    X = scale_data(data.get_X())
    y = data.get_y()

    iters = 3
    pop_size = 10

    weights_optimizer = DifferentialEvolution
    weights_optimizer_args = {"CR": 0.9}

    model = MLPEAClassifier(
        iters=iters,
        pop_size=pop_size,
        hidden_layers=(10,),
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    weights_optimizer = SelfCGA
    weights_optimizer_args = {"K": 0.33}

    model = MLPEAClassifier(
        iters=iters,
        pop_size=pop_size,
        hidden_layers=(1, 2),
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    model = MLPEAClassifier(
        iters=iters,
        pop_size=pop_size,
        hidden_layers=(0,),
    )

    model.fit(X, y)

    model.predict(X)

    optimizer = model.get_optimizer()
    net = model.get_net()

    assert isinstance(net, Net)
    assert isinstance(optimizer, model._weights_optimizer_class)
