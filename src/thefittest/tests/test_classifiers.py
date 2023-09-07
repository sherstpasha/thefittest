from ..benchmarks import IrisDataset
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..classifiers import MLPClassifierEA
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import SelfCGP
from ..tools.transformations import scale_data


def test_GeneticProgrammingNeuralNetClassifier():

    data = IrisDataset()
    X = data.get_X()
    y = data.get_y()
    X_scaled = scale_data(X)

    iters = 5
    pop_size = 15
    input_block_size = 1
    max_hidden_block_size = 5
    offset = True
    output_activation = 'softmax'
    test_sample_ratio = 0.5
    no_increase_num = None
    show_progress_each = 1
    keep_history = False
    optimizer = SelfCGP
    optimizer_weights = DifferentialEvolution
    optimizer_weights_eval_num = 2500
    optimizer_weights_n_bit = 16

    model = GeneticProgrammingNeuralNetClassifier(
        iters=iters,
        pop_size=pop_size,
        input_block_size=input_block_size,
        max_hidden_block_size=max_hidden_block_size,
        offset=offset,
        output_activation=output_activation,
        test_sample_ratio=test_sample_ratio,
        no_increase_num=no_increase_num,
        show_progress_each=show_progress_each,
        keep_history=keep_history,
        optimizer=optimizer,
        optimizer_weights=optimizer_weights,
        optimizer_weights_eval_num=optimizer_weights_eval_num,
        optimizer_weights_n_bit=optimizer_weights_n_bit)

    assert isinstance(model.optimizer, SelfCGP)
    assert isinstance(model.optimizer_weights, DifferentialEvolution)
    assert model.optimizer._iters == iters
    assert model.optimizer._pop_size == pop_size
    assert model._input_block_size == input_block_size
    assert model._max_hidden_block_size == max_hidden_block_size
    assert model._offset == offset
    assert model._output_activation == output_activation
    assert model._test_sample_ratio == test_sample_ratio
    assert model.optimizer._no_increase_num == no_increase_num
    assert model.optimizer._show_progress_each == show_progress_each
    assert model.optimizer._keep_history == keep_history

    model.optimizer.set_strategy(max_level_param=10)
    assert model.optimizer._max_level == 10

    model.optimizer_weights.set_strategy(CR_param=0.9)
    assert model.optimizer_weights._CR == 0.9

    model.fit(X_scaled, y)

    predict = model.predict(X_scaled)

    assert len(predict) == len(X_scaled)

    model = GeneticProgrammingNeuralNetClassifier(
        iters=iters,
        pop_size=pop_size,
        input_block_size=input_block_size,
        max_hidden_block_size=max_hidden_block_size,
        offset=offset,
        output_activation=output_activation,
        test_sample_ratio=test_sample_ratio,
        no_increase_num=no_increase_num,
        show_progress_each=show_progress_each,
        keep_history=keep_history,
        optimizer=optimizer,
        optimizer_weights=GeneticAlgorithm,
        optimizer_weights_eval_num=optimizer_weights_eval_num,
        optimizer_weights_n_bit=optimizer_weights_n_bit)

    assert isinstance(model.optimizer_weights, GeneticAlgorithm)

    model.optimizer_weights.set_strategy(tour_size_param=7)
    assert model.optimizer_weights._tour_size == 7

    model.fit(X, y)

    predict = model.predict(X_scaled)

    assert len(predict) == len(X_scaled)

    model = GeneticProgrammingNeuralNetClassifier(
        iters=iters,
        pop_size=pop_size,
        input_block_size=input_block_size,
        max_hidden_block_size=max_hidden_block_size,
        offset=False,
        output_activation=output_activation,
        test_sample_ratio=test_sample_ratio,
        no_increase_num=no_increase_num,
        show_progress_each=show_progress_each,
        keep_history=keep_history,
        optimizer=optimizer,
        optimizer_weights=GeneticAlgorithm,
        optimizer_weights_eval_num=optimizer_weights_eval_num,
        optimizer_weights_n_bit=optimizer_weights_n_bit)

    assert isinstance(model.optimizer_weights, GeneticAlgorithm)

    model.optimizer_weights.set_strategy(tour_size_param=7)
    assert model.optimizer_weights._tour_size == 7

    model.fit(X, y)


def test_MLPClassifierEA():
    data = IrisDataset()
    X = data.get_X()
    y = data.get_y()
    X_scaled = scale_data(X)

    iters = 25
    pop_size = 25
    hidden_layers = (100, 10)
    activation = 'relu'
    output_activation = 'softmax'
    offset = True
    show_progress_each = 1
    no_increase_num = 30
    keep_history = True
    optimizer_weights = GeneticAlgorithm
    optimizer_weights_bounds = (-10, 10)
    optimizer_weights_n_bit = 16

    model = MLPClassifierEA(iters=iters,
                            pop_size=pop_size,
                            hidden_layers=hidden_layers,
                            activation=activation,
                            output_activation=output_activation,
                            offset=offset,
                            no_increase_num=no_increase_num,
                            show_progress_each=show_progress_each,
                            keep_history=keep_history,
                            optimizer_weights=optimizer_weights,
                            optimizer_weights_bounds=optimizer_weights_bounds,
                            optimizer_weights_n_bit=optimizer_weights_n_bit)

    model.optimizer_weights.set_strategy(tour_size_param=7)
    assert model.optimizer_weights._tour_size == 7

    model.fit(X_scaled, y)

    predict = model.predict(X_scaled)

    assert len(predict) == len(X_scaled)

    model = MLPClassifierEA(iters=iters,
                            pop_size=pop_size,
                            hidden_layers=hidden_layers,
                            activation=activation,
                            output_activation=output_activation,
                            offset=False,
                            no_increase_num=no_increase_num,
                            show_progress_each=show_progress_each,
                            keep_history=keep_history,
                            optimizer_weights=DifferentialEvolution,
                            optimizer_weights_bounds=optimizer_weights_bounds,
                            optimizer_weights_n_bit=optimizer_weights_n_bit)

    model.optimizer_weights.set_strategy(CR_param=0.9)
    assert model.optimizer_weights._CR == 0.9

    model.fit(X_scaled, y)

    predict = model.predict(X_scaled)

    assert len(predict) == len(X_scaled)
