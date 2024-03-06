from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import Tree
from ..base import UniversalSet
from ..base import Net
from ..base._net import ACTIV_NAME_INV
from ..base._tree import UniversalSet
from ..base._tree import init_net_uniset
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..utils._metrics import categorical_crossentropy3d
from ..utils._metrics import root_mean_square_error2d
from ..utils.transformations import GrayCode
from ..utils.random import check_random_state


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


class Model:
    def _fit(
        self,
        X: np.typing.NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> Any:
        pass

    def _predict(self, X: NDArray[np.float64]) -> Any:
        pass

    def get_optimizer(
        self: Model,
    ) -> Any:
        pass

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> Any:
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))
        return self._fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        return self._predict(X)


def fitness_function_weights(
    weights: NDArray[np.float64],
    net: "Net",
    X: NDArray[np.float64],
    targets: NDArray[np.float64],
    task_type: str = "regression",
) -> NDArray[np.float64]:
    """
    Evaluate the fitness of a neural network's weights for a given task.

    This function computes the error between the network's predictions and the actual targets. It supports both classification and regression tasks, determined by the `task_type` parameter.

    Parameters
    ----------
    weights : NDArray[np.float64]
        The weights of the neural network to be evaluated.
    net : Net
        The neural network instance. It must have a `forward` method that accepts input data and weights, and returns the network's output.
    X : NDArray[np.float64]
        The input data for which predictions are to be made. It should be in a format compatible with the `net`'s `forward` method.
    targets : NDArray[np.float64]
        The actual target values for the input data. For classification, this should be one-hot encoded.
    task_type : str, optional
        The type of task for which the fitness is being evaluated. Should be 'classification' for classification tasks and 'regression' for regression tasks. Defaults to 'regression'.

    Returns
    -------
    NDArray[np.float64]
        The computed error for the given weights and input data. For classification, this is the categorical cross-entropy error. For regression, it's the root mean square error.

    Raises
    ------
    ValueError
        If `task_type` is not 'classification' or 'regression'.
    """
    if task_type == "classification":
        output3d = net.forward(X, weights)
        error = categorical_crossentropy3d(targets, output3d)
    elif task_type == "regression":
        output2d = net.forward(X, weights)[:, :, 0]
        print(output2d.shape, targets.shape)
        error = root_mean_square_error2d(targets, output2d)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    return error


def fitness_function_structure(
    population: NDArray,
    X: NDArray[np.float64],
    targets: NDArray[np.float64],
    net_size_penalty: float,
    task_type: str = "regression",
) -> NDArray[np.float64]:
    """
    Evaluate the fitness of a population of neural networks for a given task.

    This function computes the error between the networks' predictions and the actual targets. It supports both classification and regression tasks, determined by the `task_type` parameter.

    Parameters
    ----------
    population : NDArray
        The population of neural networks to be evaluated.
    X : NDArray[np.float64]
        The input data for which predictions are to be made. It should be in a format compatible with the networks' `forward` method.
    targets : NDArray[np.float64]
        The actual target values for the input data. For classification, this should be one-hot encoded.
    net_size_penalty : float
        Penalty for the size of the neural networks in the population.
    task_type : str, optional
        The type of task for which the fitness is being evaluated. Should be 'classification' for classification tasks and 'regression' for regression tasks. Defaults to 'regression'.

    Returns
    -------
    NDArray[np.float64]
        The computed error for the given population, input data, and target values. For classification, this is the sum of categorical cross-entropy error and network size penalty. For regression, it's the sum of root mean square error and network size penalty.

    Raises
    ------
    ValueError
        If `task_type` is not 'classification' or 'regression'.
    """
    if task_type == "classification":
        output3d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)
        lens = np.array(list(map(len, population)))
        fitness = categorical_crossentropy3d(targets, output3d) + net_size_penalty * lens
    elif task_type == "regression":
        output2d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)[:, :, 0]
        lens = np.array(list(map(len, population)))
        fitness = root_mean_square_error2d(targets, output2d) + net_size_penalty * lens
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    return fitness


def genotype_to_phenotype_tree(
    tree: Tree, n_variables: int, n_outputs: int, output_activation: str, offset: bool
) -> Net:
    pack: Any = []

    n = n_variables
    for node in reversed(tree._nodes):
        args = []
        for _ in range(node._n_args):
            args.append(pack.pop())
        if isinstance(node, FunctionalNode):
            pack.append(node._value(*args))
        else:
            if type(node) is TerminalNode:
                unit = Net(inputs=node._value)
            else:
                end = n + node._value._size
                hidden_id = set(range(n, end))
                activs = dict(zip(hidden_id, [node._value._activ] * len(hidden_id)))
                n = end
                unit = Net(hidden_layers=[hidden_id], activs=activs)
            pack.append(unit)
    end = n + n_outputs
    output_id = set(range(n, end))
    activs = dict(zip(output_id, [ACTIV_NAME_INV[output_activation]] * len(output_id)))
    to_return = pack[0] > Net(outputs=output_id, activs=activs)
    to_return = to_return._fix(set(range(n_variables)))
    to_return._offset = offset

    return to_return


def genotype_to_phenotype(
    population_g: NDArray,
    n_outputs: int,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    weights_optimizer_args: Dict,
    weights_optimizer_class: weights_type_optimizer_alias,
    output_activation: str,
    offset: bool,
    task_type: str = "regression",
) -> NDArray:

    print(task_type, "genotype_to_phenotype")
    n_variables: int = X_train.shape[1]

    population_ph = np.array(
        [
            train_net_weights(
                genotype_to_phenotype_tree(
                    individ_g, n_variables, n_outputs, output_activation, offset
                ),
                X_train=X_train,
                y_train=y_train,
                weights_optimizer_args=weights_optimizer_args,
                weights_optimizer=weights_optimizer_class,
                fitness_function=fitness_function_weights,
                task_type=task_type,
            )
            for individ_g in population_g
        ],
        dtype=object,
    )

    return population_ph


def train_net_weights(
    net: Net,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    weights_optimizer_args: Dict,
    weights_optimizer: weights_type_optimizer_alias,
    fitness_function: Callable,
    task_type: str = "regression",
) -> Net:

    print(task_type, "train_net_weights")
    net = net.copy()
    weights_optimizer_args = weights_optimizer_args.copy()

    weights_optimizer_args["fitness_function"] = fitness_function
    weights_optimizer_args["fitness_function_args"] = {
        "net": net,
        "X": X_train,
        "targets": y_train,
        "task_type": task_type,
    }

    left: NDArray[np.float64] = np.full(shape=len(net._weights), fill_value=-10, dtype=np.float64)
    right: NDArray[np.float64] = np.full(shape=len(net._weights), fill_value=10, dtype=np.float64)

    initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = (
        DifferentialEvolution.float_population(weights_optimizer_args["pop_size"], left, right)
    )
    initial_population[0] = net._weights.copy()

    if weights_optimizer in (SHADE, DifferentialEvolution, jDE):
        weights_optimizer_args["left"] = left
        weights_optimizer_args["right"] = right
    else:
        genotype_to_phenotype = GrayCode().fit(
            left_border=-10.0,
            right_border=10.0,
            num_variables=len(net._weights),
            bits_per_variable=16,
        )
        weights_optimizer_args["str_len"] = np.sum(genotype_to_phenotype._bits_per_variable)
        weights_optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype.transform

    weights_optimizer_args["minimization"] = True
    optimizer = weights_optimizer(**weights_optimizer_args)
    optimizer.fit()

    phenotype = optimizer.get_fittest()["phenotype"]

    return phenotype, optimizer._stats


def train_net_structure(
    uniset: UniversalSet,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    optimizer_args: Dict,
    optimizer,
    fitness_function_structure: Callable,
    net_size_penalty: float,
    weights_optimizer_args: Dict,
    weights_optimizer: weights_type_optimizer_alias,
    fitness_function_weights: Callable,
    offset: bool,
    task_type: str = "regression",
):

    print(task_type, "train_net_structure")
    if task_type == "regression":
        n_outputs = 1
        output_activation = "sigma"
    else:
        n_outputs = X_train.shape[1]
        output_activation = "softmax"

    optimizer_args["fitness_function"] = fitness_function_structure
    optimizer_args["fitness_function_args"] = {
        "X": X_test,
        "targets": y_test,
        "net_size_penalty": net_size_penalty,
    }

    optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype
    optimizer_args["genotype_to_phenotype_args"] = {
        "n_outputs": n_outputs,
        "X_train": X_train,
        "y_train": y_train,
        "weights_optimizer_args": weights_optimizer_args,
        "weights_optimizer_class": weights_optimizer,
        "output_activation": output_activation,
        "offset": offset,
        "task_type": task_type,
    }

    optimizer_args["uniset"] = uniset
    optimizer_args["minimization"] = True

    optimizer = optimizer(**optimizer_args)

    optimizer.fit()

    net = optimizer.get_fittest()["phenotype"].copy()

    return net


class BaseMLPEA(BaseEstimator, metaclass=ABCMeta):
    """
    Attributes that have been estimated from the data must always have a name ending with trailing underscore,
    for example the coefficients of some regression estimator would be stored in a coef_ attribute after fit has been called.
    The estimated attributes are expected to be overridden when you call fit a second time.

    In iterative algorithms, the number of iterations should be specified by an integer called n_iter.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        n_iter: int,
        pop_size: int,
        hidden_layers: Tuple[int, ...],
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.offset = offset
        self.weights_optimizer = weights_optimizer
        self.weights_optimizer_args = weights_optimizer_args
        self.random_state = random_state

    def _defitne_net(self, n_inputs: int, n_outputs: int) -> Net:
        start = 0
        end = n_inputs
        inputs_id = set(range(start, end))

        net = Net(inputs=inputs_id)

        for n_layer in self.hidden_layers:
            start = end
            end = end + n_layer
            inputs_id = {(n_inputs - 1)}
            hidden_id = set(range(start, end))
            activs = dict(zip(hidden_id, [ACTIV_NAME_INV[self.activation]] * len(hidden_id)))

            if self.offset:
                layer_net = Net(inputs=inputs_id) > Net(hidden_layers=[hidden_id], activs=activs)
            else:
                layer_net = Net(hidden_layers=[hidden_id], activs=activs)

            net = net > layer_net

        start = end
        end = end + n_outputs
        inputs_id = {(n_inputs - 1)}
        output_id = set(range(start, end))

        if isinstance(self, ClassifierMixin):
            outputs_activation = [ACTIV_NAME_INV["softmax"]] * len(output_id)
        else:
            outputs_activation = [ACTIV_NAME_INV["sigma"]] * len(output_id)

        activs = dict(zip(output_id, outputs_activation))

        if self.offset:
            layer_net = Net(inputs=inputs_id) > Net(outputs=output_id, activs=activs)
        else:
            layer_net = Net(outputs=output_id, activs=activs)

        net = net > layer_net
        net._offset = self.offset
        return net

    def array_like_to_numpy_X_y(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        return X, y

    def get_optimizer(
        self,
    ) -> Union[
        DifferentialEvolution,
        GeneticAlgorithm,
        GeneticProgramming,
        jDE,
        SelfCGA,
        SelfCGP,
        SHADE,
        SHAGA,
    ]:
        return self.optimizer_

    def get_net(self) -> Net:
        return self.net_

    def check_weights_optimizer_args(self) -> dict:

        if self.weights_optimizer_args is None:
            weights_optimizer_args = {"iters": 100, "pop_size": 100}
        else:
            weights_optimizer_args = self.weights_optimizer_args.copy()
            for arg in (
                "fitness_function",
                "iters",
                "pop_size",
                "left",
                "right",
                "str_len",
                "genotype_to_phenotype",
                "minimization",
            ):
                assert (
                    arg not in weights_optimizer_args.keys()
                ), f"""Do not set the "{arg}"
                to the "weights_optimizer_args". It is defined automatically"""
        weights_optimizer_args["iters"] = self.n_iter
        weights_optimizer_args["pop_size"] = self.pop_size

        return weights_optimizer_args

    def fit(self, X: ArrayLike, y: ArrayLike):

        weights_optimizer_args = self.check_weights_optimizer_args()

        check_random_state(self.random_state)
        self._target_scaler = MinMaxScaler()

        if isinstance(self, ClassifierMixin):
            X, y = self._validate_data(X, y, y_numeric=False, reset=True)
            check_classification_targets(y)
            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(
                sparse_output=False, categories="auto", dtype=np.float64
            )

            numeric_labels = self._label_encoder.fit_transform(y)
            y = self._one_hot_encoder.fit_transform(np.array(numeric_labels).reshape(-1, 1))
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)
        else:
            X, y = self._validate_data(X, y, y_numeric=True, reset=True)

            y = self._target_scaler.fit_transform(y.reshape(-1, 1))[:, 0]

        X, y = self.array_like_to_numpy_X_y(X, y)

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        if isinstance(self, ClassifierMixin):
            self.net_ = self._defitne_net(X.shape[1], len(self.classes_))
            self.net_._weights, self.optimizer_stats_ = train_net_weights(
                self.net_,
                X,
                y,
                weights_optimizer_args,
                self.weights_optimizer,
                fitness_function_weights,
                task_type="classification",
            )

        else:
            self.net_ = self._defitne_net(X.shape[1], 1)
            self.net_._weights, self.optimizer_stats_ = train_net_weights(
                self.net_,
                X,
                y,
                weights_optimizer_args,
                self.weights_optimizer,
                fitness_function_weights,
                task_type="regression",
            )

        return self

    def predict(self, X: NDArray[np.float64]):
        check_is_fitted(self)

        X = check_array(X)
        self._validate_data
        n_features = X.shape[1]

        if self.n_features_in_ != n_features:
            raise ValueError(
                f"Number of features of the model must match the "
                "input. Model n_features is {self.n_features_in_} and input "
                "n_features is {n_features}."
            )

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self.net_.forward(X)[0]

        if isinstance(self, ClassifierMixin):
            indeces = np.argmax(output, axis=1)
            y = self._label_encoder.inverse_transform(indeces)
        else:
            y = self._target_scaler.inverse_transform(output)[:, 0]

        return y


class BaseGPNN(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        *,
        n_iter: int,
        pop_size: int,
        input_block_size: int = 1,
        max_hidden_block_size: int = 9,
        offset: bool = True,
        test_sample_ratio: float = 0.5,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        net_size_penalty: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.input_block_size = input_block_size
        self.max_hidden_block_size = max_hidden_block_size
        self.offset = offset
        self.test_sample_ratio = test_sample_ratio
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.weights_optimizer = weights_optimizer
        self.weights_optimizer_args = weights_optimizer_args
        self.net_size_penalty = net_size_penalty
        self.random_state = random_state

    def array_like_to_numpy_X_y(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        return X, y

    def genotype_to_phenotype_tree(
        tree: Tree, n_variables: int, n_outputs: int, output_activation: str, offset: bool
    ) -> Net:
        pack: Any = []

        n = n_variables
        for node in reversed(tree._nodes):
            args = []
            for _ in range(node._n_args):
                args.append(pack.pop())
            if isinstance(node, FunctionalNode):
                pack.append(node._value(*args))
            else:
                if type(node) is TerminalNode:
                    unit = Net(inputs=node._value)
                else:
                    end = n + node._value._size
                    hidden_id = set(range(n, end))
                    activs = dict(zip(hidden_id, [node._value._activ] * len(hidden_id)))
                    n = end
                    unit = Net(hidden_layers=[hidden_id], activs=activs)
                pack.append(unit)
        end = n + n_outputs
        output_id = set(range(n, end))
        activs = dict(zip(output_id, [ACTIV_NAME_INV[output_activation]] * len(output_id)))
        to_return = pack[0] > Net(outputs=output_id, activs=activs)
        to_return = to_return._fix(set(range(n_variables)))
        to_return._offset = offset

        return to_return

    def check_optimizer_args(self) -> dict:

        if self.optimizer_args is None:
            optimizer_args = {"iters": 30, "pop_size": 100}
        else:
            optimizer_args = self.optimizer_args.copy()
            for arg in (
                "fitness_function",
                "iters",
                "pop_size",
                "uniset",
                "genotype_to_phenotype",
                "minimization",
            ):
                assert (
                    arg not in optimizer_args.keys()
                ), f"""Do not set the "{arg}"
                to the "optimizer_args". It is defined automatically"""
        optimizer_args["iters"] = self.n_iter
        optimizer_args["pop_size"] = self.pop_size

        return optimizer_args

    def check_weights_optimizer_args(self) -> dict:

        if self.weights_optimizer_args is None:
            weights_optimizer_args = {"iters": 100, "pop_size": 100}
        else:
            weights_optimizer_args = self.weights_optimizer_args.copy()
            for arg in (
                "fitness_function",
                "left",
                "right",
                "str_len",
                "genotype_to_phenotype",
                "minimization",
            ):
                assert (
                    arg not in weights_optimizer_args.keys()
                ), f"""Do not set the "{arg}"
                to the "weights_optimizer_args". It is defined automatically"""

        return weights_optimizer_args

    def fit(self, X: ArrayLike, y: ArrayLike):

        optimizer_args = self.check_optimizer_args()
        weights_optimizer_args = self.check_weights_optimizer_args()

        print(optimizer_args)
        print(weights_optimizer_args)

        check_random_state(self.random_state)
        self._target_scaler = MinMaxScaler()

        if isinstance(self, ClassifierMixin):
            X, y = self._validate_data(X, y, y_numeric=False, reset=True)
            check_classification_targets(y)
            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(
                sparse_output=False, categories="auto", dtype=np.float64
            )

            numeric_labels = self._label_encoder.fit_transform(y)
            y = self._one_hot_encoder.fit_transform(np.array(numeric_labels).reshape(-1, 1))
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)
        else:
            X, y = self._validate_data(X, y, y_numeric=True, reset=True)

            y = self._target_scaler.fit_transform(y.reshape(-1, 1))[:, 0]

        X, y = self.array_like_to_numpy_X_y(X, y)

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_sample_ratio)

        uniset: UniversalSet = init_net_uniset(
            n_variables=X.shape[1],
            input_block_size=self.input_block_size,
            max_hidden_block_size=self.max_hidden_block_size,
            offset=self.offset,
        )

        if isinstance(self, ClassifierMixin):
            task_type = "classification"
        else:
            task_type = "regression"

        print(task_type)
        self.net_ = train_net_structure(
            uniset=uniset,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            optimizer_args=optimizer_args,
            optimizer=self.optimizer,
            fitness_function_structure=fitness_function_structure,
            net_size_penalty=self.net_size_penalty,
            weights_optimizer_args=weights_optimizer_args,
            weights_optimizer=self.weights_optimizer,
            fitness_function_weights=fitness_function_weights,
            offset=self.offset,
            task_type=task_type,
        )

        return self

    # def _define_optimizer(
    #     self,
    #     uniset: UniversalSet,
    #     n_outputs: int,
    #     X_train: NDArray[np.float64],
    #     target_train: NDArray[np.float64],
    #     X_test: NDArray[np.float64],
    #     target_test: NDArray[np.float64],
    #     fitness_function: Callable,
    #     evaluate_nets: Callable,
    # ) -> Union[SelfCGP, GeneticProgramming]:
    #     optimizer_args: dict[str, Any]

    #     if self._optimizer_args is not None:
    #         assert (
    #             "iters" not in self._optimizer_args.keys()
    #             and "pop_size" not in self._optimizer_args.keys()
    #         ), """Do not set the "iters" or "pop_size" in the "optimizer_args". Instead,
    #           use the "SymbolicRegressionGP" arguments"""
    #         for arg in (
    #             "fitness_function",
    #             "uniset",
    #             "minimization",
    #         ):
    #             assert (
    #                 arg not in self._optimizer_args.keys()
    #             ), f"""Do not set the "{arg}"
    #             to the "optimizer_args". It is defined automatically"""
    #         optimizer_args = self._optimizer_args.copy()

    #     else:
    #         optimizer_args = {}

    #     optimizer_args["fitness_function"] = fitness_function
    #     optimizer_args["fitness_function_args"] = {
    #         "X": X_test,
    #         "targets": target_test,
    #         "net_size_penalty": self._net_size_penalty,
    #     }

    #     optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype
    #     optimizer_args["genotype_to_phenotype_args"] = {
    #         "n_outputs": n_outputs,
    #         "X_train": X_train,
    #         "proba_train": target_train,
    #         "weights_optimizer_args": self._weights_optimizer_args,
    #         "weights_optimizer_class": self._weights_optimizer_class,
    #         "output_activation": self._output_activation,
    #         "offset": self._offset,
    #         "evaluate_nets": evaluate_nets,
    #     }

    #     optimizer_args["iters"] = self._iters
    #     optimizer_args["pop_size"] = self._pop_size
    #     optimizer_args["uniset"] = uniset
    #     optimizer_args["minimization"] = True

    #     return self._optimizer_class(**optimizer_args)
