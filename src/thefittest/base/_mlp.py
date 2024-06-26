from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import check_classification_targets

from ..base import Net
from ..base._ea import Statistics
from ..base._net import ACTIV_NAME_INV
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import jDE
from ..utils import array_like_to_numpy_X_y
from ..utils._metrics import categorical_crossentropy3d
from ..utils._metrics import root_mean_square_error2d
from ..utils.random import check_random_state
from ..utils.transformations import GrayCode


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


def check_optimizer_args(
    optimizer_args, args_auto_defined: Optional[list] = None, args_in_class: Optional[list] = None
) -> None:
    if args_auto_defined is not None:
        for arg in args_auto_defined:
            assert (
                arg not in optimizer_args.keys()
            ), f"""Do not set the "{arg}"
            to the "weights_optimizer_args". It is defined automatically"""

    if args_in_class is not None:
        for arg in args_in_class:
            assert (
                arg not in optimizer_args.keys()
            ), f"Do not set '{arg}' in 'optimizer_args'. Instead, use the arguments of the class."


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
        error = root_mean_square_error2d(targets, output2d)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    return error


def train_net_weights(
    net: Net,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    weights_optimizer_args: Dict,
    weights_optimizer: weights_type_optimizer_alias,
    fitness_function: Callable,
    task_type: str = "regression",
) -> Net:

    net = net.copy()
    weights_optimizer_args = weights_optimizer_args.copy()

    weights_optimizer_args["fitness_function"] = fitness_function
    weights_optimizer_args["fitness_function_args"] = {
        "net": net,
        "X": X_train,
        "targets": y_train,
        "task_type": task_type,
    }

    initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = (
        DifferentialEvolution.float_population(
            weights_optimizer_args["pop_size"], -10, 10, len(net._weights)
        )
    )
    initial_population[0] = net._weights.copy()

    if weights_optimizer in (SHADE, DifferentialEvolution, jDE):
        weights_optimizer_args["left_border"] = -10
        weights_optimizer_args["right_border"] = 10
        weights_optimizer_args["num_variables"] = len(net._weights)
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

    net = optimizer.get_fittest()["phenotype"]

    return net, optimizer._stats


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
        weights_optimizer: weights_type_optimizer_alias = SHAGA,
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
            outputs_activation = [ACTIV_NAME_INV["ln"]] * len(output_id)

        activs = dict(zip(output_id, outputs_activation))

        if self.offset:
            layer_net = Net(inputs=inputs_id) > Net(outputs=output_id, activs=activs)
        else:
            layer_net = Net(outputs=output_id, activs=activs)

        net = net > layer_net
        net._offset = self.offset
        return net

    def get_stats(self) -> Statistics:
        return self.optimizer_stats_

    def get_net(self) -> Net:
        return self.net_

    def fit(self, X: ArrayLike, y: ArrayLike):

        if self.weights_optimizer_args is not None:
            weights_optimizer_args = self.weights_optimizer_args.copy()
            check_optimizer_args(
                weights_optimizer_args,
                args_auto_defined=[
                    "fitness_function",
                    "fitness_function_args",
                    "left_border",
                    "right_border",
                    "num_variables",
                    "str_len",
                    "genotype_to_phenotype",
                    "genotype_to_phenotype_args",
                    "minimization",
                    "init_population",
                ],
                args_in_class=["iters", "pop_size"],
            )
        else:
            weights_optimizer_args = {}

        weights_optimizer_args["iters"] = self.n_iter
        weights_optimizer_args["pop_size"] = self.pop_size

        check_random_state(self.random_state)

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

        X, y = array_like_to_numpy_X_y(X, y)

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
