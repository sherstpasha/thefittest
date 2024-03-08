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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import Net
from ..base._net import ACTIV_NAME_INV
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
                "Number of features of the model must match the "
                f"input. Model n_features is {self.n_features_in_} and input "
                f"n_features is {n_features}."
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
