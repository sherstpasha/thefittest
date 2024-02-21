from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Any
from typing import Union
from typing import Optional
from typing import Tuple
from typing import Type

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

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


def fitness_function_classifier(
    weights: NDArray[np.float64],
    net: Net,
    X: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    output3d = net.forward(X, weights)
    error = categorical_crossentropy3d(targets, output3d)
    return error


def fitness_function_regressor(
    weights: NDArray[np.float64],
    net: Net,
    X: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    output2d = net.forward(X, weights)[:, :, 0]
    error = root_mean_square_error2d(targets, output2d)
    return error


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
        iters: int,
        pop_size: int,
        hidden_layers: Tuple[int, ...],
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.iters = iters
        self.pop_size = pop_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.offset = offset
        self.weights_optimizer = weights_optimizer
        self.weights_optimizer_args = weights_optimizer_args
        self.random_state = random_state

    def _defitne_net(self: BaseEstimator, n_inputs: int, n_outputs: int) -> Net:
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
            outputs_activation = [ACTIV_NAME_INV["relu"]] * len(output_id)

        activs = dict(zip(output_id, outputs_activation))

        if self.offset:
            layer_net = Net(inputs=inputs_id) > Net(outputs=output_id, activs=activs)
        else:
            layer_net = Net(outputs=output_id, activs=activs)

        net = net > layer_net
        net._offset = self.offset
        return net

    def _train_net(
        self,
        net: Net,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        net = net.copy()

        if self.weights_optimizer_args is not None:
            for arg in (
                "fitness_function",
                "left",
                "right",
                "str_len",
                "genotype_to_phenotype",
                "minimization",
            ):
                assert (
                    "iters" not in self.weights_optimizer_args.keys()
                    and "pop_size" not in self.weights_optimizer_args.keys()
                ), """Do not set the "iters" or "pop_size", or "uniset" in the "optimizer_args".
                  Instead, use the "MLPClassifierEA" arguments"""
                assert (
                    arg not in self.weights_optimizer_args.keys()
                ), f"""Do not set the "{arg}"
              to the "weights_optimizer_args". It is defined automatically"""
            weights_optimizer_args = self.weights_optimizer_args.copy()
        else:
            weights_optimizer_args = {}

        weights_optimizer_args["iters"] = self.iters
        weights_optimizer_args["pop_size"] = self.pop_size
        left: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=-10, dtype=np.float64
        )
        right: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=10, dtype=np.float64
        )
        initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = (
            DifferentialEvolution.float_population(weights_optimizer_args["pop_size"], left, right)
        )
        initial_population[0] = net._weights.copy()

        if isinstance(self, ClassifierMixin):
            weights_optimizer_args["fitness_function"] = fitness_function_classifier
            weights_optimizer_args["fitness_function_args"] = {
                "net": net,
                "X": X_train,
                "targets": y_train,
            }
        else:
            weights_optimizer_args["fitness_function"] = fitness_function_regressor
            weights_optimizer_args["fitness_function_args"] = {
                "net": net,
                "X": X_train,
                "targets": y_train,
            }

        if self.weights_optimizer in (SHADE, DifferentialEvolution, jDE):
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

        optimizer = self.weights_optimizer(**weights_optimizer_args).fit()
        phenotype = optimizer.get_fittest()["phenotype"]
        optimizer._fitness_function_args["net"] = optimizer._fitness_function_args["net"].copy()
        self.optimizer_ = optimizer

        return phenotype

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

    def fit(self, X: ArrayLike, y: ArrayLike):

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

        X, y = self.array_like_to_numpy_X_y(X, y)

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        if isinstance(self, ClassifierMixin):    
            self.net_ = self._defitne_net(X.shape[1], len(self.classes_))
        else:
            self.net_ = self._defitne_net(X.shape[1], 1)

        self.net_._weights = self._train_net(self.net_, X, y)

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
            y = output[:,0]

        return y
