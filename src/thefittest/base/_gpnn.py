from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import check_classification_targets

from ..base import FunctionalNode
from ..base import Net
from ..base import TerminalNode
from ..base import Tree
from ..base import UniversalSet
from ..base._ea import Statistics
from ..base._mlp import check_optimizer_args
from ..base._mlp import fitness_function_weights
from ..base._mlp import train_net_weights
from ..base._mlp import weights_type_optimizer_alias
from ..base._net import ACTIV_NAME_INV
from ..base._tree import init_net_uniset
from ..optimizers import GeneticProgramming
from ..optimizers import SHAGA
from ..optimizers import SelfCGP
from ..utils import array_like_to_numpy_X_y
from ..utils._metrics import categorical_crossentropy3d
from ..utils._metrics import root_mean_square_error2d
from ..utils.random import check_random_state


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

    n_variables: int = X_train.shape[1]

    population_ph = np.array(
        [
            genotype_to_phenotype_tree(individ_g, n_variables, n_outputs, output_activation, offset)
            for individ_g in population_g
        ],
        dtype=object,
    )

    for i, net in enumerate(population_ph):
        population_ph[i]._weights, _ = train_net_weights(
            net=net,
            X_train=X_train,
            y_train=y_train,
            weights_optimizer_args=weights_optimizer_args,
            weights_optimizer=weights_optimizer_class,
            fitness_function=fitness_function_weights,
            task_type=task_type,
        )

    return population_ph


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
    offset: bool,
    n_outputs: int,
    task_type: str = "regression",
):
    if task_type == "regression":
        output_activation = "ln"
    else:
        output_activation = "softmax"

    optimizer_args["fitness_function"] = fitness_function_structure
    optimizer_args["fitness_function_args"] = {
        "X": X_test,
        "targets": y_test,
        "net_size_penalty": net_size_penalty,
        "task_type": task_type,
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
    tree = optimizer.get_fittest()["genotype"].copy()

    return net, tree, optimizer._stats


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
        weights_optimizer: weights_type_optimizer_alias = SHAGA,
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

    def get_net(self) -> Net:
        return self.net_

    def get_tree(self) -> Tree:
        return self.tree_

    def get_stats(self) -> Statistics:
        return self.optimizer_stats_

    @staticmethod
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

    def fit(self, X: ArrayLike, y: ArrayLike):

        if self.optimizer_args is not None:
            optimizer_args = self.optimizer_args.copy()
            check_optimizer_args(
                optimizer_args,
                args_auto_defined=[
                    "fitness_function",
                    "fitness_function_args",
                    "uniset",
                    "genotype_to_phenotype",
                    "genotype_to_phenotype_args",
                    "init_population",
                    "minimization",
                ],
                args_in_class=["iters", "pop_size"],
            )
        else:
            optimizer_args = {}
        optimizer_args["iters"] = self.n_iter
        optimizer_args["pop_size"] = self.pop_size

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
            )
        else:
            weights_optimizer_args = {}

        if "iters" not in weights_optimizer_args:
            weights_optimizer_args["iters"] = 300
        if "pop_size" not in weights_optimizer_args:
            weights_optimizer_args["pop_size"] = 300

        random_state = check_random_state(self.random_state)

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_sample_ratio, random_state=random_state
        )

        uniset: UniversalSet = init_net_uniset(
            n_variables=X.shape[1],
            input_block_size=self.input_block_size,
            max_hidden_block_size=self.max_hidden_block_size,
            offset=self.offset,
        )

        if isinstance(self, ClassifierMixin):
            task_type = "classification"
            n_outputs = self.n_classes_
        else:
            task_type = "regression"
            n_outputs = 1

        self.net_, self.tree_, self.optimizer_stats_ = train_net_structure(
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
            offset=self.offset,
            n_outputs=n_outputs,
            task_type=task_type,
        )

        return self
