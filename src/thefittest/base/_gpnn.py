from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from typing import TYPE_CHECKING

from collections import OrderedDict
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if TYPE_CHECKING:
        import torch

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
from ..optimizers import NSGPII
from ..utils import array_like_to_numpy_X_y
from ..utils.random import check_random_state
from ..utils._metrics import _ce_from_probs_torch, _rmse_torch


class LimitedSizeDict(OrderedDict):
    def __init__(self, max_size: int, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            super().__delitem__(key)
        elif len(self) >= self.max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)


def net_complexity(net: Net, criterion: str = "n_connections") -> int:
    """Measure the structure of a network phenotype.

    Parameters
    ----------
    net : Net
        Network whose structure is measured.
    criterion : str
        "n_neurons" counts hidden and output neurons; "n_connections" counts
        connections; "n_layers" counts nonempty hidden layers and the output
        layer; "n_inputs" counts input nodes, including a bias input if present.
    """
    if criterion == "n_neurons":
        return len(set().union(net._outputs, *net._hidden_layers))
    if criterion == "n_connections":
        return len(net._connects)
    if criterion == "n_layers":
        return sum(bool(layer) for layer in net._hidden_layers) + bool(net._outputs)
    if criterion == "n_inputs":
        return len(net._inputs)
    raise ValueError(f"Unknown network complexity criterion: {criterion}.")


def fitness_function_structure(
    population: NDArray,
    X: torch.Tensor,
    targets: torch.Tensor,
    net_size_penalty: float,
    task_type: str = "regression",
    fitness_cache: Optional[dict[str, dict]] = None,
) -> NDArray[np.float64]:

    losses = np.empty(len(population), dtype=np.float64)
    with torch.no_grad():
        for i, net in enumerate(population):
            sig = net.signature()

            if fitness_cache is not None and sig in fitness_cache:
                net._weights = fitness_cache[sig]["weights"].clone().to(X.device)
                net.compile_torch()

                if fitness_cache[sig]["fitness"] is not None:
                    losses[i] = fitness_cache[sig]["fitness"]
                    continue

            out = net.forward(X)
            if task_type == "classification":
                loss_t = _ce_from_probs_torch(out, targets)
            elif task_type == "regression":
                loss_t = _rmse_torch(out, targets)
            else:
                raise ValueError("task_type must be 'classification' or 'regression'")

            fitness_val = float(loss_t.item()) + net_size_penalty * len(net)
            losses[i] = fitness_val

            if fitness_cache is not None:
                if sig not in fitness_cache:
                    fitness_cache[sig] = {}
                fitness_cache[sig]["fitness"] = None
                fitness_cache[sig]["weights"] = net._weights.detach().cpu().clone()

    return losses


def fitness_function_structure_multiobjective(
    population: NDArray[Any],
    X: torch.Tensor,
    targets: torch.Tensor,
    net_size_penalty: float,
    complexity: str = "n_connections",
    task_type: str = "regression",
    fitness_cache: Optional[dict[str, dict]] = None,
) -> NDArray[np.float64]:
    """Return penalized prediction loss and network complexity as minimization objectives."""
    complexities = np.array(
        [net_complexity(net, complexity) for net in population], dtype=np.float64
    )
    losses = fitness_function_structure(
        population,
        X,
        targets,
        net_size_penalty,
        task_type=task_type,
        fitness_cache=fitness_cache,
    )
    return np.column_stack((losses, complexities))


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
    fitness_cache: Optional[dict[str, dict]] = None,
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
        sig = net.signature()

        if fitness_cache is not None and sig in fitness_cache:
            if fitness_cache[sig]["weights"] is not None:
                net._weights = fitness_cache[sig]["weights"].clone().to(X_train.device)
                net.compile_torch()
            continue

        population_ph[i]._weights, _ = train_net_weights(
            net=net,
            X_train=X_train,
            y_train=y_train,
            weights_optimizer_args=weights_optimizer_args,
            weights_optimizer=weights_optimizer_class,
            fitness_function=fitness_function_weights,
            task_type=task_type,
        )
        population_ph[i].compile_torch()

        if fitness_cache is not None:
            fitness_cache[sig] = {
                "fitness": None,
                "weights": net._weights.detach().cpu().clone(),
            }

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
    fitness_cache: Optional[dict[str, dict]] = None,
    complexity: str = "n_connections",
    model_selection: str = "best_error",
) -> tuple[Net, Tree, Statistics, Optional[dict[str, Any]], Optional[int]]:
    multiobjective = issubclass(optimizer, NSGPII)
    if multiobjective:
        if complexity not in ("n_neurons", "n_connections", "n_layers", "n_inputs"):
            raise ValueError(f"Unknown network complexity criterion: {complexity}.")
        if model_selection not in ("best_error", "simplest"):
            raise ValueError('model_selection must be "best_error" or "simplest".')
    if task_type == "regression":
        output_activation = "ln"
    else:
        output_activation = "softmax"

    optimizer_args["fitness_function"] = (
        fitness_function_structure_multiobjective if multiobjective else fitness_function_structure
    )
    optimizer_args["fitness_function_args"] = {
        "X": X_test,
        "targets": y_test,
        "net_size_penalty": net_size_penalty,
        "task_type": task_type,
        "fitness_cache": fitness_cache,
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
        "fitness_cache": fitness_cache,
    }

    optimizer_args["uniset"] = uniset
    if multiobjective:
        optimizer_args["fitness_function_args"]["complexity"] = complexity
        optimizer_args["minimization"] = (True, True)
    else:
        optimizer_args["minimization"] = True

    optimizer = optimizer(**optimizer_args)

    optimizer.fit()

    pareto_front = None
    selected_index = None
    if multiobjective:
        pareto_front = optimizer.get_pareto_front()
        errors, complexities = pareto_front["fitness"].T
        order = (
            np.lexsort((complexities, errors))
            if model_selection == "best_error"
            else np.lexsort((errors, complexities))
        )
        selected_index = int(order[0])
        net = pareto_front["phenotype"][selected_index].copy()
        tree = pareto_front["genotype"][selected_index].copy()
    else:
        fittest = optimizer.get_fittest()
        net = fittest["phenotype"].copy()
        tree = fittest["genotype"].copy()

    return net, tree, optimizer.get_stats(), pareto_front, selected_index


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
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming], Type[NSGPII]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        weights_optimizer: weights_type_optimizer_alias = SHAGA,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        net_size_penalty: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        device: str = "cpu",
        use_fitness_cache: bool = False,
        fitness_cache_size: int = 1000,
        complexity: str = "n_connections",
        model_selection: str = "best_error",
    ) -> None:
        super().__init__()
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
        self.device = device
        self.use_fitness_cache = use_fitness_cache
        self.fitness_cache_size = fitness_cache_size
        self.complexity = complexity
        self.model_selection = model_selection

    def get_net(self) -> Net:
        """
        Get the evolved and trained neural network.

        Returns
        -------
        net : Net
            The neural network with GP-evolved architecture and optimized weights.
            Can be used for visualization or further analysis.
        """
        return self.net_

    def get_tree(self) -> Tree:
        """
        Get the evolved tree representing the network architecture.

        Returns
        -------
        tree : Tree
            The tree expression that encodes the evolved neural network structure.
            Each node represents network layers and connections.
        """
        return self.tree_

    def get_stats(self) -> Statistics:
        """
        Get optimization statistics from the architecture evolution process.

        Returns
        -------
        stats : Statistics
            Statistics object containing fitness history and other metrics
            collected during the architecture evolution process.
        """
        return self.optimizer_stats_

    def get_pareto_front(self) -> dict[str, Any]:
        """Return a detached copy of the fitted Pareto networks, trees and objectives."""
        check_is_fitted(self, ["net_", "tree_"])
        if not hasattr(self, "pareto_front_"):
            raise ValueError("A Pareto front is available only after fitting with NSGPII.")
        return deepcopy(self.pareto_front_)

    def select_net(self, index: int) -> BaseGPNN:
        """Activate a Pareto network and its tree without retraining weights.

        Parameters
        ----------
        index : int
            Zero-based position in the fitted Pareto front.
        """
        check_is_fitted(self, ["net_", "tree_"])
        if not hasattr(self, "pareto_front_"):
            raise ValueError("A Pareto front is available only after fitting with NSGPII.")
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer.")
        if not 0 <= index < len(self.pareto_front_["fitness"]):
            raise IndexError("index is outside the Pareto front.")

        net = self.pareto_front_["phenotype"][index].copy()
        tree = self.pareto_front_["genotype"][index].copy()
        net.compile_torch()
        self.net_ = net
        self.tree_ = tree
        self.selected_net_index_ = int(index)
        return self

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

        if self.use_fitness_cache:
            self._fitness_cache = LimitedSizeDict(max_size=self.fitness_cache_size)
        else:
            self._fitness_cache = None

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
            if hasattr(self, "_validate_data"):
                X, y = self._validate_data(X, y, reset=True)
            else:
                X, y = validate_data(self, X, y, reset=True)

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
            if hasattr(self, "_validate_data"):
                X, y = self._validate_data(X, y, reset=True)
            else:
                X, y = validate_data(self, X, y, reset=True)

        X, y = array_like_to_numpy_X_y(X, y)

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device(self.device)
        X_t = torch.as_tensor(X, dtype=torch.float64, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float64, device=device)

        X_train, X_test, y_train, y_test = train_test_split(
            X_t, y_t, test_size=self.test_sample_ratio, random_state=random_state
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

        self.net_, self.tree_, self.optimizer_stats_, pareto_front, selected_index = (
            train_net_structure(
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
                fitness_cache=self._fitness_cache,
                complexity=self.complexity,
                model_selection=self.model_selection,
            )
        )
        if pareto_front is not None:
            self.pareto_front_ = pareto_front
            self.selected_net_index_ = selected_index
        else:
            self.__dict__.pop("pareto_front_", None)
            self.__dict__.pop("selected_net_index_", None)
        self.net_.compile_torch()
        return self
