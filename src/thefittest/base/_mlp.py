from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y

from ..base import Net
from ..base._ea import Statistics
from ..base._net import ACTIV_NAME_INV
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import jDE
from ..utils import (
    array_like_to_numpy_X_y,
    _is_torch_optimizer,
)
from ..utils._metrics import _ce_from_probs_torch
from ..utils._metrics import _rmse_torch
from ..utils.random import check_random_state
from ..utils.transformations import GrayCode

try:
    import torch
    import warnings
    from torch.optim import Optimizer as TorchOptimizer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if TYPE_CHECKING:
        import torch
        from torch.optim import Optimizer as TorchOptimizer
    else:
        # Заглушка для runtime
        class TorchOptimizer:  # type: ignore
            pass


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
    Type[TorchOptimizer],
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
    X: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = "regression",
) -> NDArray[np.float64]:

    weights_t = torch.as_tensor(weights, dtype=X.dtype, device=X.device)
    pop_size = weights.shape[0]
    losses = np.empty(pop_size, dtype=np.float64)

    with torch.no_grad():
        if task_type == "classification":
            for i in range(pop_size):
                out = net.forward(X, weights_t[i])
                loss = _ce_from_probs_torch(out, targets)
                losses[i] = float(loss.item())
        elif task_type == "regression":
            for i in range(pop_size):
                out = net.forward(X, weights_t[i])
                loss = _rmse_torch(out, targets)
                losses[i] = float(loss.item())
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
    return losses


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
    net.compile_torch(X_train.device)

    if _is_torch_optimizer(weights_optimizer):
        w_param = torch.nn.Parameter(
            net._weights.detach().to(dtype=X_train.dtype, device=X_train.device)
        )

        opt_args = dict(weights_optimizer_args) if weights_optimizer_args is not None else {}
        epochs = int(opt_args.pop("epochs", opt_args.pop("iters", 100)))
        show_each = opt_args.pop("show_progress_each", None)

        if show_each is not None:
            show_each = int(show_each)

        if "pop_size" in opt_args:
            warnings.warn(
                "`pop_size` игнорируется для torch.optim.* оптимизаторов.",
                RuntimeWarning,
            )
            opt_args.pop("pop_size", None)

        torch_opt = weights_optimizer([w_param], **opt_args)
        losses_history: list[float] = []

        for epoch in range(epochs):
            torch_opt.zero_grad(set_to_none=True)
            out = net.forward(X_train, w_param)
            if task_type == "classification":
                loss = _ce_from_probs_torch(out, y_train)
            elif task_type == "regression":
                loss = _rmse_torch(out, y_train)
            else:
                raise ValueError("task_type must be 'classification' or 'regression'")
            loss.backward()
            torch_opt.step()
            losses_history.append(float(loss.item()))

            if show_each is not None and (epoch % show_each == 0):
                print(f"[epoch {epoch}/{epochs}] loss={loss.item():.6f}", flush=True)

        weights = w_param.detach()
        return weights, None
    else:
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
        initial_population[0] = net._weights.cpu().numpy().copy()

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

        weights = torch.as_tensor(
            optimizer.get_fittest()["phenotype"], dtype=X_train.dtype, device=X_train.device
        )

        return weights, optimizer._stats


class BaseMLPEA(BaseEstimator, metaclass=ABCMeta):
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
        device: str = "cpu",
    ):
        super().__init__()
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.offset = offset
        self.weights_optimizer = weights_optimizer
        self.weights_optimizer_args = weights_optimizer_args
        self.random_state = random_state
        self.device = device

    def _defitne_net(self, n_inputs: int, n_outputs: int) -> Net:
        start = 0
        end = n_inputs
        inputs_id = set(range(start, end))

        net = Net(inputs=inputs_id)

        for n_layer in self.hidden_layers:
            start = end
            end = end + n_layer
            inputs_id = {n_inputs - 1}
            hidden_id = set(range(start, end))
            activs = dict(zip(hidden_id, [ACTIV_NAME_INV[self.activation]] * len(hidden_id)))

            if self.offset:
                layer_net = Net(inputs=inputs_id) > Net(hidden_layers=[hidden_id], activs=activs)
            else:
                layer_net = Net(hidden_layers=[hidden_id], activs=activs)

            net = net > layer_net

        start = end
        end = end + n_outputs
        inputs_id = {n_inputs - 1}
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
        """
        Get optimization statistics from the weight training process.

        Returns
        -------
        stats : Statistics
            Statistics object containing fitness history and other metrics
            collected during the weight optimization process.
            Returns None if torch.optim optimizer was used.
        """
        return self.optimizer_stats_

    def get_net(self) -> Net:
        """
        Get the trained neural network.

        Returns
        -------
        net : Net
            The trained neural network with optimized weights.
            Can be used for visualization or further analysis.
        """
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
            X, y = check_X_y(X, y)
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
            X, y = check_X_y(X, y)

        X, y = array_like_to_numpy_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device(self.device)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)

        if isinstance(self, ClassifierMixin):
            self.net_ = self._defitne_net(X_t.shape[1], len(self.classes_))
            self.net_._weights, self.optimizer_stats_ = train_net_weights(
                self.net_,
                X_t,
                y_t,
                weights_optimizer_args,
                self.weights_optimizer,
                fitness_function_weights,
                task_type="classification",
            )
        else:
            self.net_ = self._defitne_net(X_t.shape[1], 1)
            self.net_._weights, self.optimizer_stats_ = train_net_weights(
                self.net_,
                X_t,
                y_t,
                weights_optimizer_args,
                self.weights_optimizer,
                fitness_function_weights,
                task_type="regression",
            )

        self.net_.compile_torch()
        return self
