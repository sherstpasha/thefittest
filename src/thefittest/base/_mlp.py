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
from ..utils import (
    _snapshot_tensor_meta,
    _to_numpy_for_validation,
    _back_to_torch,
)
from ..utils._metrics import categorical_crossentropy3d
from ..utils._metrics import root_mean_square_error2d
from ..utils.random import check_random_state
from ..utils.transformations import GrayCode
import torch


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


def _ce_from_probs_torch(
    probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Кросс-энтропия для one-hot targets по вероятностям (probs уже после softmax).
    Возвращает скалярный loss (mean по сэмплам).
    """
    # Сгладим распространённые формы вывода сети: (N,C,1) -> (N,C); (N,C,K) -> (N*K,C)
    if probs.ndim == 3 and probs.size(-1) == 1:
        probs = probs.squeeze(-1)
    if targets.ndim == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    if probs.ndim == 3:
        probs = probs.reshape(-1, probs.size(-1))
        targets = targets.reshape(-1, targets.size(-1))
    # безопасный лог
    probs = probs.clamp_min(eps)
    loss_vec = -(targets * probs.log()).sum(dim=-1)
    return loss_vec.mean()


def _rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    RMSE, аккуратно выжимаем размерности вида (N,1) / (N,1,1).
    Возвращает скалярный loss (mean по сэмплам).
    """
    for t in ("pred", "target"):
        pass
    if pred.ndim == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.ndim == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)
    if pred.ndim == 2 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.ndim == 2 and target.size(-1) == 1:
        target = target.squeeze(-1)
    return torch.sqrt(torch.mean((pred - target) ** 2))


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
        return self.optimizer_stats_

    def get_net(self) -> Net:
        return self.net_

    def fit(self, X, y):
        # === 1) подготовка аргументов оптимизатора (как было) ===
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

        # === 2) снимем метаданные тензоров, временно переведём в NumPy для валидации ===
        Xm, ym = _snapshot_tensor_meta(X), _snapshot_tensor_meta(y)
        X_np = _to_numpy_for_validation(X)
        y_np = _to_numpy_for_validation(y)

        # === 3) валидация/предобработка как раньше ===
        if isinstance(self, ClassifierMixin):
            X_np, y_np = self._validate_data(X_np, y_np, y_numeric=False, reset=True)
            check_classification_targets(y_np)

            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(
                sparse_output=False, categories="auto", dtype=np.float64
            )

            numeric_labels = self._label_encoder.fit_transform(y_np)
            y_np = self._one_hot_encoder.fit_transform(np.array(numeric_labels).reshape(-1, 1))
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)
        else:
            X_np, y_np = self._validate_data(X_np, y_np, y_numeric=True, reset=True)

        # === 4) offset как было ===
        if self.offset:
            X_np = np.hstack([X_np, np.ones((X_np.shape[0], 1))])

        # === 5) вернёмся в torch на исходные device/dtype ===
        X_t = _back_to_torch(X_np.astype(np.float32, copy=False), Xm, dtype=torch.float32)
        if isinstance(self, ClassifierMixin):
            # one-hot уже в y_np
            y_t = _back_to_torch(y_np.astype(np.float32, copy=False), ym, dtype=torch.float32)
        else:
            # регрессия — числовой вектор/столбец
            y_t = _back_to_torch(np.asarray(y_np, dtype=np.float32), ym, dtype=torch.float32)

        # === 6) создание сети и обучение (как было) ===
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
        # === 7) совместимость со sklearn ===
        self.n_features_in_ = X_np.shape[1]
        return self
