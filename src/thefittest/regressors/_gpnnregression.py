from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

import torch

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ..base._gpnn import BaseGPNN
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SelfCGP


class GeneticProgrammingNeuralNetRegressor(RegressorMixin, BaseGPNN):
    def __init__(
        self,
        *,
        n_iter: int = 15,
        pop_size: int = 50,
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
        device: str = 'cpu',
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            input_block_size=input_block_size,
            max_hidden_block_size=max_hidden_block_size,
            offset=offset,
            test_sample_ratio=test_sample_ratio,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
            net_size_penalty=net_size_penalty,
            random_state=random_state,
            device=device
        )

    def predict(self, X: NDArray[np.float64]):

        check_is_fitted(self)

        if hasattr(self, "_validate_data"):
            X = self._validate_data(X, reset=False)
        else:
            X = validate_data(self, X, reset=False)

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device(self.device)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            out = self.net_.forward(X_t)

        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        if out.ndim == 3 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.ndim == 2 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out.reshape(-1)