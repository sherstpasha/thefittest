from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray
from numpy.typing import ArrayLike

import torch

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ..base._gpnn import BaseGPNN
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SelfCGP


class GeneticProgrammingNeuralNetClassifier(ClassifierMixin, BaseGPNN):
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
        device: str = "cpu",
        use_fitness_cache: bool = False,
        fitness_cache_size: int = 1000,
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
            device=device,
            use_fitness_cache=use_fitness_cache,
            fitness_cache_size=fitness_cache_size,
        )

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
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
            proba_t = self.net_.forward(X_t)

        return proba_t.detach().cpu().numpy().astype(np.float64)

    def predict(self, X: ArrayLike):
        proba = self.predict_proba(X)
        indeces = np.argmax(proba, axis=1)

        return self._label_encoder.inverse_transform(indeces)
