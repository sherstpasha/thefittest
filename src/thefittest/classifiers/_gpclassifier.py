from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from ..base._gp import BaseGP
from ..base._gp import safe_sigmoid
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP


class GeneticProgrammingClassifier(ClassifierMixin, BaseGP):
    def __init__(
        self,
        *,
        n_iter: int = 300,
        pop_size: int = 1000,
        functional_set_names: Tuple[str, ...] = ("cos", "sin", "add", "sub", "mul", "div"),
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        use_fitness_cache: bool = False,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            functional_set_names=functional_set_names,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            random_state=random_state,
            use_fitness_cache=use_fitness_cache,
        )

    def predict_proba(self, X: NDArray[np.float64]):
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_}."
            )

        def _tree_out(tree):
            tfp = tree.set_terminals(**{f"x{i}": X[:, i] for i in range(self.n_features_in_)})
            return tfp() * np.ones(len(X))

        # OVR: если в fit обучили по дереву на класс
        if (
            hasattr(self, "trees_")
            and isinstance(self.trees_, (list, tuple))
            and len(self.trees_) > 0
        ):
            logits = np.column_stack([_tree_out(t) for t in self.trees_])  # (N, K)
            P = safe_sigmoid(logits)
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            proba = (P / row_sums).astype(np.float64, copy=False)
            return proba

        tree_output = _tree_out(self.tree_)
        p1 = safe_sigmoid(tree_output)
        proba = np.vstack([1.0 - p1, p1]).T.astype(np.float64, copy=False)
        return proba

    def predict(self, X: NDArray[np.float64]):
        proba = self.predict_proba(X)  # ndarray (N, K)
        indeces = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indeces)
