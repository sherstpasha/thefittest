from __future__ import annotations

from typing import Any
from typing import Union

import numpy as np
from numpy.typing import NDArray


class Model:
    def _fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[Union[np.float32, np.int64]],
    ) -> Any:
        pass

    def _predict(self, X: NDArray[np.float32]) -> Any:
        pass

    def get_optimizer(
        self: Model,
    ) -> Any:
        pass

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[Union[np.float32, np.int64]],
    ) -> Any:
        # assert np.all(np.isfinite(X))
        # assert np.all(np.isfinite(y))
        return self._fit(X, y)

    def predict(self, X: NDArray[np.float32]) -> NDArray[Union[np.float32, np.int64]]:
        return self._predict(X).astype(np.float32)
