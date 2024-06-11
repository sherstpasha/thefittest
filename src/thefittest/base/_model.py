from __future__ import annotations

from typing import Any
from typing import Union

import numpy as np
from numpy.typing import NDArray

import cloudpickle


class Model:
    def _fit(
        self,
        X: NDArray[np.float64],
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

    def save_to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as file:
            cloudpickle.dump(self, file)

    @classmethod
    def load_from_file(cls, file_path: str) -> Model:
        with open(file_path, "rb") as file:
            return cloudpickle.load(file)