from typing import Any
from typing import Union

import numpy as np
from numpy.typing import NDArray


class Model:
    def _fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> Any:
        pass

    def _predict(self, X: NDArray[np.float64]) -> Any:
        pass

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> Any:
        return self._fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        return self._predict(X)
