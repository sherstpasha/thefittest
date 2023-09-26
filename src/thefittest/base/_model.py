from __future__ import annotations

from typing import Any
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE


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
    ) -> Union[
        DifferentialEvolution,
        GeneticAlgorithm,
        GeneticProgramming,
        jDE,
        SelfCGA,
        SelfCGP,
        SHADE,
        SHAGA,
    ]:
        pass

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> Any:
        return self._fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        return self._predict(X)
