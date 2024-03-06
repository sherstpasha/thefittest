from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base._net import Net
from ..classifiers._gpnnclassifier import weights_type_optimizer_alias
from ..optimizers import DifferentialEvolution
from ..optimizers import SHADE
from ..optimizers import jDE
from ..utils._metrics import root_mean_square_error2d
from ..utils.transformations import GrayCode
from thefittest.base._model import BaseMLPEA
from sklearn.base import RegressorMixin


class MLPEARegressor(RegressorMixin, BaseMLPEA):
    def __init__(
        self,
        *,
        n_iter: int = 200,
        pop_size: int = 500,
        hidden_layers: Tuple[int, ...] = (100,),
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            hidden_layers=hidden_layers,
            activation=activation,
            offset=offset,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
            random_state=random_state,
        )
