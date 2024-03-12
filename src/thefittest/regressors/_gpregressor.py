from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Type
from typing import Union

import numpy as np

from sklearn.base import RegressorMixin

from ..base import UniversalSet
from ..base._gp import BaseGP
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP


class GeneticProgrammingRegressor(RegressorMixin, BaseGP):
    def __init__(
        self,
        *,
        n_iter: int = 50,
        pop_size: int = 500,
        uniset: Optional[UniversalSet] = None,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            uniset=uniset,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            random_state=random_state,
        )
