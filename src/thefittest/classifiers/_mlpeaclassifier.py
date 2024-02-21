from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import jDE
from thefittest.base._model import BaseMLPEA
from sklearn.base import ClassifierMixin

weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]


class MLPEAClassifier(ClassifierMixin, BaseMLPEA):
    def __init__(
        self,
        *,
        iters: int = 100,
        pop_size: int = 500,
        hidden_layers: Tuple[int, ...] = (100,),
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            iters=iters,
            pop_size=pop_size,
            hidden_layers=hidden_layers,
            activation=activation,
            offset=offset,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
            random_state=random_state,
        )
