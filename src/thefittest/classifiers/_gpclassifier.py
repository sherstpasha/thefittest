from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

from sklearn.base import ClassifierMixin

from ..base._gp import BaseGP
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP


class GeneticProgrammingClassifier(ClassifierMixin, BaseGP):
    def __init__(
        self,
        *,
        n_iter: int = 200,
        pop_size: int = 500,
        functional_set_names: Tuple[str, ...] = ("cos", "sin", "add", "inv", "neg", "mul"),
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            functional_set_names=functional_set_names,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            random_state=random_state,
        )

    def _more_tags(self):
        return {"binary_only": True}
