from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
from functools import reduce
import numpy as np
from ..tools.operators import LogisticSigmoid
from ..tools.operators import ReLU
from ..tools.operators import SoftMax
from ..base._model import Model


class MultilayerPerceptron(Model):
    def __init__(
            self,
            hidden_layers: Tuple,
            activation: Union[LogisticSigmoid, ReLU] = LogisticSigmoid,
            activation_output: Union[LogisticSigmoid, SoftMax] = SoftMax) -> None:
        Model.__init__(self)
        self._hidden_layers = hidden_layers
        self._structure: Tuple
        self._weights: List
        self._activation = activation()
        self._activation_output = activation_output()

    def _define_structure(self,
                          n_inputs: int,
                          n_outputs: int) -> None:
        self._structure = tuple([n_inputs] + list(self._hidden_layers) + [n_outputs])


    def _define_weights(self) -> List:
        weights = []
        for i in range(len(self._structure) - 1):
            size = (self._structure[i]+1, self._structure[i+1])
            weights_i = np.random.uniform(low=-1, high=1, size=size)
            weights.append(weights_i)
        self._weights = weights

    def _culc_hidden_layer(self,
                           X_i: np.ndarray,
                           w_i: np.ndarray) -> np.ndarray:
        output = self._activation(np.dot(X_i, w_i[:-1]) + w_i[-1])
        return output

    def _culc_output_layer(self,
                           X_i: np.ndarray,
                           w_i: np.ndarray) -> np.ndarray:
        output = self._activation_output(np.dot(X_i, w_i[:-1]) + w_i[-1])
        return output

    def forward(self,
                X: np.ndarray,
                weights: Optional[List] = None):
        if weights is None:
            weights = self._weights
        hidden_output = reduce(self._culc_hidden_layer, weights[:-1], X)
        output = self._culc_output_layer(hidden_output, weights[-1])
        return output
