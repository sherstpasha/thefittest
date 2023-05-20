from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
from typing import Set
from typing import Dict
from functools import reduce
import numpy as np
from numpy.typing import NDArray
from ..tools.operators import LogisticSigmoid
from ..tools.operators import ReLU
from ..tools.operators import SoftMax
from ..base._model import Model
from thefittest.tools.transformations import scale_data
from itertools import product
import random
from ..tools.operators import forward_softmax2d


INPUT_COLOR_CODE = (0.11, 0.67, 0.47, 1)
HIDDEN_COLOR_CODE = (0.0, 0.74, 0.99, 1)
OUTPUT_COLOR_CODE = (0.94, 0.50, 0.50, 1)
SOFTMAX_F = SoftMax()


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
        self._structure = tuple(
            [n_inputs] + list(self._hidden_layers) + [n_outputs])

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


class Net:
    def __init__(self,
                 inputs: Set = set(),
                 hidden_layers: Optional[List] = None,
                 outputs: Set = set(),
                 connects:  NDArray = np.empty((0, 2), dtype=np.int64),
                 weights: NDArray = np.empty((0), dtype=np.float64),
                 activs: Dict = dict()):
        self._inputs = inputs
        self._hidden_layers = hidden_layers or []
        self._outputs = outputs
        self._connects = connects
        self._weights = weights
        self._activs = activs

        self._forward_inputs_array: NDArray[np.int64]
        self._forward_outputs_array: NDArray[np.int64]
        self._forward_cond_h: NDArray[np.bool]
        self._forward_cond_o = NDArray[np.bool]
        self._forward_culc_order_h = NDArray[np.int64]
        self._forward_culc_order_o = NDArray[np.int64]
        self._forward_activ_code = NDArray[np.int64]

    def __len__(self):
        return len(self._weights)

    def copy(self):
        return Net(inputs=self._inputs.copy(),
                   hidden_layers=self._hidden_layers.copy(),
                   outputs=self._outputs.copy(),
                   connects=self._connects.copy(),
                   weights=self._weights.copy(),
                   activs=self._activs.copy())

    def _assemble_hiddens(self) -> Set:
        if len(self._hidden_layers) > 0:
            return set.union(*self._hidden_layers)
        else:
            return set([])

    def _merge_layers(self,
                      layers: List) -> Set:
        return layers[0].union(layers[1])

    def _connect(self,
                 left: Union[Set, int],
                 right: Union[Set, int]) -> Tuple:
        if len(left) and len(right):
            connects = np.array(list(product(left, right)), dtype=np.int64)
            weights = np.random.normal(0, 1, len(connects)).astype(np.float64)
            return (connects, weights)
        else:
            return (np.zeros((0, 2), dtype=int),
                    np.zeros((0), dtype=float))

    def __add__(self, other):
        len_i_1, len_i_2 = len(self._inputs), len(other._inputs)
        len_h_1, len_h_2 = len(self._hidden_layers), len(other._hidden_layers)

        if (len_i_1 > 0 and len_i_2 == 0) and (len_h_1 == 0 and len_h_2 > 0):
            return self > other
        elif (len_i_1 == 0 and len_i_2 > 0) and (len_h_1 > 0 and len_h_2 == 0):
            return other > self

        map_res = map(self._merge_layers, zip(
            self._hidden_layers, other._hidden_layers))
        if len_h_1 < len_h_2:
            excess = other._hidden_layers[len_h_1:]
        elif len_h_1 > len_h_2:
            excess = self._hidden_layers[len_h_2:]
        else:
            excess = []

        hidden = list(map_res) + excess
        return Net(inputs=self._inputs.union(other._inputs),
                   hidden_layers=hidden,
                   outputs=self._outputs.union(other._outputs),
                   connects=np.vstack([self._connects, other._connects]),
                   weights=np.hstack([self._weights, other._weights]),
                   activs={**self._activs, **other._activs})

    def __gt__(self, other):
        len_i_1, len_i_2 = len(self._inputs), len(other._inputs)
        len_h_1, len_h_2 = len(self._hidden_layers), len(other._hidden_layers)

        if (len_i_1 > 0 and len_h_1 == 0) and (len_i_2 > 0 and len_h_2 == 0):
            return self + other
        elif (len_i_1 == 0 and len_h_1 > 0) and (len_i_2 > 0 and len_h_2 == 0):
            return other > self

        inputs_hidden = self._inputs.union(self._assemble_hiddens())
        from_ = inputs_hidden.difference(self._connects[:, 0])

        cond = other._connects[:, 0][:, np.newaxis] == np.array(
            list(other._inputs))
        cond = np.any(cond, axis=1)

        connects_no_i = other._connects[:, 1][~cond]
        hidden_outputs = other._assemble_hiddens().union(other._outputs)
        to_ = hidden_outputs.difference(connects_no_i)

        connects, weights = self._connect(from_, to_)
        return Net(inputs=self._inputs.union(other._inputs),
                   hidden_layers=self._hidden_layers + other._hidden_layers,
                   outputs=self._outputs.union(other._outputs),
                   connects=np.vstack(
                       [self._connects, other._connects, connects]),
                   weights=np.hstack([self._weights, other._weights, weights]),
                   activs={**self._activs, **other._activs})

    def _fix(self, inputs):
        hidden_outputs = self._assemble_hiddens().union(self._outputs)
        to_ = hidden_outputs.difference(self._connects[:, 1])
        if len(to_) > 0:
            if not len(self._inputs):
                self._inputs = inputs

            connects, weights = self._connect(self._inputs, to_)
            self._connects = np.vstack([self._connects, connects])
            self._weights = np.hstack([self._weights, weights])

        self._connects = np.unique(self._connects, axis=0)
        self._weights = self._weights[:len(self._connects)]
        return self

    def _get_order(self):
        activs_code = []
        hidden = self._assemble_hiddens()

        from_ = self._connects[:, 0]
        to_ = self._connects[:, 1]

        calculated = self._inputs.copy()

        bool_hidden = to_[:, np.newaxis] == np.array(list(hidden))
        order = {j: set(from_[bool_hidden[:, i]])
                 for i, j in enumerate(hidden)}

        hidden_conds = np.full((len(hidden), len(to_)), fill_value=False)
        hidden_culc_order = np.zeros(shape=(len(hidden)), dtype=np.int64)
        output_conds = np.full(
            (len(self._outputs), len(to_)), fill_value=False)
        output_culc_order = np.zeros(
            shape=(len(self._outputs)), dtype=np.int64)

        k = 0
        current = self._inputs.union(hidden)
        while calculated != current:
            for i in hidden:
                if order[i].issubset(calculated) and i not in calculated:
                    hidden_conds[k] = to_ == i
                    hidden_culc_order[k] = i
                    calculated.add(i)
                    k += 1
                    activs_code.append(self._activs[i].id_)

        k = 0
        for i in self._outputs:
            output_conds[k] = to_ == i
            output_culc_order[k] = i
            k += 1

        self._forward_inputs_array = np.array(
            list(self._inputs), dtype=np.int64)
        self._forward_outputs_array = np.array(
            list(self._outputs), dtype=np.int64)
        self._forward_cond_h = hidden_conds
        self._forward_cond_o = output_conds
        self._forward_culc_order_h = hidden_culc_order
        self._forward_culc_order_o = output_culc_order
        self._forward_activ_code = np.array(activs_code, dtype=np.int64)

    def forward_softmax(self,
                        X: NDArray[np.float64],
                        weights: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:

        if weights is None:
            weights = self._weights.reshape(1, -1)
        else:
            weights = weights

        outputs = forward_softmax2d(X,
                                    self._forward_inputs_array,
                                    self._forward_outputs_array,
                                    self._forward_cond_h,
                                    self._forward_cond_o,
                                    self._forward_culc_order_h,
                                    self._forward_culc_order_o,
                                    weights,
                                    self._connects[:, 0],
                                    self._forward_activ_code)
        return outputs

    def get_graph(self) -> Dict:
        weights_scale = scale_data(self._weights)
        nodes = list(self._inputs)

        len_i = len(self._inputs)
        len_h = len(self._assemble_hiddens())
        len_o = len(self._outputs)
        sum_ = len_i + len_h + len_o

        positions = np.zeros((sum_, 2), dtype=float)
        colors = np.zeros((sum_, 4))
        w_colors = np.zeros((len(weights_scale), 4))
        labels = {**dict(zip(self._inputs, self._inputs)),
                  **{key: value.string for key, value in self._activs.items()},
                  **dict(zip(self._outputs, range(len_o)))}

        w_colors[:, 0] = 1 - weights_scale
        w_colors[:, 2] = weights_scale
        w_colors[:, 3] = 0.8
        positions[:len_i][:, 1] = np.arange(len_i) - (len_i)/2
        colors[:len_i] = INPUT_COLOR_CODE

        n = len_i
        for i, layer in enumerate(self._hidden_layers):
            nodes.extend(list(layer))
            positions[n:n + len(layer)][:, 0] = i + 1
            positions[n:n + len(layer)][:, 1] = np.arange(len(layer)) \
                - len(layer)/2
            colors[n:n + len(layer)] = HIDDEN_COLOR_CODE
            n += len(layer)

        nodes.extend(list(self._outputs))
        positions[n: n + len_o][:, 0] = len(self._hidden_layers) + 1
        positions[n: n + len_o][:, 1] = np.arange(len_o) - len_o/2
        colors[n: n + len_o] = OUTPUT_COLOR_CODE

        positions_dict = dict(zip(nodes, positions))

        to_return = {'nodes': nodes,
                     'labels': labels,
                     'positions': positions_dict,
                     'colors': colors,
                     'weights_colors': w_colors,
                     'connects': self._connects}

        return to_return


class HiddenBlock:
    def __init__(self,
                 activ: Optional[Union[LogisticSigmoid, ReLU]] = None,
                 size: Optional[int] = None) -> None:
        self.activ = activ
        self.size = size
        self.__name__ = 'HiddenBlock'

    def __str__(self) -> str:
        return '{}{}'.format(self.activ.string, self.size)

    def __call__(self,
                 max_size: int):
        size = random.randrange(1, max_size)
        activ = random.sample([LogisticSigmoid, ReLU], k=1)[0]
        return HiddenBlock(activ(),  size)
