from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
from typing import Set
from typing import Dict
from functools import reduce
import numpy as np
from numpy.typing import NDArray
from ..base._model import Model
from thefittest.tools.transformations import scale_data
from itertools import product
import random
from ..tools.operators import forward2d
from numba.typed import List as numbaList
from collections import defaultdict


INPUT_COLOR_CODE = (0.11, 0.67, 0.47, 1)
HIDDEN_COLOR_CODE = (0.0, 0.74, 0.99, 1)
OUTPUT_COLOR_CODE = (0.94, 0.50, 0.50, 1)
ACTIVATION_NAME = {0: 'sg', 1: 'rl', 2: 'gs', 3: 'th', 4: 'sm'}


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

        self._numpy_inputs: Optional[NDArray[np.int64]] = None
        self._numpy_outputs: NDArray[np.int64]
        self._n_hiddens: np.int64

        self._numba_from: numbaList[NDArray[np.int64]]
        self._numba_to: numbaList[NDArray[np.int64]]
        self._numba_weights_id: numbaList[NDArray[np.int64]]
        self._numba_activs_code: numbaList[NDArray[np.int64]]
        self._numba_activs_nodes: numbaList[numbaList[NDArray[np.int64]]]

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
            weights = np.random.uniform(-2, 2,
                                        len(connects)).astype(np.float64)
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
        hidden = self._assemble_hiddens()
        from_ = self._connects[:, 0]
        to_ = self._connects[:, 1]
        indexes = np.arange(len(from_), dtype=np.int64)

        argsort = np.argsort(to_)

        from_sort = from_[argsort]
        to_sort = to_[argsort]
        index_sort = indexes[argsort]

        groups_to, cut_index = np.unique(to_sort, return_index=True)
        groups_from = np.split(from_sort, cut_index)[1:]
        group_index = np.split(index_sort, cut_index)[1:]

        pairs = defaultdict(list)
        weights_id = defaultdict(list)

        for groups_to_i, groups_from_i, group_index_i in zip(groups_to,
                                                             groups_from,
                                                             group_index):

            argsort = np.argsort(groups_from_i)
            groups_from_i_sort = tuple(groups_from_i[argsort])
            pairs[groups_from_i_sort].append(groups_to_i)
            weights_id[groups_from_i_sort].append(group_index_i[argsort])

        for key, value in weights_id.items():
            weights_id[key] = np.array(value, dtype=np.int64)

        numba_from = numbaList()
        numba_to = numbaList()
        numba_weight_id = numbaList()

        activ_code = numbaList()
        active_nodes = numbaList()

        calculated = self._inputs.copy()
        purpose = self._inputs.union(hidden).union(self._outputs)

        while calculated != purpose:
            for from_i, to_i in pairs.items():
                if set(from_i).issubset(calculated) and not set(to_i).issubset(calculated):
                    calculated = calculated.union(set(to_i))
                    numba_from.append(np.array(from_i, dtype=np.int64))
                    numba_to.append(np.array(to_i, dtype=np.int64))
                    numba_weight_id.append(weights_id[from_i])

                    nodes_i = defaultdict(list)
                    for to_i_i in to_i:
                        nodes_i[self._activs[to_i_i]].append(to_i_i)

                    activ_code.append(
                        np.array(list(nodes_i.keys()), dtype=np.int64))

                    nodes_i_list = numbaList()
                    for value in nodes_i.values():
                        nodes_i_list.append(np.array(value, dtype=np.int64))

                    active_nodes.append(nodes_i_list)

        self._numpy_inputs = np.array(list(self._inputs), dtype=np.int64)
        self._numpy_outputs = np.array(list(self._outputs), dtype=np.int64)
        self._n_hiddens = len(hidden)
        self._numba_from = numba_from
        self._numba_to = numba_to
        self._numba_weights_id = numba_weight_id
        self._numba_activs_code = activ_code
        self._numba_activs_nodes = active_nodes

    def forward(self,
                X: NDArray[np.float64],
                weights: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:

        if weights is None:
            weights = self._weights.reshape(1, -1)
        else:
            weights = weights

        if self._numpy_inputs is None:
            self._get_order()

        outputs = forward2d(X,
                            self._numpy_inputs,
                            self._n_hiddens,
                            self._numpy_outputs,
                            self._numba_from,
                            self._numba_to,
                            self._numba_weights_id,
                            self._numba_activs_code,
                            self._numba_activs_nodes,
                            weights)
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
                  **{key: ACTIVATION_NAME[value] for key, value in self._activs.items()},
                  **dict(zip(self._outputs, range(len_o)))
                  }

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
                 activ: Optional[int] = None,
                 size: Optional[int] = None) -> None:
        self._activ = activ
        self._size = size
        self.__name__ = 'HiddenBlock'

    def __str__(self) -> str:
        return '{}{}'.format(ACTIVATION_NAME[self._activ], self._size)

    def __call__(self,
                 max_size: int):
        self._activ = random.sample([0, 1, 2, 3], k=1)[0]
        self._size = random.randrange(1, max_size)
        return self
