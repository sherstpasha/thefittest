from __future__ import annotations

import random
from collections import defaultdict
from itertools import product
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from numba.typed import List as numbaList

import numpy as np
from numpy.typing import NDArray

from thefittest.tools.transformations import scale_data

from ..tools.operators import forward2d


INPUT_COLOR_CODE = (0.11, 0.67, 0.47, 1)
HIDDEN_COLOR_CODE = (0.0, 0.74, 0.99, 1)
OUTPUT_COLOR_CODE = (0.94, 0.50, 0.50, 1)
ACTIVATION_NAME = {0: "sg", 1: "rl", 2: "gs", 3: "th", 4: "ln", 5: "sm"}
ACTIV_NAME_INV = {"sigma": 0, "relu": 1, "gauss": 2, "tanh": 3, "ln": 4, "softmax": 5}


class Net:
    def __init__(
        self,
        inputs: Optional[Set] = None,
        hidden_layers: Optional[List] = None,
        outputs: Optional[Set] = None,
        connects: Optional[NDArray[np.int64]] = None,
        weights: Optional[NDArray[np.float64]] = None,
        activs: Optional[Dict[int, int]] = None,
    ):
        self._inputs = inputs or set()
        self._hidden_layers = hidden_layers or []
        self._outputs = outputs or set()
        self._connects = self._set_connects(values=connects)
        self._weights = self._set_weights(values=weights)
        self._activs = activs or {}
        self._offset: bool = False

        self._numpy_inputs: Optional[NDArray[np.int64]] = None
        self._numpy_outputs: NDArray[np.int64]
        self._n_hiddens: np.int64

        self._numba_from: numbaList[NDArray[np.int64]]
        self._numba_to: numbaList[NDArray[np.int64]]
        self._numba_weights_id: numbaList[NDArray[np.int64]]
        self._numba_activs_code: numbaList[NDArray[np.int64]]
        self._numba_activs_nodes: numbaList[numbaList[NDArray[np.int64]]]

    def __len__(self) -> int:
        return len(self._weights)

    def __eq__(self, other: Net) -> bool:
        if self._inputs != other._inputs:
            return False
        elif any(h_1 != h_2 for h_1, h_2 in zip(self._hidden_layers, other._hidden_layers)):
            return False
        elif self._outputs != other._outputs:
            return False
        elif len(self._connects) != len(other._connects):
            return False
        elif np.any(self._connects != other._connects):
            return False
        elif self._activs != other._activs:
            return False
        else:
            return True

    def _set_connects(self, values: Optional[NDArray[np.int64]]) -> NDArray[np.int64]:
        if values is None:
            to_return = np.empty((0, 2), dtype=np.int64)
        else:
            to_return = values
        return to_return

    def _set_weights(self, values: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
        if values is None:
            to_return = np.empty((0), dtype=np.float64)
        else:
            to_return = values

        return to_return

    def copy(self) -> Net:
        hidden_layers = [layer.copy() for layer in self._hidden_layers]
        copy_net = Net(
            inputs=self._inputs.copy(),
            hidden_layers=hidden_layers,
            outputs=self._outputs.copy(),
            connects=self._connects.copy(),
            weights=self._weights.copy(),
            activs=self._activs.copy(),
        )
        copy_net._offset = self._offset
        return copy_net

    def _assemble_hiddens(self) -> Set[int]:
        if len(self._hidden_layers) > 0:
            return set.union(*self._hidden_layers)
        else:
            return set()

    def _get_connect(
        self, left: Set[int], right: Set[int]
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        if len(left) and len(right):
            connects = np.array(list(product(left, right)), dtype=np.int64)
            weights = np.random.uniform(-2, 2, len(connects)).astype(np.float64)
            return (connects, weights)
        else:
            return (np.zeros((0, 2), dtype=np.int64), np.zeros((0), dtype=np.float64))

    def __add__(self, other: Net) -> Net:
        len_i_1, len_i_2 = len(self._inputs), len(other._inputs)
        len_h_1, len_h_2 = len(self._hidden_layers), len(other._hidden_layers)

        if (len_i_1 > 0 and len_i_2 == 0) and (len_h_1 == 0 and len_h_2 > 0):
            return self > other
        elif (len_i_1 == 0 and len_i_2 > 0) and (len_h_1 > 0 and len_h_2 == 0):
            return other > self

        map_res = map(
            lambda layers: layers[0].union(layers[1]),
            zip(self._hidden_layers, other._hidden_layers),
        )

        if len_h_1 < len_h_2:
            excess = other._hidden_layers[len_h_1:]
        elif len_h_1 > len_h_2:
            excess = self._hidden_layers[len_h_2:]
        else:
            excess = []

        hidden = list(map_res) + excess
        return Net(
            inputs=self._inputs.union(other._inputs),
            hidden_layers=hidden,
            outputs=self._outputs.union(other._outputs),
            connects=np.vstack([self._connects, other._connects]),
            weights=np.hstack([self._weights, other._weights]),
            activs={**self._activs, **other._activs},
        )

    def __gt__(self, other: Net) -> Net:
        len_i_1, len_i_2 = len(self._inputs), len(other._inputs)
        len_h_1, len_h_2 = len(self._hidden_layers), len(other._hidden_layers)

        if (len_i_1 > 0 and len_h_1 == 0) and (len_i_2 > 0 and len_h_2 == 0):
            return self + other
        elif (len_i_1 == 0 and len_h_1 > 0) and (len_i_2 > 0 and len_h_2 == 0):
            return other > self

        inputs_hidden = self._inputs.union(self._assemble_hiddens())
        from_ = inputs_hidden.difference(self._connects[:, 0])

        cond = other._connects[:, 0][:, np.newaxis] == np.array(list(other._inputs))
        cond = np.any(cond, axis=1)

        connects_no_i = other._connects[:, 1][~cond]
        hidden_outputs = other._assemble_hiddens().union(other._outputs)
        to_ = hidden_outputs.difference(connects_no_i)

        connects, weights = self._get_connect(from_, to_)
        return Net(
            inputs=self._inputs.union(other._inputs),
            hidden_layers=self._hidden_layers + other._hidden_layers,
            outputs=self._outputs.union(other._outputs),
            connects=np.vstack([self._connects, other._connects, connects]),
            weights=np.hstack([self._weights, other._weights, weights]),
            activs={**self._activs, **other._activs},
        )

    def _fix(self, inputs: Set[int]) -> Net:
        hidden_outputs = self._assemble_hiddens().union(self._outputs)
        to_ = hidden_outputs.difference(self._connects[:, 1])
        if len(to_) > 0:
            if not len(self._inputs):
                self._inputs = inputs

            connects, weights = self._get_connect(self._inputs, to_)
            self._connects = np.vstack([self._connects, connects])
            self._weights = np.hstack([self._weights, weights])

        self._connects = np.unique(self._connects, axis=0)
        self._weights = self._weights[: len(self._connects)]
        return self

    def _get_order(self) -> None:
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
        weights_id_list = defaultdict(list)
        weights_id_ndarray = {}

        for groups_to_i, groups_from_i, group_index_i in zip(groups_to, groups_from, group_index):
            argsort = np.argsort(groups_from_i)
            groups_from_i_sort = tuple(groups_from_i[argsort])
            pairs[groups_from_i_sort].append(groups_to_i)
            weights_id_list[groups_from_i_sort].append(group_index_i[argsort])

        for key, value in weights_id_list.items():
            weights_id_ndarray[key] = np.array(value, dtype=np.int64)

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
                    numba_weight_id.append(weights_id_ndarray[from_i])

                    nodes_i = defaultdict(list)
                    for to_i_i in to_i:
                        nodes_i[self._activs[to_i_i]].append(to_i_i)

                    activ_code.append(np.array(list(nodes_i.keys()), dtype=np.int64))

                    nodes_i_list = numbaList()
                    for value in nodes_i.values():
                        nodes_i_list.append(np.array(value, dtype=np.int64))

                    active_nodes.append(nodes_i_list)

        self._numpy_inputs = np.array(list(self._inputs), dtype=np.int64)
        self._numpy_outputs = np.array(list(self._outputs), dtype=np.int64)
        self._n_hiddens = np.int64(len(hidden))
        self._numba_from = numba_from
        self._numba_to = numba_to
        self._numba_weights_id = numba_weight_id
        self._numba_activs_code = activ_code
        self._numba_activs_nodes = active_nodes

    def forward(
        self, X: NDArray[np.float64], weights: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        if weights is None:
            weights = self._weights.reshape(1, -1)
        else:
            weights = weights

        if self._numpy_inputs is None:
            self._get_order()
        outputs = forward2d(
            X,
            self._numpy_inputs,
            self._n_hiddens,
            self._numpy_outputs,
            self._numba_from,
            self._numba_to,
            self._numba_weights_id,
            self._numba_activs_code,
            self._numba_activs_nodes,
            weights,
        )
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
        labels = {
            **dict(zip(self._inputs, self._inputs)),
            **{key: ACTIVATION_NAME[value] for key, value in self._activs.items()},
            **dict(zip(self._outputs, range(len_o))),
        }

        w_colors[:, 0] = 1 - weights_scale
        w_colors[:, 2] = weights_scale
        w_colors[:, 3] = 0.8
        positions[:len_i][:, 1] = np.arange(len_i) - (len_i) / 2
        colors[:len_i] = INPUT_COLOR_CODE

        n = len_i
        for i, layer in enumerate(self._hidden_layers):
            nodes.extend(list(layer))
            positions[n : n + len(layer)][:, 0] = i + 1
            positions[n : n + len(layer)][:, 1] = np.arange(len(layer)) - len(layer) / 2
            colors[n : n + len(layer)] = HIDDEN_COLOR_CODE
            n += len(layer)

        nodes.extend(list(self._outputs))
        positions[n : n + len_o][:, 0] = len(self._hidden_layers) + 1
        positions[n : n + len_o][:, 1] = np.arange(len_o) - len_o / 2
        colors[n : n + len_o] = OUTPUT_COLOR_CODE

        positions_dict = dict(zip(nodes, positions))

        to_return = {
            "nodes": nodes,
            "labels": labels,
            "positions": positions_dict,
            "colors": colors,
            "weights_colors": w_colors,
            "connects": self._connects,
        }

        return to_return


class HiddenBlock:
    def __init__(self, max_size: int) -> None:
        self._activ = random.sample([0, 1, 2, 3, 4], k=1)[0]
        self._size = random.randrange(1, max_size)

    def __str__(self) -> str:
        return "{}{}".format(ACTIVATION_NAME[self._activ], self._size)


class NetEnsemble:
    def __init__(self, nets: NDArray, meta_algorithm: Optional[Net] = None):
        self._nets = nets
        self._meta_algorithm = meta_algorithm
        self._meta_tree = None

    def __len__(self) -> int:
        return len(self._nets)

    def __eq__(self, other: NetEnsemble) -> bool:
        if len(self) != len(other):
            return False
        elif any((net_i != net_j for net_i, net_j in zip(self._nets, other._nets))):
            return False
        else:
            return True

    def get_nets(self: NetEnsemble) -> NDArray:
        return self._nets

    def forward(
        self: NetEnsemble,
        X: NDArray[np.float64],
        weights_list: Optional[List[NDArray[np.float64]]] = None,
    ) -> NDArray[np.float64]:
        if weights_list is not None:
            to_return = [
                net_i.forward(X, weights_i) for net_i, weights_i in zip(self._nets, weights_list)
            ]
        else:
            to_return = [net_i.forward(X) for net_i in self._nets]

        return np.array(to_return, dtype=np.float64)

    def _get_meta_inputs(
        self: NetEnsemble, X: NDArray[np.float64], offset: bool = True
    ) -> NDArray[np.float64]:
        outputs = self.forward(X)[:, 0]
        X_input = np.concatenate(outputs, axis=1)
        if offset:
            X_input = np.hstack([X_input, np.ones((X_input.shape[0], 1))])

        return X_input

    def meta_output(
        self: NetEnsemble,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._meta_algorithm is not None:
            X_input = self._get_meta_inputs(X, offset=self._meta_algorithm._offset)
            output = self._meta_algorithm.forward(X=X_input)[0]
            return output
        else:
            raise ValueError("text222")

    def copy(self) -> NetEnsemble:
        copy_ = NetEnsemble(np.array([net_i.copy() for net_i in self._nets], dtype=object))
        if self._meta_algorithm is not None:
            copy_._meta_algorithm = self._meta_algorithm.copy()

        if self._meta_tree is not None:
            copy_._meta_tree = self._meta_tree.copy()
        return copy_
