from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING

import hashlib

import numpy as np
from numpy.typing import NDArray

from ..utils import forward2d
from ..utils.transformations import minmax_scale
from ..utils.random import random_sample
from ..utils.random import uniform

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if TYPE_CHECKING:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    else:
        # Заглушки для runtime без torch
        class _DummyModule:
            """Dummy base class when torch is not available"""

            pass

        nn = type("nn", (), {"Module": _DummyModule})()  # type: ignore

SIGMA, RELU, GAUSS, TANH, LN, SOFTMAX = 0, 1, 2, 3, 4, 5


def _act(code: int, x: "torch.Tensor") -> "torch.Tensor":
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for neural network operations. "
            "Install with: pip install thefittest[torch]"
        )
    if code == SIGMA:
        return torch.sigmoid(x)
    if code == RELU:
        return F.relu(x)
    if code == GAUSS:
        return torch.exp(-x * x)
    if code == TANH:
        return torch.tanh(x)
    if code == LN:
        return x
    if code == SOFTMAX:
        return F.softmax(x, dim=1)  # softmax по оси узлов
    raise ValueError(f"Unknown activation code {code}")


ACTIVATION_NAME = {0: "sg", 1: "rl", 2: "gs", 3: "th", 4: "ln", 5: "sm"}
ACTIV_NAME_INV = {"sigma": 0, "relu": 1, "gauss": 2, "tanh": 3, "ln": 4, "softmax": 5}


if TORCH_AVAILABLE:

    class _Block(nn.Module):
        """Контейнер для индексов одного топологического шага (только буферы, веса не дублируем)."""

        def __init__(
            self,
            fr: np.ndarray,
            to: np.ndarray,
            widx: np.ndarray,
            act_codes: np.ndarray,
            act_nodes_list: list[np.ndarray],
        ):
            super().__init__()
            self.register_buffer("from_idx", torch.as_tensor(fr, dtype=torch.long))
            self.register_buffer("to_idx", torch.as_tensor(to, dtype=torch.long))
            self.register_buffer("weight_idx", torch.as_tensor(widx, dtype=torch.long))
            self.register_buffer("act_codes", torch.as_tensor(act_codes, dtype=torch.long))
            lens = [len(a) for a in act_nodes_list]
            maxlen = max(lens) if lens else 0
            pad = torch.full((len(act_nodes_list), maxlen), -1, dtype=torch.long)
            for i, arr in enumerate(act_nodes_list):
                if len(arr):
                    pad[i, : len(arr)] = torch.as_tensor(arr, dtype=torch.long)
            self.register_buffer("act_nodes", pad)
            self.register_buffer("act_nodes_len", torch.as_tensor(lens, dtype=torch.long))

else:

    class _Block:  # type: ignore
        """Dummy _Block when torch not available"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for neural network operations. "
                "Install with: pip install thefittest[torch]"
            )


class Net:
    def __init__(
        self,
        inputs: Optional[Set] = None,
        hidden_layers: Optional[List] = None,
        outputs: Optional[Set] = None,
        connects: Optional[NDArray[np.int64]] = None,
        weights: Optional["torch.Tensor"] = None,
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

        self.blocks: list[_Block] | None = None
        self.inputs: "torch.Tensor" | None = None
        self.outputs: "torch.Tensor" | None = None
        self.n_nodes: int | None = None

        self._activation_name = {0: "sg", 1: "rl", 2: "gs", 3: "th", 4: "ln", 5: "sm"}
        self._activ_name_inv = {"sigma": 0, "relu": 1, "gauss": 2, "tanh": 3, "ln": 4, "softmax": 5}

    def to(self, device: str | "torch.device", dtype: Optional["torch.dtype"] = None) -> "Net":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for 'to' method. "
                "Install with: pip install thefittest[torch]"
            )
        dev = torch.device(device)
        self._weights = self._weights.to(device=dev if dtype is None else dev, dtype=dtype)

        if self.inputs is not None:
            self.inputs = self.inputs.to(dev)
        if self.outputs is not None:
            self.outputs = self.outputs.to(dev)
        if self.blocks is not None:
            for b in self.blocks:
                b.to(dev)
        return self

    def cuda(self) -> Net:
        return self.to("cuda")

    def cpu(self) -> Net:
        return self.to("cpu")

    def __len__(self) -> int:
        if TORCH_AVAILABLE and self._weights is not None:
            return int(self._weights.numel())
        else:
            # Без torch возвращаем количество connections
            return len(self._connects)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Net):
            return NotImplemented

        if self._inputs != other._inputs:
            return False

        if len(self._hidden_layers) != len(other._hidden_layers):
            return False
        if any(h1 != h2 for h1, h2 in zip(self._hidden_layers, other._hidden_layers)):
            return False

        if self._outputs != other._outputs:
            return False

        if set(map(tuple, self._connects)) != set(map(tuple, other._connects)):
            return False

        if self._activs != other._activs:
            return False

        return True

    def _set_connects(self, values: Optional[NDArray[np.int64]]) -> NDArray[np.int64]:
        if values is None:
            to_return = np.empty((0, 2), dtype=np.int64)
        else:
            to_return = values
        return to_return

    def _set_weights(self, values: Optional["torch.Tensor"]) -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            # Возвращаем None когда torch не доступен
            return None  # type: ignore
        if values is None:
            return torch.empty(0, dtype=torch.float32, device="cpu")
        return values.to(dtype=torch.float32)

    def copy(self) -> "Net":
        hidden_layers = [layer.copy() for layer in self._hidden_layers]

        # Копируем weights только если torch доступен
        if TORCH_AVAILABLE and self._weights is not None:
            weights_copy = self._weights.detach().clone().to(self._weights.device)
        else:
            weights_copy = None

        copy_net = Net(
            inputs=self._inputs.copy(),
            hidden_layers=hidden_layers,
            outputs=self._outputs.copy(),
            connects=self._connects.copy(),
            weights=weights_copy,
            activs=self._activs.copy(),
        )
        copy_net._offset = self._offset
        return copy_net

    def _assemble_hiddens(self) -> Set[int]:
        if len(self._hidden_layers) > 0:
            return set.union(*self._hidden_layers)
        else:
            return set()

    def signature(self) -> str:
        inputs = tuple(sorted(self._inputs))
        outputs = tuple(sorted(self._outputs))

        hidden_layers = tuple(tuple(sorted(layer)) for layer in self._hidden_layers)

        connects = tuple(sorted(map(tuple, self._connects.tolist())))

        activs = tuple(sorted(self._activs.items()))

        offset = self._offset

        sig_tuple = (inputs, hidden_layers, outputs, connects, activs, offset)

        sig_str = repr(sig_tuple).encode()
        return hashlib.sha1(sig_str).hexdigest()

    def _get_connect(
        self, left: Set[int], right: Set[int]
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        if len(left) and len(right):
            connects = np.array(list(product(left, right)), dtype=np.int64)
            weights = uniform(-2, 2, len(connects))
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

        # Объединяем weights только если torch доступен
        if TORCH_AVAILABLE and self._weights is not None and other._weights is not None:
            weights = torch.cat([self._weights, other._weights], dim=0)
        else:
            weights = None

        return Net(
            inputs=self._inputs.union(other._inputs),
            hidden_layers=hidden,
            outputs=self._outputs.union(other._outputs),
            connects=np.vstack([self._connects, other._connects]),
            weights=weights,
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

        connects, weights_np = self._get_connect(from_, to_)

        # Объединяем weights только если torch доступен
        if TORCH_AVAILABLE and self._weights is not None and other._weights is not None:
            weights_t = torch.as_tensor(
                weights_np, dtype=torch.float32, device=self._weights.device
            )
            weights = torch.cat([self._weights, other._weights, weights_t], dim=0)
        else:
            weights = None

        return Net(
            inputs=self._inputs.union(other._inputs),
            hidden_layers=self._hidden_layers + other._hidden_layers,
            outputs=self._outputs.union(other._outputs),
            connects=np.vstack([self._connects, other._connects, connects]),
            weights=weights,
            activs={**self._activs, **other._activs},
        )

    def _fix(self, inputs: Set[int]) -> Net:
        hidden_outputs = self._assemble_hiddens().union(self._outputs)
        to_ = hidden_outputs.difference(self._connects[:, 1])
        if len(to_) > 0:
            if not len(self._inputs):
                self._inputs = inputs
            connects, weights_np = self._get_connect(self._inputs, to_)

            # Добавляем weights только если torch доступен
            if TORCH_AVAILABLE and self._weights is not None:
                weights_t = torch.as_tensor(
                    weights_np, dtype=torch.float32, device=self._weights.device
                )
                self._connects = np.vstack([self._connects, connects])
                self._weights = torch.cat([self._weights, weights_t], dim=0)
            else:
                self._connects = np.vstack([self._connects, connects])

        self._connects = np.unique(self._connects, axis=0)
        if (
            TORCH_AVAILABLE
            and self._weights is not None
            and self._weights.numel() > self._connects.shape[0]
        ):
            self._weights = self._weights[: self._connects.shape[0]]
        return self

    def _build_order(self) -> list[_Block]:
        hidden = set.union(*self._hidden_layers) if self._hidden_layers else set()
        fr = self._connects[:, 0]
        to = self._connects[:, 1]
        idx = np.arange(len(fr), dtype=np.int64)

        order = np.argsort(to)
        fr_s, to_s, idx_s = fr[order], to[order], idx[order]

        groups_to, cuts = np.unique(to_s, return_index=True)
        groups_from = np.split(fr_s, cuts)[1:]
        group_idx = np.split(idx_s, cuts)[1:]

        pairs: dict[tuple[int, ...], list[int]] = {}
        widx_map: dict[tuple[int, ...], np.ndarray] = {}

        for g_to, g_from, g_i in zip(groups_to, groups_from, group_idx):
            ord2 = np.argsort(g_from)
            f_sorted = tuple(g_from[ord2])
            pairs.setdefault(f_sorted, []).append(int(g_to))
            widx_map.setdefault(f_sorted, []).append(g_i[ord2])

        widx_map = {k: np.asarray(v, dtype=np.int64) for k, v in widx_map.items()}

        calculated = set(self._inputs)
        purpose = set(self._inputs) | hidden | set(self._outputs)

        blocks: list[_Block] = []
        while calculated != purpose:
            progressed = False
            for f_tuple, to_nodes in pairs.items():
                if set(f_tuple).issubset(calculated) and not set(to_nodes).issubset(calculated):
                    progressed = True
                    calculated |= set(to_nodes)

                    nodes_map: dict[int, list[int]] = {}
                    for t in to_nodes:
                        nodes_map.setdefault(self._activs[t], []).append(t)

                    act_codes = np.fromiter(nodes_map.keys(), dtype=np.int64)
                    act_nodes = [np.asarray(v, dtype=np.int64) for v in nodes_map.values()]

                    blocks.append(
                        _Block(
                            fr=np.asarray(f_tuple, dtype=np.int64),
                            to=np.asarray(to_nodes, dtype=np.int64),
                            widx=widx_map[f_tuple],
                            act_codes=act_codes,
                            act_nodes_list=act_nodes,
                        )
                    )
            if not progressed:
                raise RuntimeError("Topological build stalled (cycle or disconnected nodes).")
        return blocks

    def compile_torch(self, device: "torch.device" | str | None = None) -> "Net":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for compile_torch. "
                "Install with: pip install thefittest[torch]"
            )
        dev = torch.device(device) if device is not None else self._weights.device

        all_nodes = set(self._inputs)
        if self._hidden_layers:
            all_nodes |= set.union(*self._hidden_layers)
        all_nodes |= set(self._outputs)
        self.n_nodes = (max(all_nodes) + 1) if all_nodes else 0

        self.inputs = torch.as_tensor(sorted(self._inputs), dtype=torch.long, device=dev)
        self.outputs = torch.as_tensor(sorted(self._outputs), dtype=torch.long, device=dev)

        blocks = self._build_order()
        for b in blocks:
            b.to(dev)
        self.blocks = blocks
        return self

    def ensure_compiled(self) -> None:
        if (
            self.blocks is None
            or self.inputs is None
            or self.outputs is None
            or self.n_nodes is None
        ):
            self.compile_torch()

    def forward(
        self,
        X: "torch.Tensor",
        weights: "torch.Tensor" | None = None,
        keep_weight_dim: bool = False,
        autocast_input: bool = True,
    ) -> "torch.Tensor":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for forward. " "Install with: pip install thefittest[torch]"
            )
        base_w = self._weights if weights is None else weights
        w = base_w.view(1, -1) if base_w.ndim == 1 else base_w
        W = w.shape[0]

        if autocast_input:
            X = X.to(device=w.device, dtype=w.dtype)
        else:
            if X.device != w.device:
                raise RuntimeError(f"X on {X.device}, weights on {w.device}. Move X to {w.device}.")
            if X.dtype != w.dtype:
                raise RuntimeError(f"dtype mismatch: X {X.dtype} vs weights {w.dtype}.")

        B = X.shape[0]

        vals = torch.zeros(W, self.n_nodes, B, device=X.device, dtype=X.dtype)
        if X.shape[1] == self.inputs.numel():
            X_in = X
        else:
            X_in = X.index_select(1, self.inputs)

        vals[:, self.inputs, :] = X_in.T.unsqueeze(0).expand(W, -1, -1)

        for blk in self.blocks:
            wid_flat = blk.weight_idx.reshape(1, -1).expand(W, -1)
            w_blk = w.gather(1, wid_flat).view(W, blk.weight_idx.shape[0], blk.weight_idx.shape[1])

            z = torch.matmul(w_blk, vals[:, blk.from_idx, :])
            vals[:, blk.to_idx, :] = z

            for i in range(blk.act_codes.numel()):
                k = int(blk.act_nodes_len[i])
                if k == 0:
                    continue
                idx = blk.act_nodes[i, :k]
                code = int(blk.act_codes[i])
                vals[:, idx, :] = _act(code, vals[:, idx, :])

        out = vals[:, self.outputs, :].permute(0, 2, 1)
        if W == 1 and not keep_weight_dim:
            out = out.squeeze(0)
        return out

    def get_graph(self) -> Dict:
        if not TORCH_AVAILABLE or self._weights is None:
            raise ImportError(
                "PyTorch is required for get_graph method. "
                "Install it with: pip install thefittest[torch]"
            )

        input_color_code = (0.11, 0.67, 0.47, 1)
        hidden_color_code = (0.0, 0.74, 0.99, 1)
        output_color_code = (0.94, 0.50, 0.50, 1)

        weights_scale = minmax_scale(self._weights.detach().cpu().numpy())
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
        colors[:len_i] = input_color_code

        n = len_i
        for i, layer in enumerate(self._hidden_layers):
            nodes.extend(list(layer))
            positions[n : n + len(layer)][:, 0] = i + 1
            positions[n : n + len(layer)][:, 1] = np.arange(len(layer)) - len(layer) / 2
            colors[n : n + len(layer)] = hidden_color_code
            n += len(layer)

        nodes.extend(list(self._outputs))
        positions[n : n + len_o][:, 0] = len(self._hidden_layers) + 1
        positions[n : n + len_o][:, 1] = np.arange(len_o) - len_o / 2
        colors[n : n + len_o] = output_color_code

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

    def plot(self, ax=None) -> None:
        import networkx as nx

        graph = self.get_graph()
        G = nx.Graph()
        G.add_nodes_from(graph["nodes"])
        G.add_edges_from(graph["connects"])

        connects_label = {}
        for i, connects_i in enumerate(graph["connects"]):
            connects_label[tuple(connects_i)] = i

        nx.draw_networkx_nodes(
            G,
            pos=graph["positions"],
            node_color=graph["colors"],
            edgecolors="black",
            linewidths=0.5,
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos=graph["positions"],
            style="-",
            edge_color=graph["weights_colors"],
            ax=ax,
            alpha=0.5,
        )

        nx.draw_networkx_labels(G, graph["positions"], graph["labels"], font_size=10, ax=ax)


class HiddenBlock:
    def __init__(self, max_size: int) -> None:
        self._activ = random_sample(5, 1, True)[0]
        self._size = random_sample(max_size - 1, 1, True)[0] + 1

    def __str__(self) -> str:
        return "{}{}".format(ACTIVATION_NAME[self._activ], self._size)


class NetEnsemble:
    def __init__(self, nets: NDArray, meta_algorithm: Optional[Net] = None):
        self._nets = nets
        self._meta_algorithm = meta_algorithm
        self._meta_tree = None

    def __len__(self) -> int:
        return len(self._nets)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NetEnsemble):
            if len(self) != len(other):
                return False
            elif any((net_i != net_j for net_i, net_j in zip(self._nets, other._nets))):
                return False
            else:
                return True
        return NotImplemented

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
