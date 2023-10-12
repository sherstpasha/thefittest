from __future__ import annotations

import random
from collections import Counter
from collections import defaultdict
from inspect import signature
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..tools import find_end_subtree_from_i
from ..tools import find_id_args_from_i
from ..tools import get_levels_tree_from_i


FUNCTIONAL_COLOR_CODE = (1, 0.72, 0.43, 1)
TERMINAL_COLOR_CODE = (0.21, 0.76, 0.56, 1)


class Operator:
    def __init__(self, formula: str, name: str, sign: str) -> None:
        self._formula = formula
        self.__name__ = name
        self._sign = sign

    def _write(self, *args: Any) -> str:
        formula = self._formula.format(*args)
        return formula

    def __call__(self, *args: Any) -> None:
        pass


class Node:
    def __init__(self, value: Any, name: str, sign: str, n_args: int) -> None:
        self._value = value
        self._name = name
        self._sign = sign
        self._n_args = n_args

    def __str__(self) -> str:
        return str(self._sign)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self._name == other._name


class FunctionalNode(Node):
    def __init__(self, value: Any, sign: Optional[str] = None) -> None:
        Node.__init__(
            self,
            value=value,
            name=value.__name__,
            sign=sign or value._sign,
            n_args=len(signature(value.__call__).parameters),
        )


class TerminalNode(Node):
    def __init__(self, value: Any, name: str) -> None:
        Node.__init__(self, value=value, name=name, sign=name, n_args=0)


class EphemeralConstantNode(TerminalNode):
    def __init__(self, value: Any, name: str) -> None:
        TerminalNode.__init__(self, value=value, name=name)


class EphemeralNode(Node):
    def __init__(self, generator: Callable) -> None:
        self._generator = generator
        Node.__init__(
            self,
            value=self._generator,
            name=str(self._generator.__name__),
            sign=str(self._generator.__name__),
            n_args=0,
        )

    def __call__(self) -> EphemeralConstantNode:
        value = self._generator()
        node = EphemeralConstantNode(value=value, name=str(value))
        return node


class DualNode(Operator):
    def __init__(
        self, top_node: Union[TerminalNode, EphemeralNode], bottom_node: FunctionalNode
    ) -> None:
        self._top_node = top_node
        self._bottom_node = bottom_node

        name = f"{self._top_node._name}\n{self._bottom_node._name}"
        sign = f"{self._top_node._sign}\n{self._bottom_node._sign}"
        formula = f"{name}|" + "({} + {})"

        Operator.__init__(self, formula=formula, name=name, sign=sign)
        self.__call__ = self._bottom_node._value.__call__


class UniversalSet:
    def __init__(
        self,
        functional_set: Tuple[FunctionalNode, ...],
        terminal_set: Tuple[Union[TerminalNode, EphemeralNode], ...],
    ) -> None:
        self._functional_set = self._define_functional_set(functional_set)
        self._terminal_set = tuple(terminal_set)

    def _define_functional_set(self, functional_set: Tuple[FunctionalNode, ...]) -> defaultdict:
        _functional_set_list = defaultdict(list, {-1: list(functional_set)})
        _functional_set_tuple = defaultdict(tuple)
        for unit in functional_set:
            _functional_set_list[unit._n_args].append(unit)
        for key, value in _functional_set_list.items():
            _functional_set_tuple[key] = tuple(value)
        return _functional_set_tuple

    def _random_terminal_or_ephemeral(self) -> Union[EphemeralConstantNode, TerminalNode]:
        choosen = random.choice(self._terminal_set)
        if isinstance(choosen, EphemeralNode):
            return choosen()
        else:
            return choosen

    def _random_functional(self, n_args: int = -1) -> FunctionalNode:
        n_args_functionals = self._functional_set[n_args]
        node = random.choice(n_args_functionals)
        return node


class EnsembleUniversalSet(UniversalSet):
    def __init__(
        self,
        functional_set: Tuple[FunctionalNode, ...],
        terminal_set: Tuple[Union[TerminalNode, EphemeralNode], ...],
    ) -> None:
        UniversalSet.__init__(self, functional_set, terminal_set)
        self._functional_set_proba = self._define_functional_set_proba()

    def _define_functional_set_proba(self: EnsembleUniversalSet) -> defaultdict:
        functional_set_proba = defaultdict(list)
        for n_args, value in self._functional_set.items():
            count = Counter([type(node._value) for node in value])
            for node in value:
                type_ = type(node._value)
                functional_set_proba[n_args].append(count[type_])
            proba = np.array(functional_set_proba[n_args], dtype=np.float64)
            functional_set_proba[n_args] = 1 / proba

        return functional_set_proba

    def _random_functional(self, n_args: int = -1) -> FunctionalNode:
        from ..tools.random import random_weighted_sample

        n_args_functionals = self._functional_set[n_args]
        weights = self._functional_set_proba[n_args]
        index = random_weighted_sample(weights=weights, quantity=1, replace=True)[0]
        choosen = n_args_functionals[index]

        if isinstance(choosen._value, DualNode):
            if isinstance(choosen._value._top_node, EphemeralNode):
                choosen = FunctionalNode(
                    DualNode(
                        top_node=choosen._value._top_node(), bottom_node=choosen._value._bottom_node
                    )
                )
        return choosen


class Tree:
    def __init__(
        self,
        nodes: List[Union[FunctionalNode, TerminalNode, EphemeralNode]],
        n_args: Optional[Union[List[int], NDArray[np.int64]]] = None,
    ) -> None:
        self._nodes = nodes
        if n_args is None:
            self._n_args = self._init_n_args()
        else:
            self._n_args = np.array(n_args, dtype=np.int64)

    def _init_n_args(self) -> NDArray[np.int64]:
        n_args = np.empty(shape=len(self._nodes), dtype=np.int64)
        for i, node_i in enumerate(self._nodes):
            n_args[i] = node_i._n_args
        return n_args

    def __len__(self) -> int:
        return len(self._nodes)

    def __str__(self) -> str:
        pack: List[str] = []
        for node in reversed(self._nodes):
            args = []
            for _ in range(node._n_args):
                args.append(pack.pop())
            if isinstance(node, FunctionalNode):
                pack.append(node._value._write(*args))
            else:
                pack.append(node._name)
        return pack[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented
        else:
            if len(self) != len(other):
                return False
            else:
                for node_1, node_2 in zip(self._nodes, other._nodes):
                    if node_1 != node_2:
                        return False
            return True

    def __call__(self) -> Any:
        pack: Any = []
        for node in reversed(self._nodes):
            args = []
            for _ in range(node._n_args):
                args.append(pack.pop())
            if isinstance(node, FunctionalNode):
                pack.append(node._value(*args))
            else:
                pack.append(node._value)
        return pack[0]

    def copy(self) -> Tree:
        return Tree(self._nodes.copy(), self._n_args.copy())

    def set_terminals(self, **kwargs: Any) -> Tree:
        tree_copy = self.copy()
        for i, node in enumerate(tree_copy._nodes):
            if isinstance(node, TerminalNode):
                for name, value in kwargs.items():
                    if node._name == name:
                        tree_copy._nodes[i] = TerminalNode(value=value, name=node._name)
        return tree_copy

    def subtree_id(self, index: int) -> Tuple[int, int]:
        n_index = find_end_subtree_from_i(np.int64(index), self._n_args)
        return (index, n_index)

    def subtree(self, index: int) -> Tree:
        n_index = find_end_subtree_from_i(np.int64(index), self._n_args)
        new_tree = Tree(self._nodes[index:n_index].copy(), self._n_args[index:n_index].copy())
        return new_tree

    def concat(self, index: int, other_tree: Tree) -> Tree:
        to_return = self.copy()
        left, right = self.subtree_id(index)
        to_return._nodes[left:right] = other_tree._nodes.copy()
        to_return._n_args = np.r_[
            to_return._n_args[:left], other_tree._n_args.copy(), to_return._n_args[right:]
        ]
        return to_return

    def get_args_id(self, index: int) -> NDArray[np.int64]:
        args_id = find_id_args_from_i(np.int64(index), self._n_args)
        return args_id

    def get_levels(self, index: int) -> NDArray[np.int64]:
        return get_levels_tree_from_i(np.int64(index), self._n_args)

    def get_max_level(self) -> np.int64:
        return max(self.get_levels(0))

    def get_graph(self, keep_id: bool = False) -> Dict:
        pack: List[int] = []
        edges = []
        nodes = []
        labels = {}
        for i, node in enumerate(reversed(self._nodes)):
            index = len(self) - i - 1
            if keep_id:
                labels[index] = str(index) + ". " + node._sign[:60]
            else:
                labels[index] = node._sign[:60]

            nodes.append(index)

            for _ in range(node._n_args):
                edges.append((index, len(self) - pack.pop() - 1))
            pack.append(i)

        edges.reverse()
        nodes.reverse()

        levels = self.get_levels(0)
        colors = np.zeros(shape=(len(nodes), 4))
        pos = np.zeros(shape=(len(self), 2))
        for i, lvl_i in enumerate(levels):
            total = 0
            cond = lvl_i == np.array(levels)
            h = 1 / (1 + np.sum(cond))
            arange = np.arange(len(pos))[cond]

            for a_j in arange:
                total += h
                pos[a_j][0] = total

            pos[i][1] = -lvl_i

            if isinstance(self._nodes[i], FunctionalNode):
                colors[i] = FUNCTIONAL_COLOR_CODE
            else:
                colors[i] = TERMINAL_COLOR_CODE

        to_return = {"edges": edges, "labels": labels, "nodes": nodes, "pos": pos, "colors": colors}
        return to_return
