import random
from collections import defaultdict
from typing import Union
from typing import Callable
from typing import Optional
from typing import Any
from typing import Tuple
from typing import Dict
import numpy as np
from inspect import signature
from ..tools.numba_funcs import find_end_subtree_from_i
from ..tools.numba_funcs import find_id_args_from_i
from ..tools.numba_funcs import get_levels_tree_from_i


class Tree:
    def __init__(self,
                 nodes: list,
                 n_args: list) -> None:
        self._nodes = nodes
        self._n_args = np.array(n_args, dtype=np.int32)

    def __len__(self):
        return len(self._nodes)

    def __str__(self):
        pack = []
        for node in reversed(self._nodes):
            args = []
            for _ in range(node._n_args):
                args.append(pack.pop())
            if node.is_functional():
                pack.append(node._value._write(*args))
            else:
                pack.append(node._name)
        return pack[0]

    def __eq__(self, other) -> bool:
        if len(self) != len(other):
            return False
        else:
            for node_1, node_2 in zip(self._nodes, other._nodes):
                if node_1 != node_2:
                    return False
        return True

    def __call__(self):
        pack = []
        for node in reversed(self._nodes):
            args = []
            for _ in range(node._n_args):
                args.append(pack.pop())
            if node.is_functional():
                pack.append(node._value(*args))
            else:
                pack.append(node._value)
        return pack[0]

    def copy(self):
        return Tree(self._nodes.copy(), self._n_args.copy())

    def set_terminals(self,
                      args: Dict):
        tree_copy = self.copy()
        for i, node in enumerate(tree_copy._nodes):
            if node.is_terminal():
                for name, value in args.items():
                    if node._name == name:
                        tree_copy._nodes[i] = TerminalNode(value=value,
                                                           name=node._name)
        return tree_copy

    def subtree(self, index: np.int32, return_class=False):
        n_index = find_end_subtree_from_i(index, self._n_args)
        if return_class:
            new_tree = Tree(self._nodes[index:n_index].copy(),
                            self._n_args[index:n_index].copy())
            return new_tree
        return index, n_index

    def concat(self, index, other_tree):
        to_return = self.copy()
        left, right = self.subtree(index)
        to_return._nodes[left:right] = other_tree._nodes.copy()
        to_return._n_args = np.r_[to_return._n_args[:left],
                                  other_tree._n_args.copy(),
                                  to_return._n_args[right:]]
        return to_return

    def get_args_id(self, index):
        args_id = find_id_args_from_i(np.int32(index), self._n_args)
        return args_id

    def get_levels(self, index):
        return get_levels_tree_from_i(np.int32(index), self._n_args)

    def get_max_level(self):
        return max(self.get_levels(0))

    def get_graph(self, keep_id=False):
        pack = []
        edges = []
        nodes = []
        labels = {}
        for i, node in enumerate(reversed(self._nodes)):
            index = len(self) - i - 1
            if keep_id:
                labels[index] = str(index) + '. ' + node._sign[:6]
            else:
                labels[index] = node._sign[:6]

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
            h = 1/(1 + np.sum(cond))
            arange = np.arange(len(pos))[cond]

            for a_j in arange:
                total += h
                pos[a_j][0] = total

            pos[i][1] = -lvl_i

            if self._nodes[i].is_functional():
                colors[i] = (1, 0.72, 0.43, 1)
            else:
                colors[i] = (0.21, 0.76, 0.56, 1)

        to_return = {'edges': edges,
                     'labels': labels,
                     'nodes': nodes,
                     'pos': pos,
                     'colors': colors}
        return to_return


class Node:
    def __init__(self,
                 value: Union[Any, Callable],
                 name: str,
                 sign: str,
                 n_args: int) -> None:
        self._value = value
        self._name = name
        self._sign = sign
        self._n_args = n_args

    def __str__(self) -> str:
        return str(self._sign)

    def __eq__(self, other) -> bool:
        return self._name == other._name

    def is_functional(self) -> bool:
        return type(self) == FunctionalNode

    def is_ephemeral(self) -> bool:
        return type(self) == EphemeralNode

    def is_terminal(self) -> bool:
        return type(self) == TerminalNode


class FunctionalNode(Node):
    def __init__(self,
                 value: Callable,
                 sign: Optional[str] = None) -> None:
        Node.__init__(self,
                      value=value,
                      name=value.__name__,
                      sign=sign or value._sign,
                      n_args=len(signature(value).parameters))


class TerminalNode(Node):
    def __init__(self,
                 value: Any,
                 name: str) -> None:
        Node.__init__(self,
                      value=value,
                      name=name,
                      sign=name,
                      n_args=0)


class EphemeralConstantNode(TerminalNode):
    def __init__(self,
                 value: Any,
                 name: str) -> None:
        TerminalNode.__init__(self,
                              value=value,
                              name=name)


class EphemeralNode(Node):
    def __init__(self,
                 generator: Callable) -> None:
        self._generator = generator
        Node.__init__(self,
                      value=self._generator,
                      name=str(self._generator.__name__),
                      sign=str(self._generator.__name__),
                      n_args=0)

    def __call__(self) -> EphemeralConstantNode:
        value = self._generator()
        node = EphemeralConstantNode(value=value, name=str(value))
        return node


class UniversalSet:
    def __init__(self,
                 functional_set: Tuple[FunctionalNode],
                 terminal_set: Tuple[Union[TerminalNode, EphemeralNode]]) -> None:
        self._functional_set = self._define_functional_set(functional_set)
        self._terminal_set = tuple(terminal_set)

    def _define_functional_set(self,
                               functional_set: Tuple[FunctionalNode]) -> defaultdict:
        _functional_set = defaultdict(list, {'any': functional_set})
        for unit in functional_set:
            _functional_set[unit._n_args].append(unit)
        for key, value in _functional_set.items():
            _functional_set[key] = tuple(value)
        return _functional_set

    def _random_terminal_or_ephemeral(
            self) -> Union[EphemeralConstantNode, TerminalNode]:
        choosen = random.choice(self._terminal_set)
        node = choosen() if choosen.is_ephemeral() else choosen
        return node

    def _random_functional(self,
                           n_args='any') -> FunctionalNode:
        n_args_functionals = self._functional_set[n_args]
        node = random.choice(n_args_functionals)
        return node
