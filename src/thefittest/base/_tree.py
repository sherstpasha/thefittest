from __future__ import annotations

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

from ._net import HiddenBlock
from ..utils import find_end_subtree_from_i
from ..utils import find_id_args_from_i
from ..utils import get_levels_tree_from_i
from ..utils import common_region
from ..utils import common_region_two_trees
from ..utils.random import random_sample
from ..utils.random import random_weighted_sample
from ..utils.random import uniform


MIN_VALUE = np.finfo(np.float64).min
MAX_VALUE = np.finfo(np.float64).max
FUNCTIONAL_COLOR_CODE = (1, 0.72, 0.43, 1)
TERMINAL_COLOR_CODE = (0.21, 0.76, 0.56, 1)


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
        index = random_sample(len(self._terminal_set), 1, True)[0]
        choosen = self._terminal_set[index]
        if isinstance(choosen, EphemeralNode):
            return choosen()
        else:
            return choosen

    def _random_functional(self, n_args: int = -1) -> FunctionalNode:
        n_args_functionals = self._functional_set[n_args]
        index = random_sample(len(n_args_functionals), 1, True)[0]
        choosen = n_args_functionals[index]
        return choosen


class EnsembleUniversalSet(UniversalSet):
    def __init__(
        self,
        functional_set: Tuple[FunctionalNode, ...],
        terminal_set: Tuple[Union[TerminalNode, EphemeralNode], ...],
    ) -> None:
        UniversalSet.__init__(self, functional_set, terminal_set)
        self._functional_set_proba = self._define_functional_set_proba()

    def _define_functional_set_proba(self: EnsembleUniversalSet) -> Dict[int, NDArray[np.float64]]:
        functional_set_proba: Dict[int, NDArray[np.float64]] = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        for n_args, value in self._functional_set.items():
            count = Counter([type(node._value) for node in value])
            for node in value:
                type_ = type(node._value)
                functional_set_proba[n_args] = np.append(functional_set_proba[n_args], count[type_])
            proba = functional_set_proba[n_args]
            functional_set_proba[n_args] = 1 / proba
        return functional_set_proba

    def _random_functional(self, n_args: int = -1) -> FunctionalNode:
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
            raise TypeError(f"Cannot compare Tree with {type(other).__name__}")
        if len(self) != len(other):
            return False
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

    def get_common_region(self, other_trees: Union[List, NDArray]) -> Tuple:
        if len(other_trees) == 1:
            to_return = common_region_two_trees(self._n_args, other_trees[0]._n_args)
        else:
            trees = [self] + list(other_trees)
            to_return = common_region(trees)

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

    def plot(self, ax: Any = None) -> None:
        import networkx as nx

        graph = self.get_graph(keep_id=False)

        G = nx.Graph()
        G.add_nodes_from(graph["nodes"])
        G.add_edges_from(graph["edges"])

        nx.draw_networkx_nodes(
            G, graph["pos"], node_color=graph["colors"], edgecolors="black", linewidths=0.5, ax=ax
        )
        nx.draw_networkx_edges(G, graph["pos"], style="-", ax=ax)
        nx.draw_networkx_labels(G, graph["pos"], graph["labels"], font_size=10, ax=ax)

    @classmethod
    def full_growing_method(cls, uniset: UniversalSet, max_level: int) -> Tree:
        nodes: List[Union[FunctionalNode, TerminalNode, EphemeralNode]] = []
        levels = []
        n_args = []
        possible_steps = [1]
        previous_levels = [-1]
        level_i = -1
        while len(possible_steps):
            possible_steps[-1] = possible_steps[-1] - 1
            if possible_steps[-1] == 0:
                possible_steps.pop()
                level_i = previous_levels.pop() + 1
            else:
                level_i = previous_levels[-1] + 1
            levels.append(level_i)
            if level_i == max_level:
                nodes.append(uniset._random_terminal_or_ephemeral())
                n_args.append(0)
            else:
                nodes.append(uniset._random_functional())
                n_i = nodes[-1]._n_args
                n_args.append(n_i)
                possible_steps.append(n_i)
                previous_levels.append(level_i)
        tree = cls(nodes, n_args)
        return tree

    @classmethod
    def growing_method(cls, uniset: UniversalSet, max_level: int) -> Tree:
        nodes: List[Union[FunctionalNode, TerminalNode, EphemeralNode]] = []
        levels = []
        n_args = []
        possible_steps = [1]
        previous_levels = [-1]
        level_i = -1
        while len(possible_steps):
            possible_steps[-1] = possible_steps[-1] - 1
            if possible_steps[-1] == 0:
                possible_steps.pop()
                level_i = previous_levels.pop() + 1
            else:
                level_i = previous_levels[-1] + 1
            levels.append(level_i)

            if level_i == max_level:
                nodes.append(uniset._random_terminal_or_ephemeral())
                n_args.append(0)
            elif level_i == 0:
                nodes.append(uniset._random_functional())
                n_i = nodes[-1]._n_args
                n_args.append(n_i)
                possible_steps.append(n_i)
                previous_levels.append(level_i)
            else:
                if uniform(low=0, high=1, size=1)[0] < 0.5:
                    nodes.append(uniset._random_terminal_or_ephemeral())
                else:
                    nodes.append(uniset._random_functional())
                n_i = nodes[-1]._n_args
                n_args.append(n_i)

                if n_i > 0:
                    possible_steps.append(n_i)
                    previous_levels.append(level_i)
        tree = cls(nodes, n_args)
        return tree

    @classmethod
    def random_tree(cls, uniset: UniversalSet, max_level: int) -> Tree:
        if uniform(low=0, high=1, size=1)[0] < 0.5:
            tree = cls.full_growing_method(uniset, max_level)
        else:
            tree = cls.growing_method(uniset, max_level)
        return tree


class Cos(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="cos({})", name="cos", sign="cos")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.cos(x)
        return result


class Sin(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="sin({})", name="sin", sign="sin")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.sin(x)
        return result


class Add(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="({} + {})", name="add", sign="+")

    def __call__(
        self, x: Union[float, NDArray[np.float64]], y: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        result = x + y
        return result


class Sub(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="({} - {})", name="sub", sign="-")

    def __call__(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        result = x - y
        return result


class Neg(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="-{}", name="neg", sign="-")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = -x
        return result


class Mul(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="({} * {})", name="mul", sign="*")

    def __call__(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        result = x * y
        return result


class Pow2(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="({}**2)", name="pow2", sign="**2")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.clip(x**2, MIN_VALUE, MAX_VALUE)
        return result


class Div(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="({}/{})", name="div", sign="/")

    def __call__(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        result: Union[float, np.ndarray]
        if isinstance(y, np.ndarray):
            result = np.divide(x, y, out=np.ones_like(y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                result = 0.0
            else:
                result = x / y
        result = np.clip(result, MIN_VALUE, MAX_VALUE)
        return result


class Inv(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="(1/{})", name="Inv", sign="1/")

    def __call__(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result: Union[float, np.ndarray]

        if isinstance(y, np.ndarray):
            result = np.divide(1, y, out=np.ones_like(y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                result = 1
            else:
                result = 1 / y
        result = np.clip(result, MIN_VALUE, MAX_VALUE)
        return result


class LogAbs(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="log(abs({}))", name="log(abs)", sign="log(abs)")

    def __call__(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        y_ = np.abs(y)
        if isinstance(y_, np.ndarray):
            result = np.log(y_, out=np.ones_like(y_, dtype=np.float64), where=y_ != 0)
        else:
            if y_ == 0:
                result = 1
            else:
                result = np.log(y_)
        return result


class Exp(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="exp({})", name="exp", sign="exp")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.clip(np.exp(x), MIN_VALUE, MAX_VALUE)
        return result


class SqrtAbs(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="sqrt(abs({}))", name="sqrt(abs)", sign="sqrt(abs)")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.sqrt(np.abs(x))
        return result


class Abs(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="abs({})", name="abs()", sign="abs()")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.abs(x)
        return result


class More(Operator):
    def __init__(self) -> None:
        Operator.__init__(self, formula="({} > {})", name="more", sign=">")

    def __call__(
        self, x: Union[float, NDArray[np.float64]], y: Union[float, NDArray[np.float64]]
    ) -> Union[bool, NDArray[np.bool_]]:
        result = x > y
        return result


NET_FUNCTION_NAME = {"add": Add, "more": More}


def init_symbolic_regression_uniset(
    X: NDArray[np.float64],
    functional_set_names: Tuple[str, ...] = ("cos", "sin", "add", "inv", "neg", "mul"),
    ephemeral_node_generators: Optional[Tuple[Callable, ...]] = None,
):
    SYMBOLIC_FUNCTION_NAME = {
        "cos": Cos,
        "sin": Sin,
        "add": Add,
        "sub": Sub,
        "neg": Neg,
        "mul": Mul,
        "div": Div,
        "inv": Inv,
    }

    for func_name in functional_set_names:
        if func_name not in SYMBOLIC_FUNCTION_NAME:
            raise ValueError(
                f"Invalid function name '{func_name}'. Available values are: {', '.join(SYMBOLIC_FUNCTION_NAME.keys())}"
            )

    uniset: UniversalSet
    terminal_set: Union[
        List[Union[TerminalNode, EphemeralNode]], Tuple[Union[TerminalNode, EphemeralNode]]
    ]
    functional_set: Union[
        List[Union[TerminalNode, EphemeralNode]], Tuple[Union[TerminalNode, EphemeralNode]]
    ] = []
    n_dimension: int = X.shape[1]

    for functional_name in functional_set_names:
        function = SYMBOLIC_FUNCTION_NAME[functional_name]
        functional_set.append(FunctionalNode(function()))

    terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
    if ephemeral_node_generators is not None:
        for generator in ephemeral_node_generators:
            terminal_set.append(EphemeralNode(generator))

    uniset = UniversalSet(tuple(functional_set), tuple(terminal_set))
    return uniset


def init_net_uniset(
    n_variables: int, input_block_size: int, max_hidden_block_size: int, offset: bool = True
):
    if offset:
        n_dimension = n_variables - 1
    else:
        n_dimension = n_variables

    cut_id: NDArray[np.int64] = np.arange(
        input_block_size, n_dimension, input_block_size, dtype=np.int64
    )

    variables_pool: List = np.split(np.arange(n_dimension), cut_id)
    functional_set = (
        FunctionalNode(NET_FUNCTION_NAME["add"]()),
        FunctionalNode(NET_FUNCTION_NAME["more"]()),
    )

    def random_hidden_block() -> HiddenBlock:
        return HiddenBlock(max_hidden_block_size)

    terminal_set: List[Union[TerminalNode, EphemeralNode]] = [
        TerminalNode(set(variables), "in{}".format(i)) for i, variables in enumerate(variables_pool)
    ]

    if offset:
        terminal_set.append(
            TerminalNode(value={(n_dimension)}, name="in{}".format(len(variables_pool)))
        )
    terminal_set.append(EphemeralNode(random_hidden_block))

    uniset = UniversalSet(functional_set, tuple(terminal_set))
    return uniset
