from __future__ import annotations
from typing import Any, List, Optional, Tuple, Type, Union
from sklearn.base import clone
from ..base import Tree, EphemeralNode, FunctionalNode, TerminalNode, UniversalSet
from ..base._model import Model
from ..base._tree import DualNode, EphemeralConstantNode
from ..optimizers import (
    GeneticProgramming,
    SelfCGP,
    SelfCGA,
    DifferentialEvolution,
    GeneticAlgorithm,
    SHADE,
    SHAGA,
    jDE,
)
from ..tools.metrics import root_mean_square_error
from thefittest.tools.operators import Add, Sub, Mul, Div
import numpy as np
from numpy.typing import NDArray


def fitness_function(
    trees: NDArray, y: NDArray[np.float32], meta_model: Optional[Any] = None
) -> NDArray[np.float32]:

    fitness = []
    for i_group, tree_group in enumerate(trees):
        # Проверка наличия EphemeralConstantNode внутри DualNode
        for i_tree, tree in enumerate(tree_group):
            for node in tree._nodes:
                print(node)
                if isinstance(node, FunctionalNode) and isinstance(node._value, DualNode):
                    top_node = node._value._top_node
                    if isinstance(top_node, EphemeralConstantNode):
                        print(
                            f"[DEBUG] group {i_group}, tree {i_tree}: DualNode with EphemeralConstantNode ({top_node._name})"
                        )
                    else:
                        print(
                            f"[DEBUG] group {i_group}, tree {i_tree}: DualNode with non-ephemeral top ({type(top_node).__name__})"
                        )

        X_features = [tree() * np.ones(len(y)) for tree in tree_group]
        X_mat = np.vstack(X_features).T

        if meta_model is None:
            raise ValueError("meta_model must be provided.")
        model = clone(meta_model)
        model.fit(X_mat, y)
        y_pred = model.predict(X_mat)
        fitness_val = root_mean_square_error(y, y_pred.astype(np.float32))
        fitness.append(fitness_val)

    return np.array(fitness, dtype=np.float32)


def genotype_to_phenotype(x):
    def split_tree(tree: Tree) -> Tuple[Tree, Tree]:
        new_tree = Tree([])
        remain_tree = tree.copy()
        for i, node in enumerate(reversed(tree._nodes)):
            index = len(tree) - i - 1
            if isinstance(node._value, DualNode):
                begin, end = tree.subtree_id(index)
                new_nodes = tree._nodes[begin:end].copy()
                new_nodes[0] = node._value._bottom_node
                new_tree = Tree(nodes=new_nodes)

                remain_nodes = tree._nodes[:begin].copy() + tree._nodes[end - 1 :].copy()
                remain_nodes[begin] = node._value._top_node
                remain_tree = Tree(nodes=remain_nodes)
                break
        return remain_tree, new_tree

    phenotypes = []
    for tree in x:
        trees = []
        remain_tree, new_tree = split_tree(tree)
        if len(new_tree) > 0:
            trees.append(new_tree)
        while True:
            remain_tree, new_tree = split_tree(remain_tree)
            if len(new_tree) > 0:
                trees.append(new_tree)
            else:
                break
        if len(remain_tree) > 0:
            trees.append(remain_tree)
        phenotypes.append(trees)
    arr = np.empty(len(phenotypes), dtype=object)
    for i, item in enumerate(phenotypes):
        arr[i] = item
    return arr


def generator1() -> float:
    return np.round(np.random.uniform(0, 10), 4)


def generator2() -> int:
    return np.random.randint(0, 10)


class SymbolicRegressionGP_DUAL(Model):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        uniset: Optional[UniversalSet] = None,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        p_dual: float = 0.5,
        meta_model: Optional[Any] = None,  # sklearn-совместимая модель
    ):
        super().__init__()
        self._iters = iters
        self._pop_size = pop_size
        self._uniset = uniset
        self._optimizer_args = optimizer_args or {}
        self._optimizer_class = optimizer
        self._optimizer = None
        self._p_dual = p_dual
        self._meta_model = meta_model
        self._fitted_meta_model = None

    def _get_uniset(self, X: NDArray[np.float32]) -> UniversalSet:
        n_dimension = X.shape[1]
        base_functionals = [FunctionalNode(op()) for op in [Add, Sub, Mul, Div]]

        # Реальные переменные
        terminal_variables = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]

        # Шаблоны эфемерных констант
        ephemeral_templates = [EphemeralNode(generator1), EphemeralNode(generator2)]

        # Всё множество терминалов
        terminal_set = terminal_variables + ephemeral_templates

        # Функциональное множество
        functional_set = base_functionals.copy()

        # Добавляем DualNode: и с переменными, и с эфемерными
        for term in terminal_set:
            for base_func in base_functionals:
                dual = DualNode(term, base_func)
                functional_set.append(FunctionalNode(dual))

        return UniversalSet(tuple(functional_set), tuple(terminal_set), p_dual=self._p_dual)

    def _fit(self, X: NDArray[np.float32], y: NDArray) -> SymbolicRegressionGP_DUAL:
        if self._meta_model is None:
            raise ValueError("meta_model must be provided.")

        uniset = self._uniset or self._get_uniset(X)
        args = self._optimizer_args.copy()
        args.update(
            {
                "fitness_function": lambda trees, y: fitness_function(trees, y, self._meta_model),
                "fitness_function_args": {"y": y},
                "minimization": True,
                "iters": self._iters,
                "pop_size": self._pop_size,
                "uniset": uniset,
                "genotype_to_phenotype": genotype_to_phenotype,
            }
        )
        self._optimizer = self._optimizer_class(**args)
        self._optimizer.fit()

        # Обучение финальной мета-модели на лучшем решении
        solution = self._optimizer.get_fittest()
        trees = [
            tree.set_terminals(**{f"x{i}": X[:, i] for i in range(X.shape[1])})
            for tree in solution["phenotype"]
        ]
        outputs = np.vstack([tree() * np.ones(len(X)) for tree in trees]).T
        self._fitted_meta_model = clone(self._meta_model)
        self._fitted_meta_model.fit(outputs, y)
        return self

    def _predict(self, X: NDArray[np.float32]) -> NDArray:
        if self._fitted_meta_model is None:
            raise RuntimeError("Model not fitted yet.")
        solution = self._optimizer.get_fittest()
        trees = [
            tree.set_terminals(**{f"x{i}": X[:, i] for i in range(X.shape[1])})
            for tree in solution["phenotype"]
        ]
        outputs = np.vstack([tree() * np.ones(len(X)) for tree in trees]).T
        return self._fitted_meta_model.predict(outputs)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray:
        if self._fitted_meta_model is None:
            raise RuntimeError("Model not fitted yet.")
        if not hasattr(self._fitted_meta_model, "predict_proba"):
            raise NotImplementedError("Meta-model does not support predict_proba.")
        solution = self._optimizer.get_fittest()
        trees = [
            tree.set_terminals(**{f"x{i}": X[:, i] for i in range(X.shape[1])})
            for tree in solution["phenotype"]
        ]
        outputs = np.vstack([tree() * np.ones(len(X)) for tree in trees]).T
        return self._fitted_meta_model.predict_proba(outputs)

    def get_optimizer(
        self,
    ) -> Union[
        DifferentialEvolution,
        GeneticAlgorithm,
        GeneticProgramming,
        jDE,
        SelfCGA,
        SelfCGP,
        SHADE,
        SHAGA,
    ]:
        return self._optimizer
