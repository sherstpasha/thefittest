import numpy as np
import pytest

from ..base import Tree, TerminalNode, init_symbolic_regression_uniset
from ..base._gp import tree_complexity, fitness_function_gp_multiobjective


@pytest.mark.parametrize(
    "criterion, expected",
    [("n_nodes", 4), ("n_levels", 3), ("n_functions", 2), ("n_terminals", 2)],
)
def test_tree_complexity_counts_structure(criterion: str, expected: int) -> None:
    x = np.arange(4, dtype=float)[:, None]
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("cos", "add"))
    tree = Tree(
        [
            uniset._functional_set[1][0],
            uniset._functional_set[2][0],
            TerminalNode(x[:, 0], "x0"),
            TerminalNode(1.0, "constant"),
        ]
    )
    assert tree_complexity(tree, criterion) == expected


def test_terminal_complexity_and_invalid_criterion() -> None:
    tree = Tree([TerminalNode(1.0, "constant")])
    assert tree_complexity(tree, "n_levels") == 1
    assert tree_complexity(tree, "n_functions") == 0
    assert tree_complexity(tree, "n_terminals") == 1
    with pytest.raises(ValueError, match="Unknown tree complexity criterion"):
        tree_complexity(tree, "unknown")


def test_multiobjective_fitness_reuses_scalar_cache_across_complexity_criteria() -> None:
    x = np.array([[1.0], [2.0]])
    y = x[:, 0].copy()
    uniset = init_symbolic_regression_uniset(x, functional_set_names=("add",))
    trees = np.array(
        [
            Tree([TerminalNode(x[:, 0], "x0")]),
            Tree(
                [
                    uniset._functional_set[2][0],
                    TerminalNode(x[:, 0], "x0"),
                    TerminalNode(1.0, "constant"),
                ]
            ),
        ],
        dtype=object,
    )
    cache = {}
    size_fitness = fitness_function_gp_multiobjective(trees, y, "n_nodes", cache)
    np.testing.assert_allclose(size_fitness, [[0.0, 1.0], [1.0, 3.0]])
    assert size_fitness.dtype == np.float64
    assert cache == {trees[0].signature(): 0.0, trees[1].signature(): 1.0}
    terminal_fitness = fitness_function_gp_multiobjective(trees, y, "n_terminals", cache)
    np.testing.assert_allclose(terminal_fitness, [[0.0, 1.0], [1.0, 2.0]])
    assert len(cache) == 2
