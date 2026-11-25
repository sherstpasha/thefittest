import numpy as np
import pytest

from ..base import Net
from ..base._gpnn import net_complexity


@pytest.mark.parametrize(
    "criterion, expected",
    [("n_neurons", 4), ("n_connections", 5), ("n_layers", 3), ("n_inputs", 2)],
)
def test_net_complexity_counts_phenotype_structure(criterion: str, expected: int) -> None:
    net = Net(
        inputs={0, 1},
        hidden_layers=[{2, 3}, set(), {4}],
        outputs={5},
        connects=np.array([[0, 2], [1, 3], [2, 4], [3, 4], [4, 5]], dtype=np.int64),
    )
    assert net_complexity(net, criterion) == expected


def test_net_complexity_without_hidden_layers() -> None:
    net = Net(inputs={0, 1}, outputs={2}, connects=np.array([[0, 2], [1, 2]], dtype=np.int64))
    assert net_complexity(net) == 2
    assert net_complexity(net, "n_neurons") == 1
    assert net_complexity(net, "n_layers") == 1
    assert net_complexity(net, "n_inputs") == 2


def test_net_complexity_rejects_unknown_criterion() -> None:
    with pytest.raises(ValueError, match="Unknown network complexity criterion"):
        net_complexity(Net(), "unknown")
