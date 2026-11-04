import numpy as np
import pytest

from ..base import Net
from ..base._gpnn import fitness_function_structure, fitness_function_structure_multiobjective
from ..base._gpnn import net_complexity
from ..base._net import ACTIV_NAME_INV

torch = pytest.importorskip("torch")


def make_population(task_type: str) -> np.ndarray:
    outputs = {2, 3} if task_type == "classification" else {2}
    activation = "softmax" if task_type == "classification" else "ln"
    nets = []
    for inputs in ({0}, {0, 1}):
        connections = np.array(
            [(i, j) for i in sorted(inputs) for j in sorted(outputs)], dtype=np.int64
        )
        net = Net(
            inputs=inputs,
            outputs=outputs,
            connects=connections,
            weights=torch.linspace(0.1, 0.3, len(connections)),
            activs={output: ACTIV_NAME_INV[activation] for output in outputs},
        )
        nets.append(net.compile_torch())
    return np.array(nets, dtype=object)


@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("complexity", ["n_connections", "n_inputs"])
def test_multiobjective_fitness_preserves_existing_network_fitness(
    task_type: str, complexity: str
) -> None:
    population = make_population(task_type)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.5], [-1.0, 0.0]])
    targets = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        if task_type == "classification"
        else torch.tensor([0.0, 1.0, -1.0])
    )
    expected = fitness_function_structure(population, x, targets, 0.2, task_type=task_type)
    actual = fitness_function_structure_multiobjective(
        population, x, targets, 0.2, complexity=complexity, task_type=task_type
    )
    assert actual.shape == (2, 2) and actual.dtype == np.float64
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(
        actual[:, 1], [net_complexity(net, complexity) for net in population]
    )


def test_multiobjective_fitness_reuses_scalar_cache_and_restores_weights() -> None:
    population = make_population("regression")[:1]
    net = population[0]
    weights = net._weights.clone()
    cache = {net.signature(): {"fitness": 7.0, "weights": weights}}
    net._weights = torch.zeros_like(weights)
    result = fitness_function_structure_multiobjective(
        population,
        torch.ones((3, 2)),
        torch.zeros(3),
        0.2,
        complexity="n_inputs",
        fitness_cache=cache,
    )
    np.testing.assert_array_equal(result, [[7.0, 1.0]])
    assert torch.equal(net._weights, weights)
    assert cache[net.signature()]["fitness"] == 7.0
