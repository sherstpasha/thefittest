import numpy as np
import pytest
from unittest.mock import patch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ..base._gpnn import net_complexity
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..regressors import GeneticProgrammingNeuralNetRegressor
from ..optimizers import GeneticProgramming, NSGPII, SelfCNSGPII, PDPNSGPII

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.filterwarnings("ignore:.*pop_size.*:RuntimeWarning")


@pytest.mark.parametrize(
    "estimator_type", [GeneticProgrammingNeuralNetClassifier, GeneticProgrammingNeuralNetRegressor]
)
@pytest.mark.parametrize("optimizer_type", [NSGPII, SelfCNSGPII, PDPNSGPII])
def test_gpnn_fits_with_multiobjective_optimizers(
    estimator_type: type, optimizer_type: type
) -> None:
    x = np.random.default_rng(42).normal(size=(24, 2))
    classifier = estimator_type is GeneticProgrammingNeuralNetClassifier
    y = np.where(x[:, 0] > 0, "positive", "negative") if classifier else x[:, 0] ** 2
    model = estimator_type(
        n_iter=2,
        pop_size=4,
        max_hidden_block_size=2,
        optimizer=optimizer_type,
        optimizer_args={"init_level": 2, "max_level": 3, "keep_history": True},
        weights_optimizer=torch.optim.Adam,
        weights_optimizer_args={"iters": 2, "lr": 0.02, "pop_size": 4},
        complexity="n_inputs" if classifier else "n_connections",
        model_selection="simplest" if optimizer_type is PDPNSGPII else "best_error",
        random_state=42,
    )
    assert model.fit(x, y) is model
    front = model.pareto_front_
    assert front["fitness"].shape == (len(front["phenotype"]), 2)
    assert np.isfinite(front["fitness"]).all()
    np.testing.assert_array_equal(
        front["fitness"][:, 1],
        [net_complexity(net, model.complexity) for net in front["phenotype"]],
    )
    errors, complexities = front["fitness"].T
    order = (
        np.lexsort((complexities, errors))
        if model.model_selection == "best_error"
        else np.lexsort((errors, complexities))
    )
    assert model.selected_net_index_ == order[0]
    assert str(model.get_tree()) == str(front["genotype"][order[0]])
    assert model.get_net() == front["phenotype"][order[0]]
    assert model.get_net() is not front["phenotype"][order[0]]
    assert model.predict(x).shape == y.shape
    if classifier:
        probabilities = model.predict_proba(x)
        assert probabilities.shape == (len(x), 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    else:
        assert np.isfinite(model.predict(x)).all()
    assert len(model.get_stats()["fitness"]) == 2
    assert clone(model).get_params(deep=False) == model.get_params(deep=False)
    saved_stats = model.get_stats()
    x_tensor = torch.as_tensor(np.hstack((x, np.ones((len(x), 1)))), dtype=torch.float64)
    with patch("thefittest.base._gpnn.train_net_weights", side_effect=AssertionError("Retraining")):
        for index in range(len(front["fitness"])):
            assert model.select_net(np.int64(index)) is model
            assert model.selected_net_index_ == index
            assert model.get_net() is not front["phenotype"][index]
            assert model.get_tree() is not front["genotype"][index]
            assert str(model.get_tree()) == str(front["genotype"][index])
            with torch.no_grad():
                expected = front["phenotype"][index].forward(x_tensor).cpu().numpy()
            if classifier:
                np.testing.assert_allclose(model.predict_proba(x), expected, rtol=1e-6, atol=1e-6)
            else:
                np.testing.assert_allclose(
                    model.predict(x), expected.reshape(-1), rtol=1e-6, atol=1e-6
                )
    assert model.get_stats() is saved_stats

    copied_front = model.get_pareto_front()
    original_weights = front["phenotype"][0]._weights.clone()
    copied_front["fitness"][:] = -999
    copied_front["phenotype"][0]._weights.zero_()
    copied_front["genotype"][0]._nodes.clear()
    assert np.all(front["fitness"] >= 0)
    assert torch.equal(front["phenotype"][0]._weights, original_weights)
    assert len(front["genotype"][0]) > 0

    for invalid, error in (
        (-1, IndexError),
        (len(front["fitness"]), IndexError),
        (True, TypeError),
        (0.5, TypeError),
    ):
        with pytest.raises(error):
            model.select_net(invalid)


def test_gpnn_single_objective_refit_clears_pareto_state() -> None:
    x = np.linspace(-1, 1, 12)[:, None]
    y = x[:, 0] ** 2
    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=1,
        pop_size=3,
        max_hidden_block_size=2,
        optimizer=NSGPII,
        optimizer_args={"init_level": 1, "max_level": 2},
        weights_optimizer=torch.optim.Adam,
        weights_optimizer_args={"iters": 1, "pop_size": 3, "lr": 0.01},
        random_state=42,
    ).fit(x, y)
    assert hasattr(model, "pareto_front_")
    model.set_params(optimizer=GeneticProgramming, complexity="unused", model_selection="unused")
    model.fit(x, y)
    assert not hasattr(model, "pareto_front_")
    assert not hasattr(model, "selected_net_index_")
    assert np.isfinite(model.predict(x)).all()
    with pytest.raises(ValueError, match="only after fitting with NSGPII"):
        model.get_pareto_front()
    with pytest.raises(ValueError, match="only after fitting with NSGPII"):
        model.select_net(0)


@pytest.mark.parametrize(
    "estimator_type", [GeneticProgrammingNeuralNetClassifier, GeneticProgrammingNeuralNetRegressor]
)
def test_pareto_methods_require_a_fitted_gpnn(estimator_type: type) -> None:
    model = estimator_type()
    with pytest.raises(NotFittedError):
        model.get_pareto_front()
    with pytest.raises(NotFittedError):
        model.select_net(0)
