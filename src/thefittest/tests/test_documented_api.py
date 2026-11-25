import numpy as np
import pytest

from ..base import init_symbolic_regression_uniset
from ..regressors import GeneticProgrammingRegressor


@pytest.mark.parametrize("name, expected", [("logabs", np.log(4.0)), ("sqrtabs", 2.0)])
def test_documented_symbolic_functions(name: str, expected: float) -> None:
    x = np.array([[-4.0], [4.0]])
    uniset = init_symbolic_regression_uniset(x, functional_set_names=(name,))
    function = uniset._functional_set[-1][0]._value
    np.testing.assert_allclose(function(x[:, 0]), [expected, expected])


def test_regressor_wildcard_exports() -> None:
    namespace = {}
    exec("from thefittest.regressors import *", namespace)
    assert namespace["GeneticProgrammingRegressor"] is GeneticProgrammingRegressor
