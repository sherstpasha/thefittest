from ..benchmarks import BanknoteDataset
from ..benchmarks import BreastCancerDataset
from ..benchmarks import CreditRiskDataset
from ..benchmarks import DigitsDataset
from ..benchmarks import IrisDataset
from ..benchmarks import UserKnowladgeDataset
from ..benchmarks import WineDataset


def test_IrisDataset():
    data = IrisDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (150, 4)
    assert y.shape == (150,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 4
    assert len(y_names) == 3


def test_WineDataset():
    data = WineDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (178, 13)
    assert y.shape == (178,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 13
    assert len(y_names) == 3


def test_BreastCancerDataset():
    data = BreastCancerDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (569, 30)
    assert y.shape == (569,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 30
    assert len(y_names) == 2


def test_DigitsDataset():
    data = DigitsDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (5620, 64)
    assert y.shape == (5620,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 64
    assert len(y_names) == 10


def test_CreditRiskDataset():
    data = CreditRiskDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (2000, 3)
    assert y.shape == (2000,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 3
    assert len(y_names) == 2


def test_UserKnowladgeDataset():
    data = UserKnowladgeDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (403, 5)
    assert y.shape == (403,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 5
    assert len(y_names) == 5


def test_BanknoteDataset():
    data = BanknoteDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (403, 5)
    assert y.shape == (403,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 5
    assert len(y_names) == 4
