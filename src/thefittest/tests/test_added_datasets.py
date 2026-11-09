from typing import Optional

import numpy as np
import pytest

from ..benchmarks import (
    AdultCensusDataset,
    CombinedCycleDataset,
    RingNormDataset,
    TextureDataset,
    TwoNormDataset,
)
from ..benchmarks._datasets import Dataset


@pytest.mark.parametrize(
    "dataset_type,shape,classes",
    [
        (TwoNormDataset, (7400, 20), 2),
        (RingNormDataset, (7400, 20), 2),
        (TextureDataset, (5500, 40), 11),
        (AdultCensusDataset, (32561, 95), 2),
        (CombinedCycleDataset, (680, 4), None),
    ],
)
def test_dataset_contract(
    dataset_type: type[Dataset], shape: tuple[int, int], classes: Optional[int]
) -> None:
    dataset = dataset_type()
    X, y = dataset.get_X(), dataset.get_y()
    assert X.shape == shape
    assert y.shape == (shape[0],)
    assert X.dtype == np.float64
    assert np.isfinite(X).all() and np.isfinite(y).all()
    assert set(dataset.get_X_names()) == set(range(shape[1]))
    if classes is not None:
        assert y.dtype == np.int64
        np.testing.assert_array_equal(np.unique(y), np.arange(classes))
        assert set(dataset.get_y_names()) == set(range(classes))
    else:
        assert y.dtype == np.float64
        assert dataset.get_y_names() == {0: "PE (Energy Output, MW)"}


def test_adult_feature_names_and_class_balance() -> None:
    dataset = AdultCensusDataset()
    assert dataset.get_X_names()[0] == "age"
    assert dataset.get_X_names()[94] == "native.country_Yugoslavia"
    np.testing.assert_array_equal(np.bincount(dataset.get_y()), [24720, 7841])


def test_texture_label_names_preserve_original_labels() -> None:
    dataset = TextureDataset()
    labels = [2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14]
    assert dataset.get_y_names() == {i: f"Class {label}" for i, label in enumerate(labels)}
    np.testing.assert_array_equal(np.bincount(dataset.get_y()), np.full(11, 500))
