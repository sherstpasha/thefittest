import numpy as np

from thefittest.utils._metrics import root_mean_square_error
from thefittest.utils._metrics import root_mean_square_error2d
from thefittest.utils._metrics import coefficient_determination
from thefittest.utils._metrics import coefficient_determination2d
from thefittest.utils._metrics import categorical_crossentropy
from thefittest.utils._metrics import categorical_crossentropy3d
from thefittest.utils._metrics import accuracy_score
from thefittest.utils._metrics import accuracy_score2d
from thefittest.utils._metrics import confusion_matrix
from thefittest.utils._metrics import recall_score
from thefittest.utils._metrics import recall_score2d
from thefittest.utils._metrics import precision_score
from thefittest.utils._metrics import precision_score2d
from thefittest.utils._metrics import f1_score
from thefittest.utils._metrics import f1_score2d


def test_root_mean_square_error():
    y_true = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_predict = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    assert root_mean_square_error(y_true, y_predict) == 0.0

    y_true = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_predict = np.array([2, 3, 4, 5, 6], dtype=np.float64)
    assert np.isclose(root_mean_square_error(y_true, y_predict), 1.0)


def test_root_mean_square_error2d():
    y_true = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_predict2d = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], dtype=np.float64)

    result = root_mean_square_error2d(y_true, y_predict2d)

    expected = np.array([0, 1, 2], dtype=np.float64)
    assert np.allclose(result, expected)


def test_coefficient_determination():
    y_true = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_predict = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    assert coefficient_determination(y_true, y_predict) == 1.0

    y_true = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_predict = np.array([3, 3, 3, 3, 3], dtype=np.float64)
    assert np.isclose(coefficient_determination(y_true, y_predict), 0.0)


def test_coefficient_determination2d():
    y_true = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_predict2d = np.array([[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]], dtype=np.float64)

    result = coefficient_determination2d(y_true, y_predict2d)

    expected = np.array([1, 0], dtype=np.float64)
    assert np.allclose(result, expected)


def test_categorical_crossentropy():
    target = np.array([[1, 0], [0, 1]], dtype=np.float64)
    output = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float64)
    assert np.isclose(categorical_crossentropy(target, output), 0.35667502866851847)

    target = np.array([[1, 0], [0, 1]], dtype=np.float64)
    output = np.array([[0.99999999, 0.00000001], [0.00000001, 0.99999999]], dtype=np.float64)
    assert np.isclose(categorical_crossentropy(target, output), 1.711e-06)


def test_categorical_crossentropy2d():
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float64)
    y_predict3d = np.array(
        [[[0.7, 0.3], [0.3, 0.7]], [[0.99999999, 0.00000001], [0.00000001, 0.99999999]]],
        dtype=np.float64,
    )

    result = categorical_crossentropy3d(y_true, y_predict3d)

    expected = np.array([0.35667502866851847, 1.711e-06], dtype=np.float64)
    assert np.allclose(result, expected)


def test_accuracy_score():
    y_true = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    y_predict = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert accuracy_score(y_true, y_predict) == 1.0

    y_true = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    y_predict = np.array([1, 2, 3, 4, 1], dtype=np.int64)
    assert accuracy_score(y_true, y_predict) == 0.8


def test_accuracy_score2d():
    y_true = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    y_predict2d = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 1]], dtype=np.int64)

    result = accuracy_score2d(y_true, y_predict2d)

    expected = np.array([1, 0.8], dtype=np.float64)
    assert np.allclose(result, expected)


def test_confusion_matrix():
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

    expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    assert np.array_equal(confusion_matrix(y_true, y_pred), expected)

    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
    expected = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])
    assert np.array_equal(confusion_matrix(y_true, y_pred), expected)


def test_recall_score():
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_predict = np.array([0, 2, 1, 0, 1, 2], dtype=np.int64)
    expected_recall = 2 / 3
    assert np.allclose(recall_score(y_true, y_predict), expected_recall)

    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_predict = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    expected_recall = 0.0
    assert np.allclose(recall_score(y_true, y_predict), expected_recall)


def test_recall_score2d():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_predict2d = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1]], dtype=np.int64)

    result = recall_score2d(y_true, y_predict2d)

    expected = np.array([0.0, 2 / 3], dtype=np.float64)
    assert np.allclose(result, expected)


def test_precision_score():
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_predict = np.array([0, 2, 1, 0, 1, 2], dtype=np.int64)
    expected_output = np.float64(0.6666666666666666)
    assert np.allclose(precision_score(y_true, y_predict), expected_output)

    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_predict = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    expected_output = np.float64(0.0)
    assert np.allclose(precision_score(y_true, y_predict), expected_output)


def test_precision_score2d():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_predict2d = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1]], dtype=np.int64)

    result = precision_score2d(y_true, y_predict2d)

    expected = np.array([0.0, 2 / 3], dtype=np.float64)
    assert np.allclose(result, expected)


def test_f1_score():
    y_true = np.array([0, 1, 0, 2], dtype=np.int64)
    y_predict = np.array([1, 2, 0, 2], dtype=np.int64)
    expected_output = np.float64(0.444444)
    assert np.allclose(f1_score(y_true, y_predict), expected_output)

    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_predict = np.array([0, 0, 1, 1], dtype=np.int64)
    expected_output = np.float64(1.0)
    assert np.allclose(f1_score(y_true, y_predict), expected_output)


def test_f1_score2d():
    y_true = np.array([0, 1, 0, 2], dtype=np.int64)
    y_predict2d = np.array([[1, 2, 0, 2], [0, 1, 0, 2]], dtype=np.int64)

    result = f1_score2d(y_true, y_predict2d)

    expected = np.array([0.444444, 1.0], dtype=np.float64)
    assert np.allclose(result, expected)
