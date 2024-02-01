import numpy as np

from sklearn.metrics import mean_squared_error as mean_squared_error_sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score as accuracy_score_sklearn
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import recall_score as sklearn_recall_score
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import f1_score as sklearn_f1_score

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
    # Test data
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_predict = np.array([1.1, 1.9, 3.1], dtype=np.float64)

    # Calculate RMSE using both functions
    rmse_njit = root_mean_square_error(y_true, y_predict)
    rmse_sklearn = np.sqrt(mean_squared_error_sklearn(y_true, y_predict))

    # Check if the results are equal within a small tolerance
    assert np.isclose(rmse_njit, rmse_sklearn, rtol=1e-10, atol=1e-10)

def test_coefficient_determination():
    # Test data
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_predict = np.array([1.1, 1.9, 3.1], dtype=np.float64)

    # Calculate R-squared using both functions
    r2_njit = coefficient_determination(y_true, y_predict)
    r2_sklearn = r2_score(y_true, y_predict)

    # Check if the results are equal within a small tolerance
    assert np.isclose(r2_njit, r2_sklearn, rtol=1e-10, atol=1e-10)


def test_categorical_crossentropy():
    # Test data
    target = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    output = np.array([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0], [0.0, 0.1, 0.9]], dtype=np.float64)

    # Calculate categorical crossentropy using both functions
    crossentropy_njit = categorical_crossentropy(target, output)
    crossentropy_sklearn = log_loss(target, output)

    # Check if the results are equal within a small tolerance
    assert np.isclose(crossentropy_njit, crossentropy_sklearn, rtol=1e-4, atol=1e-4)


def test_accuracy_score():
    # Test data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64)
    y_predict = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)

    # Calculate accuracy score using both functions
    accuracy_score_njit_result = accuracy_score(y_true, y_predict)
    accuracy_score_sklearn_result = accuracy_score_sklearn(y_true, y_predict)

    # Check if the results are equal within a small tolerance
    assert np.isclose(accuracy_score_njit_result, accuracy_score_sklearn_result, rtol=1e-10, atol=1e-10)


def test_confusion_matrix():
    # Test data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64)
    y_predict = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)

    # Calculate confusion matrix using both functions
    confusion_matrix_njit_result = confusion_matrix(y_true, y_predict)
    confusion_matrix_sklearn_result = sklearn_confusion_matrix(y_true, y_predict)

    # Check if the results are equal
    assert np.all(confusion_matrix_njit_result == confusion_matrix_sklearn_result)


def test_recall_score():
    # Test data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64)
    y_predict = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)

    # Calculate recall scores using both functions
    recall_score_njit_result = recall_score(y_true, y_predict)
    recall_score_sklearn_result = sklearn_recall_score(y_true, y_predict, average='macro')

    # Check if the results are almost equal (considering potential floating-point differences)
    assert np.isclose(recall_score_njit_result, recall_score_sklearn_result, rtol=1e-10, atol=1e-10)


def test_precision_score():
    # Test data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64)
    y_predict = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)

    # Calculate precision scores using both functions
    precision_score_njit_result = precision_score(y_true, y_predict)
    precision_score_sklearn_result = sklearn_precision_score(y_true, y_predict, average='macro')

    # Check if the results are almost equal (considering potential floating-point differences)
    assert np.isclose(precision_score_njit_result, precision_score_sklearn_result, rtol=1e-10, atol=1e-10)

def test_f1_score():
    # Test data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64)
    y_predict = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)

    # Calculate F1 scores using both functions
    f1_score_njit_result = f1_score(y_true, y_predict)
    f1_score_sklearn_result = sklearn_f1_score(y_true, y_predict, average='macro')

    # Check if the results are almost equal (considering potential floating-point differences)
    assert np.isclose(f1_score_njit_result, f1_score_sklearn_result, rtol=1e-10, atol=1e-10)
