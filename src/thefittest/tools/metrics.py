import numpy as np
from numpy.typing import NDArray
from numba import njit
from numba import float64
from numba import int64


@njit(float64(float64[:], float64[:]))
def root_mean_square_error(y_true: NDArray[np.float64],
                           y_predict: NDArray[np.float64]) -> np.float64:
    error = y_true - y_predict
    return np.sqrt(np.mean((error)**2))


@njit(float64[:](float64[:], float64[:, :]))
def root_mean_square_error2d(y_true: NDArray[np.float64],
                             y_predict2d: NDArray[np.float64]) -> NDArray[np.float64]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)

    for i in range(size):
        error = y_true - y_predict2d[i]
        to_return[i] = np.sqrt(np.mean((error)**2))
    return to_return


@njit(float64(float64[:], float64[:]))
def coefficient_determination(y_true: NDArray[np.float64],
                              y_predict: NDArray[np.float64]) -> np.float64:
    error = y_true - y_predict
    mean_y_true = np.mean(y_true)
    residual_sum = np.sum((error)**2)
    total_sum = np.sum((y_true - mean_y_true)**2)
    return 1 - residual_sum/total_sum


@njit(float64[:](float64[:], float64[:, :]))
def coefficient_determination2d(y_true: NDArray[np.float64],
                                y_predict2d: NDArray[np.float64]) -> NDArray[np.float64]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)

    mean_y_true = np.mean(y_true)
    total_sum = np.sum((y_true - mean_y_true)**2)
    for i in range(size):
        error = y_true - y_predict2d[i]
        residual_sum = np.sum((error)**2)
        to_return[i] = 1 - residual_sum/total_sum
    return to_return


@njit(float64(float64[:, :], float64[:, :]))
def categorical_crossentropy(target: NDArray[np.float64],
                             output: NDArray[np.float64]) -> np.float64:
    output_c = np.clip(output, 1e-7, 1 - 1e-7)
    to_return = np.mean(np.sum(target*(-np.log(output_c)), axis=1))
    return to_return


@njit(float64[:](float64[:, :], float64[:, :, :]))
def categorical_crossentropy3d(
        target: NDArray[np.float64],
        output3d: NDArray[np.float64]) -> NDArray[np.float64]:
    size = len(output3d)
    to_return = np.empty(size, dtype=np.float64)
    for i in range(size):
        to_return[i] = categorical_crossentropy(target, output3d[i])
    return to_return


@njit(float64(int64[:], int64[:]))
def accuracy_score(y_true: NDArray[np.int64],
                   y_predict: NDArray[np.int64]) -> np.float64:
    return np.mean((y_true == y_predict).astype(np.int64))


@njit(float64[:](int64[:], int64[:, :]))
def accuracy_score2d(y_true: NDArray[np.int64],
                     y_predict2d: NDArray[np.int64]) -> np.float64:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)
    for i in range(size):
        to_return[i] = accuracy_score(y_true, y_predict2d[i])
    return to_return


@njit(int64[:, :](int64[:], int64[:]))
def confusion_matrix(y_true: NDArray[np.int64],
                     y_predict: NDArray[np.int64]) -> NDArray[np.int64]:
    '''
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    '''
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    to_return = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)
    for i in range(size):
        to_return[y_true[i]][y_predict[i]] += 1
    return to_return


@njit(float64(int64[:], int64[:]))
def recall_score(y_true: NDArray[np.int64],
                 y_predict: NDArray[np.int64]) -> np.float64:
    '''
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    '''
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    up = np.zeros(shape=n_classes, dtype=np.int64)
    down = np.zeros(shape=n_classes, dtype=np.int64)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            up[y_true[i]] += 1
        else:
            down[y_true[i]] += 1

    recall = up/(down + up)
    recall[np.isnan(recall)] = 1

    return np.mean(recall)


@njit(float64[:](int64[:], int64[:, :]))
def recall_score2d(y_true: NDArray[np.int64],
                   y_predict2d: NDArray[np.int64]) -> NDArray[np.float64]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)
    for i in range(size):
        to_return[i] = recall_score(y_true, y_predict2d[i])
    return to_return


@njit(float64(int64[:], int64[:]))
def precision_score(y_true: NDArray[np.int64],
                    y_predict: NDArray[np.int64]) -> np.float64:
    '''
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    '''
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    up = np.zeros(shape=n_classes, dtype=np.int64)
    down = np.zeros(shape=n_classes, dtype=np.int64)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            up[y_true[i]] += 1
        else:
            down[y_predict[i]] += 1

    precision = up / (down + up)
    precision[np.isnan(precision)] = 1

    return np.mean(precision)


@njit(float64[:](int64[:], int64[:, :]))
def precision_score2d(y_true: NDArray[np.int64],
                      y_predict2d: NDArray[np.int64]) -> NDArray[np.float64]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)
    for i in range(size):
        to_return[i] = precision_score(y_true, y_predict2d[i])
    return to_return


@njit(float64(int64[:], int64[:]))
def f1_score(y_true: NDArray[np.int64],
             y_predict: NDArray[np.int64]) -> np.float64:
    '''
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    '''
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    up = np.zeros(shape=n_classes, dtype=np.int64)
    down_recall = np.zeros(shape=n_classes, dtype=np.int64)
    down_precision = np.zeros(shape=n_classes, dtype=np.int64)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            up[y_true[i]] += 1
        else:
            down_recall[y_true[i]] += 1
            down_precision[y_predict[i]] += 1

    precision = up/(down_precision + up)
    recall = up/(down_recall + up)

    precision[np.isnan(precision)] = 1
    recall[np.isnan(recall)] = 1

    f1 = 2 * (precision * recall) / (precision + recall)
    return np.mean(f1)


@njit(float64[:](int64[:], int64[:, :]))
def f1_score2d(y_true: NDArray[np.int64],
               y_predict2d: NDArray[np.int64]) -> NDArray[np.float64]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)
    for i in range(size):
        to_return[i] = f1_score(y_true, y_predict2d[i])
    return to_return
