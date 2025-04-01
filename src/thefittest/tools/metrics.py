from numba import float32
from numba import int64
from numba import njit

import numpy as np
from numpy.typing import NDArray


@njit(float32(float32[:], float32[:]))
def root_mean_square_error(
    y_true: NDArray[np.float32], y_predict: NDArray[np.float32]
) -> np.float32:
    error = y_true - y_predict
    return np.sqrt(np.mean((error) ** 2))


@njit(float32[:](float32[:], float32[:, :]))
def root_mean_square_error2d(
    y_true: NDArray[np.float32], y_predict2d: NDArray[np.float32]
) -> NDArray[np.float32]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float32)

    for i in range(size):
        error = y_true - y_predict2d[i]
        to_return[i] = np.sqrt(np.mean((error) ** 2))
    return to_return


@njit(float32[:](float32[:, :], float32[:, :, :]))
def root_mean_square_error3d(y_true: np.ndarray, y_predict2d: np.ndarray) -> np.ndarray:
    size = len(y_predict2d)
    num_variables = y_true.shape[1]
    to_return = np.empty(size, dtype=np.float32)

    for i in range(size):
        error_sum = 0.0
        for j in range(num_variables):
            error = y_true[:, j] - y_predict2d[i, :, j]
            error_sum += np.mean(error**2)
        to_return[i] = np.sqrt(error_sum / num_variables)

    return to_return


@njit(float32(float32[:], float32[:]))
def coefficient_determination(
    y_true: NDArray[np.float32], y_predict: NDArray[np.float32]
) -> np.float32:
    error = y_true - y_predict
    mean_y_true = np.mean(y_true)
    residual_sum = np.sum((error) ** 2)
    total_sum = np.sum((y_true - mean_y_true) ** 2)
    return 1 - residual_sum / total_sum


@njit(float32[:](float32[:], float32[:, :]))
def coefficient_determination2d(
    y_true: NDArray[np.float32], y_predict2d: NDArray[np.float32]
) -> NDArray[np.float32]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float32)

    mean_y_true = np.mean(y_true)
    total_sum = np.sum((y_true - mean_y_true) ** 2)
    for i in range(size):
        error = y_true - y_predict2d[i]
        residual_sum = np.sum((error) ** 2)
        to_return[i] = 1 - residual_sum / total_sum
    return to_return


@njit(float32(float32[:, :], float32[:, :]))
def categorical_crossentropy(
    target: NDArray[np.float32], output: NDArray[np.float32]
) -> np.float32:
    output_c = np.clip(output, 1e-7, 1 - 1e-7)
    to_return = np.mean(np.sum(target * (-np.log(output_c)), axis=1))
    return to_return


# @njit(float32[:](float32[:, :], float32[:, :, :]))
# def categorical_crossentropy3d(
#     target: NDArray[np.float32], output3d: NDArray[np.float32]
# ) -> NDArray[np.float32]:
#     size = len(output3d)
#     to_return = np.empty(size, dtype=np.float32)
#     for i in range(size):
#         to_return[i] = categorical_crossentropy(target, output3d[i])
#     return to_return
import torch


def categorical_crossentropy3d(target, output3d, epsilon=1e-12):
    """
    Вычисляет категориальную кросс-энтропию для батча, где:
      - target имеет размер (n, m)
      - output3d имеет размер (p, n, m) для p примеров.

    Для каждого примера i считается:
        loss[i] = mean( sum_{k} ( target[j,k] * (-log(output3d[i,j,k] + epsilon)) ) по k,
                        усреднённое по j )

    Это соответствует реализации на numba, где сначала берётся сумма по столбцам (axis=1),
    а затем среднее по строкам.

    Возвращается массив (тензор) размера (p,).
    """
    # Обеспечиваем численную устойчивость
    output3d = torch.clamp(output3d, 1e-7, 1 - 1e-7)
    # Вычисляем потери для каждого элемента:
    # 1. Вычисляем -log(output3d + epsilon)
    # 2. Умножаем на target (broadcast до (p, n, m))
    # 3. Суммируем по последней оси (axis=2) → получаем тензор размера (p, n)
    # 4. Берём среднее по оси 1 → получаем тензор размера (p,)
    loss = (target * (-torch.log(output3d + epsilon))).sum(dim=2).mean(dim=1)
    return loss.detach().cpu().numpy().astype(np.float32)


@njit(float32(int64[:], int64[:]))
def accuracy_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float32:
    return np.mean((y_true == y_predict).astype(np.int64))


@njit(float32[:](int64[:], int64[:, :]))
def accuracy_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float32]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float32)
    for i in range(size):
        to_return[i] = accuracy_score(y_true, y_predict2d[i])
    return to_return


@njit(int64[:, :](int64[:], int64[:]))
def confusion_matrix(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    to_return = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)
    for i in range(size):
        to_return[y_true[i]][y_predict[i]] += 1
    return to_return


@njit(float32(int64[:], int64[:]))
def recall_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float32:
    """
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    """
    n_classes = len(np.unique(y_true))
    size = len(y_true)

    up = np.zeros(shape=n_classes, dtype=np.int64)
    down = np.zeros(shape=n_classes, dtype=np.int64)
    recall = np.zeros(shape=n_classes, dtype=np.float32)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            up[y_true[i]] += 1
        else:
            down[y_true[i]] += 1

    for i in range(n_classes):
        if up[i] != 0:
            recall[i] = up[i] / (down[i] + up[i])

    return np.mean(recall)


@njit(float32[:](int64[:], int64[:, :]))
def recall_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float32]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float32)
    for i in range(size):
        to_return[i] = recall_score(y_true, y_predict2d[i])
    return to_return


@njit(float32(int64[:], int64[:]))
def precision_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float32:
    """
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    up = np.zeros(shape=n_classes, dtype=np.int64)
    down = np.zeros(shape=n_classes, dtype=np.int64)
    precision = np.zeros(shape=n_classes, dtype=np.float32)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            up[y_true[i]] += 1
        else:
            down[y_predict[i]] += 1

    for i in range(n_classes):
        if up[i] != 0:
            precision[i] = up[i] / (down[i] + up[i])

    return np.mean(precision)


@njit(float32[:](int64[:], int64[:, :]))
def precision_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float32]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float32)
    for i in range(size):
        to_return[i] = precision_score(y_true, y_predict2d[i])
    return to_return


@njit(float32(int64[:], int64[:]))
def f1_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float32:
    """
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    """
    n_classes = len(np.unique(y_true))
    size = len(y_true)

    up = np.zeros(shape=n_classes, dtype=np.int64)
    down_recall = np.zeros(shape=n_classes, dtype=np.int64)
    down_precision = np.zeros(shape=n_classes, dtype=np.int64)
    f1 = np.zeros(shape=n_classes, dtype=np.float32)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            up[y_true[i]] += 1
        else:
            down_recall[y_true[i]] += 1
            down_precision[y_predict[i]] += 1

    for i in range(n_classes):
        if up[i] != 0:
            precision = up[i] / (down_precision[i] + up[i])
            recall = up[i] / (down_recall[i] + up[i])
            f1[i] = 2 * (precision * recall) / (precision + recall)

    return np.mean(f1)


@njit(float32[:](int64[:], int64[:, :]))
def f1_score2d(y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]) -> NDArray[np.float32]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float32)
    for i in range(size):
        to_return[i] = f1_score(y_true, y_predict2d[i])
    return to_return
