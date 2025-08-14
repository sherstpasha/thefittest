from numba import float64
from numba import int64
from numba import njit

import numpy as np
from numpy.typing import NDArray

import torch


@njit(float64(float64[:], float64[:]))
def root_mean_square_error(
    y_true: NDArray[np.float64], y_predict: NDArray[np.float64]
) -> np.float64:
    error = y_true - y_predict
    mean_squared_error = np.mean(error**2)
    rmse = np.sqrt(mean_squared_error)

    return rmse


@njit(float64[:](float64[:], float64[:, :]))
def root_mean_square_error2d(
    y_true: NDArray[np.float64], y_predict2d: NDArray[np.float64]
) -> NDArray[np.float64]:
    size = len(y_predict2d)
    rmse_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        rmse_values[i] = root_mean_square_error(y_true, y_predict2d[i])

    return rmse_values


@njit(float64(float64[:], float64[:]))
def coefficient_determination(
    y_true: NDArray[np.float64], y_predict: NDArray[np.float64]
) -> np.float64:
    mean_y_true = np.mean(y_true)
    total_sum = np.sum((y_true - mean_y_true) ** 2)

    if total_sum == 0:
        total_sum = 1e-10

    error = y_true - y_predict
    residual_sum = np.sum((error) ** 2)
    r2 = 1 - residual_sum / total_sum

    return r2


@njit(float64[:](float64[:], float64[:, :]))
def coefficient_determination2d(
    y_true: NDArray[np.float64], y_predict2d: NDArray[np.float64]
) -> NDArray[np.float64]:
    size = len(y_predict2d)
    r2_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        r2_values[i] = coefficient_determination(y_true, y_predict2d[i])

    return r2_values


@njit(float64(float64[:, :], float64[:, :]))
def categorical_crossentropy(
    target: NDArray[np.float64], output: NDArray[np.float64]
) -> np.float64:
    target_clipped = np.clip(target, 1e-7, 1 - 1e-7)
    output_clipped = np.clip(output, 1e-7, 1 - 1e-7)

    log_prob = np.log(output_clipped)
    neg_log_prob = -target_clipped * log_prob
    cross_entropy = np.mean(np.sum(neg_log_prob, axis=1))

    return cross_entropy


@njit(float64[:](float64[:, :], float64[:, :, :]))
def categorical_crossentropy3d(
    target: NDArray[np.float64], output3d: NDArray[np.float64]
) -> NDArray[np.float64]:
    size = len(output3d)
    cross_entropy_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        cross_entropy_values[i] = categorical_crossentropy(target, output3d[i])

    return cross_entropy_values


@njit(float64(int64[:], int64[:]))
def accuracy_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    comparison = y_true == y_predict
    comparison_int = comparison.astype(np.int64)
    accuracy = np.mean(comparison_int)

    return accuracy


@njit(float64[:](int64[:], int64[:, :]))
def accuracy_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float64]:
    size = len(y_predict2d)
    accuracy_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        accuracy_values[i] = accuracy_score(y_true, y_predict2d[i])

    return accuracy_values


@njit(int64[:, :](int64[:], int64[:]))
def confusion_matrix(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> NDArray[np.int64]:
    classes = np.unique(y_true)
    n_classes = len(classes)

    confusion = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_predict):
        confusion[true, pred] += 1

    return confusion


@njit(float64(int64[:], int64[:]))
def recall_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    n_classes = len(np.unique(y_true))
    size = len(y_true)

    true_positives = np.zeros(shape=n_classes, dtype=np.int64)
    false_negatives = np.zeros(shape=n_classes, dtype=np.int64)
    class_recalls = np.zeros(shape=n_classes, dtype=np.float64)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            true_positives[y_true[i]] += 1
        else:
            false_negatives[y_true[i]] += 1

    for i in range(n_classes):
        if true_positives[i] != 0:
            class_recalls[i] = true_positives[i] / (false_negatives[i] + true_positives[i])

    average_recall = np.mean(class_recalls)
    return average_recall


@njit(float64[:](int64[:], int64[:, :]))
def recall_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float64]:
    size = len(y_predict2d)
    average_recall_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        average_recall_values[i] = recall_score(y_true, y_predict2d[i])

    return average_recall_values


@njit(float64(int64[:], int64[:]))
def precision_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    true_positives = np.zeros(shape=n_classes, dtype=np.int64)
    false_negatives = np.zeros(shape=n_classes, dtype=np.int64)
    class_precision = np.zeros(shape=n_classes, dtype=np.float64)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            true_positives[y_true[i]] += 1
        else:
            false_negatives[y_predict[i]] += 1

    for i in range(n_classes):
        if true_positives[i] != 0:
            class_precision[i] = true_positives[i] / (false_negatives[i] + true_positives[i])

    average_precision = np.mean(class_precision)
    return average_precision


@njit(float64[:](int64[:], int64[:, :]))
def precision_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float64]:
    size = len(y_predict2d)
    average_precision_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        average_precision_values[i] = precision_score(y_true, y_predict2d[i])

    return average_precision_values


@njit(float64(int64[:], int64[:]))
def f1_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    n_classes = len(np.unique(y_true))
    size = len(y_true)

    true_positives = np.zeros(shape=n_classes, dtype=np.int64)
    false_negatives = np.zeros(shape=n_classes, dtype=np.int64)
    down_precision = np.zeros(shape=n_classes, dtype=np.int64)
    f1 = np.zeros(shape=n_classes, dtype=np.float64)

    for i in range(size):
        if y_true[i] == y_predict[i]:
            true_positives[y_true[i]] += 1
        else:
            false_negatives[y_true[i]] += 1
            down_precision[y_predict[i]] += 1

    for i in range(n_classes):
        if true_positives[i] != 0:
            precision = true_positives[i] / (down_precision[i] + true_positives[i])
            recall = true_positives[i] / (false_negatives[i] + true_positives[i])
            f1[i] = 2 * (precision * recall) / (precision + recall)

    average_f1 = np.mean(f1)
    return average_f1


@njit(float64[:](int64[:], int64[:, :]))
def f1_score2d(y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]) -> NDArray[np.float64]:
    size = len(y_predict2d)
    average_f1_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        average_f1_values[i] = f1_score(y_true, y_predict2d[i])

    return average_f1_values


def _ce_from_probs_torch(
    probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:

    if probs.ndim == 3 and probs.size(-1) == 1:
        probs = probs.squeeze(-1)
    if targets.ndim == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    if probs.ndim == 3:
        probs = probs.reshape(-1, probs.size(-1))
        targets = targets.reshape(-1, targets.size(-1))

    probs = probs.clamp_min(eps)
    loss_vec = -(targets * probs.log()).sum(dim=-1)
    return loss_vec.mean()


def _rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.ndim == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)
    if pred.ndim == 2 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.ndim == 2 and target.size(-1) == 1:
        target = target.squeeze(-1)
    return torch.sqrt(torch.mean((pred - target) ** 2))
