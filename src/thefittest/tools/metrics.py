import numpy as np


def root_mean_square_error(y_true: np.ndarray,
                           y_predict: np.ndarray) -> float:
    error = y_true - y_predict
    return np.sqrt(np.mean((error)**2))


def coefficient_determination(y_true: np.ndarray,
                              y_predict: np.ndarray) -> float:
    error = y_true - y_predict
    mean_y_true = np.mean(y_true)
    residual_sum = np.sum((error)**2)
    total_sum = np.sum((y_true - mean_y_true)**2)
    return 1 - residual_sum/total_sum


def categorical_crossentropy(target: np.ndarray,
                             output: np.ndarray):
    output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.mean(np.sum(target * -np.log(output),
                          axis=-1, keepdims=False))