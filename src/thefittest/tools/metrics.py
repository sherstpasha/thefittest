from typing import Optional

from numba import float64
from numba import int64
from numba import njit

import numpy as np
from numpy.typing import NDArray


@njit(float64(float64[:], float64[:]))
def root_mean_square_error(
    y_true: NDArray[np.float64], y_predict: NDArray[np.float64]
) -> np.float64:
    """
    Calculate the root mean square error between the true and predicted values.

    Parameters:
        y_true (NDArray[np.float64]): The true values.
        y_predict (NDArray[np.float64]): The predicted values.

    Returns:
        np.float64: The root mean square error between the true and predicted values.
    """
    error = y_true - y_predict
    mean_squared_error = np.mean(error ** 2)
    rmse = np.sqrt(mean_squared_error)

    return rmse


@njit(float64[:](float64[:], float64[:, :]))
def root_mean_square_error2d(
    y_true: NDArray[np.float64], y_predict2d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the root mean square error (RMSE) between the true values and predicted values for each row in a 2D array.

    Parameters:
        y_true (NDArray[np.float64]): The true values.
        y_predict2d (NDArray[np.float64]): The predicted values in a 2D array.

    Returns:
        NDArray[np.float64]: An array containing the RMSE for each row in the predicted values.
    """

    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)

    for i in range(size):
        to_return[i] = root_mean_square_error(y_true, y_predict2d[i])

    return to_return


@njit(float64(float64[:], float64[:]))
def coefficient_determination(
    y_true: NDArray[np.float64],
    y_predict: NDArray[np.float64],
    total_sum: Optional[np.float64]
) -> np.float64:
    """
    The `coefficient_determination` function calculates the coefficient of determination (R^2) for a set of true and predicted values.

    Parameters:
        y_true (NDArray[np.float64]): An array of true values.
        y_predict (NDArray[np.float64]): An array of predicted values.
        total_sum (Optional[np.float64]): An optional parameter that represents the total sum of squares. If not provided, the function will calculate it based on the mean of the true values.

    Returns:
        np.float64: The calculated coefficient of determination (R^2).
    """
    if total_sum is None:
        # Calculate the mean of the true values
        mean_y_true = np.mean(y_true)

        # Calculate the total sum of squares
        total_sum = np.sum((y_true - mean_y_true) ** 2)

    # Calculate the error between the true values and predicted values
    error = y_true - y_predict

    # Calculate the residual sum of squares
    residual_sum = np.sum((error) ** 2)

    # Calculate the coefficient of determination (R^2)
    r2 = 1 - residual_sum / total_sum

    return r2


@njit(float64[:](float64[:], float64[:, :]))
def coefficient_determination2d(
    y_true: NDArray[np.float64], y_predict2d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    The `coefficient_determination2d` function calculates the coefficient of determination (R^2) for a set of true values and a 2D array of predicted values.

    Parameters:
        y_true (NDArray[np.float64]): An array of true values.
        y_predict2d (NDArray[np.float64]): A 2D array of predicted values, where each row represents a set of predicted values.

    Returns:
        NDArray[np.float64]: An array of calculated coefficients of determination (R^2), where each element corresponds to the R^2 value for the corresponding row of `y_predict2d`.
    """
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)

    mean_y_true = np.mean(y_true)
    total_sum = np.sum((y_true - mean_y_true) ** 2)

    for i in range(size):
        to_return[i] = coefficient_determination(y_true, y_predict2d[i], total_sum)

    return to_return


def categorical_crossentropy(
    target: NDArray[np.float64], output: NDArray[np.float64]
) -> np.float64:
    """
    Calculates cross-entropy for categorical classification.

    Parameters:
        target (NDArray[np.float64]): Array of dimension 2d, representing the true probability for each class 
        output (NDArray[np.float64]): An array of dimension 2d representing the predicted probability values for each class

    Returns:
        np.float64: the value of cross-entropy
    """
    # Clip values to the range (0, 1) to avoid problems with zero values in logarithms
    target_clipped = np.clip(target, 1e-7, 1 - 1e-7)
    output_clipped = np.clip(output, 1e-7, 1 - 1e-7)

    # Calculate log probability
    log_prob = np.log(output_clipped)

    # Calculate negative log probability
    neg_log_prob = - target_clipped * log_prob

    # Calculate mean negative log probability
    cross_entropy = np.mean(np.sum(neg_log_prob, axis=1))

    return cross_entropy


@njit(float64[:](float64[:, :], float64[:, :, :]))
def categorical_crossentropy3d(
    target: NDArray[np.float64], output3d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculates cross-entropy for categorical classification for a 3D array of output values.

    Parameters:
        target (NDArray[np.float64]): Array of dimension 2d, representing the true probability for each class 
        output3d (NDArray[np.float64]): A 3D array of dimension 3d representing the predicted probability values for each class

    Returns:
        NDArray[np.float64]: an array of cross-entropy values, one for each 2D slice in the 3D output array
    """
    size = len(output3d)
    to_return = np.empty(size, dtype=np.float64)

    for i in range(size):
        to_return[i] = categorical_crossentropy(target, output3d[i])

    return to_return


@njit(float64(int64[:], int64[:]))
def accuracy_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    This function calculates the accuracy of the predictions.

    Parameters:
        y_true (NDArray[np.int64]): The true values.
        y_predict (NDArray[np.int64]): The predicted values.

    Returns:
        np.float64: The accuracy of the predictions.
    """
    # Comparing the elements of the arrays
    comparison = (y_true == y_predict)

    # Converting the comparison result to an array of integers
    comparison_int = comparison.astype(np.int64)

    # Calculating the mean value of the array elements
    accuracy = np.mean(comparison_int)

    return accuracy


@njit(float64[:](int64[:], int64[:, :]))
def accuracy_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Calculate the accuracy score for each prediction in a 2D array.

    Parameters:
        y_true (NDArray[np.int64]): The true labels.
        y_predict2d (NDArray[np.int64]): The predicted labels in a 2D array.

    Returns:
        NDArray[np.float64]: An array containing the accuracy score for each prediction.
    """

    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)

    for i in range(size):
        to_return[i] = accuracy_score(y_true, y_predict2d[i])

    return to_return


@njit(int64[:, :](int64[:], int64[:]))
def confusion_matrix(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Calculate the confusion matrix for a multi-class classification problem.

    Parameters:
        y_true (NDArray[np.int64]): The true labels.
        y_predict (NDArray[np.int64]): The predicted labels.

    Returns:
        NDArray[np.int64]: The confusion matrix.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    size = len(y_true)

    confusion = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_predict):
        confusion[true, pred] += 1

    return confusion


@njit(float64(int64[:], int64[:]))
def recall_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    The recall_score function calculates the average recall score of the predictions for multi-class classification problems.

    Parameters:
        y_true (NDArray[np.int64]): The true labels.
        y_predict (NDArray[np.int64]): The predicted labels.

    Returns:
        np.int64: The average recall score of the predictions.
    """
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
    """
    Compute the recall score for each prediction in a 2D array of predictions.

    Parameters:
        y_true (NDArray[np.int64]): The true labels.
        y_predict2d (NDArray[np.int64]): The 2D array of predicted labels.

    Returns:
        NDArray[np.float64]: The recall scores for each prediction.
    """
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)

    for i in range(size):
        to_return[i] = recall_score(y_true, y_predict2d[i])

    return to_return


@njit(float64(int64[:], int64[:]))
def precision_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    Compute the precision score for each class in the true labels.

    Parameters:
        y_true (NDArray[np.int64]): The true labels.
        y_predict (NDArray[np.int64]): The predicted labels.

    Returns:
        np.float64: The average precision score.
    """
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

    average_precision = np.mean(precision)
    return average_precision


@njit(float64[:](int64[:], int64[:, :]))
def precision_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Compute the precision score for each prediction in a 2D array of predictions.

    Parameters:
        y_true (NDArray[np.int64]): The true labels.
        y_predict2d (NDArray[np.int64]): The 2D array of predicted labels.

    Returns:
        NDArray[np.float64]: The precision scores for each prediction.
    """
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)

    for i in range(size):
        to_return[i] = precision_score(y_true, y_predict2d[i])
        
    return to_return


@njit(float64(int64[:], int64[:]))
def f1_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    y_true = np.array([0, 1, 0, 2 ...], dtype = np.int64)
    y_predict = np.array([1, 2, 0, 2 ...], dtype = np.int64)
    """
    n_classes = len(np.unique(y_true))
    size = len(y_true)

    up = np.zeros(shape=n_classes, dtype=np.int64)
    down_recall = np.zeros(shape=n_classes, dtype=np.int64)
    down_precision = np.zeros(shape=n_classes, dtype=np.int64)
    f1 = np.zeros(shape=n_classes, dtype=np.float64)

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


@njit(float64[:](int64[:], int64[:, :]))
def f1_score2d(y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]) -> NDArray[np.float64]:
    size = len(y_predict2d)
    to_return = np.empty(size, dtype=np.float64)
    for i in range(size):
        to_return[i] = f1_score(y_true, y_predict2d[i])
    return to_return
