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

    Parameters
    ----------
    y_true : NDArray[np.float64]
        1D array of the true values.
    y_predict : NDArray[np.float64]
        1D array of the predicted values.

    Returns
    -------
    numpy.float64
        The root mean square error between the true and predicted values.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import root_mean_square_error
    >>>
    >>> # Generate example data
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_predict = np.array([0.8, 2.5, 3.2, 4.3, 5.2])
    >>>
    >>> # Calculate root mean square error
    >>> rmse_result = root_mean_square_error(y_true, y_predict)
    >>>
    >>> print("Root Mean Square Error:", rmse_result)
    """

    error = y_true - y_predict
    mean_squared_error = np.mean(error**2)
    rmse = np.sqrt(mean_squared_error)

    return rmse


@njit(float64[:](float64[:], float64[:, :]))
def root_mean_square_error2d(
    y_true: NDArray[np.float64], y_predict2d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the root mean square error (RMSE) between the true values and predicted values for each row in a 2D array.

    Parameters
    ----------
    y_true : NDArray[np.float64]
        1D array of the true values.
    y_predict2d : NDArray[np.float64]
        2D array of the predicted values where each row is a separate prediction.

    Returns
    -------
    NDArray[np.float64]
        An array containing the RMSE for each row in the predicted values.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import root_mean_square_error2d
    >>>
    >>> # Generate example data
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_predict2d = np.array([[1.2, 2.3, 3.5, 4.2, 5.1],
    ...                         [0.9, 1.8, 3.2, 4.1, 5.3],
    ...                         [1.1, 2.0, 2.8, 4.3, 5.0]])
    >>>
    >>> # Calculate RMSE for each row
    >>> rmse_values = root_mean_square_error2d(y_true, y_predict2d)
    >>>
    >>> print("RMSE for each row:", rmse_values)
    """

    size = len(y_predict2d)
    rmse_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        rmse_values[i] = root_mean_square_error(y_true, y_predict2d[i])

    return rmse_values


@njit(float64(float64[:], float64[:]))
def coefficient_determination(
    y_true: NDArray[np.float64], y_predict: NDArray[np.float64]
) -> np.float64:
    """
    Calculates the coefficient of determination :math:`R^2` for a set of true and predicted values.

    Parameters
    ----------
    y_true : NDArray[np.float64]
        1D array of the true values.
    y_predict : NDArray[np.float64]
        1D array of the predicted values.

    Returns
    -------
    numpy.float64
        The calculated coefficient of determination :math:`R^2`.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import coefficient_determination
    >>>
    >>> # Generate example data
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_predict = np.array([1.2, 2.3, 3.5, 4.2, 5.1])
    >>>
    >>> # Calculate coefficient of determination
    >>> r2_value = coefficient_determination(y_true, y_predict)
    >>>
    >>> print("Coefficient of Determination (R^2):", r2_value)
    """

    mean_y_true = np.mean(y_true)
    total_sum = np.sum((y_true - mean_y_true) ** 2)

    error = y_true - y_predict
    residual_sum = np.sum((error) ** 2)
    r2 = 1 - residual_sum / total_sum

    return r2


@njit(float64[:](float64[:], float64[:, :]))
def coefficient_determination2d(
    y_true: NDArray[np.float64], y_predict2d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculates the coefficient of determination :math:`R^2` for a set of true values and a 2D array of predicted values.

    Parameters
    ----------
    y_true : NDArray[np.float64]
        1D array of the true values.
    y_predict2d : NDArray[np.float64]
        2D array of the predicted values where each row is a separate prediction.

    Returns
    -------
    NDArray[np.float64]
        An array containing the :math:`R^2` for each row in the predicted values.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import coefficient_determination2d
    >>>
    >>> # Generate example data
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_predict2d = np.array([[1.2, 2.3, 3.5, 4.2, 5.1],
    ...                         [0.9, 1.8, 3.2, 4.1, 5.3],
    ...                         [1.1, 2.0, 2.8, 4.3, 5.0]])
    >>>
    >>> # Calculate coefficient of determination for each row
    >>> r2_values = coefficient_determination2d(y_true, y_predict2d)
    >>>
    >>> print("Coefficient of Determination (R^2) for each row:", r2_values)
    """

    size = len(y_predict2d)
    r2_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        r2_values[i] = coefficient_determination(y_true, y_predict2d[i])

    return r2_values


@njit(float64(float64[:, :], float64[:, :]))
def categorical_crossentropy(
    target: NDArray[np.float64], output: NDArray[np.float64]
) -> np.float64:
    """
    Calculates cross-entropy for categorical classification.

    Parameters
    ----------
    target : NDArray[np.float64]
        2D array, representing the true probability for each class.
    predicted : NDArray[np.float64]
        2D array, representing the predicted probability values for each class.

    Returns
    -------
    np.float64
        Cross entropy value.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import categorical_crossentropy
    >>>
    >>> # Generate example data
    >>> target = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    >>> predicted = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.0, 0.1, 0.9]])
    >>>
    >>> # Calculate cross-entropy
    >>> entropy_value = categorical_crossentropy(target, predicted)
    >>>
    >>> print("Cross-Entropy:", entropy_value)
    """

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
    """
    Calculates cross-entropy for categorical classification for a 3D array of output values.

    Parameters
    ----------
    target : NDArray[np.float64]
        2D array, representing the true probability for each class.
    predicted : NDArray[np.float64]
        3D array, representing the predicted probability values for each class, where each row is a separate prediction.

    Returns
    -------
    np.float64
        An array containing the cross entropy values for each row in the predicted values.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import categorical_crossentropy3d
    >>>
    >>> # Generate example data
    >>> target = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    >>> predicted_3d = np.array([[[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.0, 0.1, 0.9]],
    >>>                           [[0.2, 0.7, 0.1], [0.9, 0.05, 0.05], [0.3, 0.3, 0.4]]])
    >>>
    >>> # Calculate cross-entropy for each row
    >>> entropy_values = categorical_crossentropy3d(target, predicted_3d)
    >>>
    >>> print("Cross-Entropy values for each row:", entropy_values)
    """

    size = len(output3d)
    cross_entropy_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        cross_entropy_values[i] = categorical_crossentropy(target, output3d[i])

    return cross_entropy_values


@njit(float64(int64[:], int64[:]))
def accuracy_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    This function calculates the accuracy of the predictions.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        1D array of the true labels.
    y_predict : NDArray[np.int64]
        1D array of the predicted labels.

    Returns
    -------
    np.float64
        The accuracy of the predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import accuracy_score
    >>>
    >>> # Generate example data
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_predict = np.array([1, 0, 1, 0, 1])
    >>>
    >>> # Calculate accuracy
    >>> accuracy_value = accuracy_score(y_true, y_predict)
    >>>
    >>> print("Accuracy:", accuracy_value)
    """

    comparison = y_true == y_predict
    comparison_int = comparison.astype(np.int64)
    accuracy = np.mean(comparison_int)

    return accuracy


@njit(float64[:](int64[:], int64[:, :]))
def accuracy_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Calculate the accuracy score for each prediction in a 2D array.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        1D array of the true labels.
    y_predict2d : NDArray[np.int64]
        2D array of the predicted labels where each row is a separate prediction.

    Returns
    -------
    NDArray[np.float64]
        An array containing the accuracy score for each row in the predicted labels.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import accuracy_score2d
    >>>
    >>> # Generate example data
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_predict2d = np.array([[1, 0, 1, 1, 1],
    ...                         [0, 0, 1, 1, 0],
    ...                         [1, 1, 0, 1, 0]])
    >>>
    >>> # Calculate accuracy score for each row
    >>> accuracy_values = accuracy_score2d(y_true, y_predict2d)
    >>>
    >>> print("Accuracy scores for each row:", accuracy_values)
    """

    size = len(y_predict2d)
    accuracy_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        accuracy_values[i] = accuracy_score(y_true, y_predict2d[i])

    return accuracy_values


@njit(int64[:, :](int64[:], int64[:]))
def confusion_matrix(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Calculate the confusion matrix for a multi-class classification problem.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        1D array of the true labels.
    y_predict : NDArray[np.int64]
        1D array of the predicted labels.

    Returns
    -------
    NDArray[np.int64]
        The confusion matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import confusion_matrix
    >>>
    >>> # Generate example data
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_predict = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    >>>
    >>> # Calculate confusion matrix
    >>> matrix = confusion_matrix(y_true, y_predict)
    >>>
    >>> print("Confusion Matrix:")
    >>> print(matrix)
    """

    classes = np.unique(y_true)
    n_classes = len(classes)

    confusion = np.zeros(shape=(n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_predict):
        confusion[true, pred] += 1

    return confusion


@njit(float64(int64[:], int64[:]))
def recall_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    Calculate the average recall score for multi-class classification problems.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        1D array of the true labels.
    y_predict : NDArray[np.int64]
        1D array of the predicted labels.

    Returns
    -------
    np.float64
        The average recall score of the predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import recall_score
    >>>
    >>> # Generate example data
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_predict = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    >>>
    >>> # Calculate average recall score
    >>> recall = recall_score(y_true, y_predict)
    >>>
    >>> print("Average Recall Score:", recall)
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

    Parameters
    ----------
    y_true : NDArray[np.int64]
        1D array of the true labels.
    y_predict2d : NDArray[np.int64]
        2D array of predicted labels where each row is a separate prediction.

    Returns
    -------
    NDArray[np.float64]
        The recall scores for each prediction.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import recall_score2d
    >>>
    >>> # Generate example data
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_predict2d = np.array([[0, 1, 2, 0, 2],
    ...                         [2, 1, 0, 1, 2],
    ...                         [0, 1, 2, 2, 1]])
    >>>
    >>> # Compute recall scores for each prediction
    >>> recall_values = recall_score2d(y_true, y_predict2d)
    >>>
    >>> print("Recall Scores for each Prediction:", recall_values)
    """

    size = len(y_predict2d)
    average_recall_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        average_recall_values[i] = recall_score(y_true, y_predict2d[i])

    return average_recall_values


@njit(float64(int64[:], int64[:]))
def precision_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    Compute the average precision score.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        1D array of the true labels.
    y_predict : NDArray[np.int64]
        1D array of the predicted labels.

    Returns
    -------
    np.float64
        The average precision score.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import precision_score
    >>>
    >>> # Generate example data
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_predict = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    >>>
    >>> # Compute average precision score
    >>> precision_value = precision_score(y_true, y_predict)
    >>>
    >>> print("Average Precision Score:", precision_value)
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

    average_precision = np.mean(class_precision)
    return average_precision


@njit(float64[:](int64[:], int64[:, :]))
def precision_score2d(
    y_true: NDArray[np.int64], y_predict2d: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Compute the precision score for each prediction in a 2D array of predictions.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        1D array of the true labels.
    y_predict2d : NDArray[np.int64]
        The 2D array of predicted labels where each row is a separate prediction.

    Returns
    -------
    NDArray[np.float64]
        The precision scores for each prediction.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import precision_score2d
    >>>
    >>> # Generate example data
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_predict2d = np.array([[0, 1, 2, 0, 2, 2, 0, 1, 1],
    ...                          [1, 0, 2, 0, 1, 2, 2, 1, 0]])
    >>>
    >>> # Compute precision scores for each prediction
    >>> precision_values = precision_score2d(y_true, y_predict2d)
    >>>
    >>> print("Precision Scores for Each Prediction:", precision_values)
    """

    size = len(y_predict2d)
    average_precision_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        average_precision_values[i] = precision_score(y_true, y_predict2d[i])

    return average_precision_values


@njit(float64(int64[:], int64[:]))
def f1_score(y_true: NDArray[np.int64], y_predict: NDArray[np.int64]) -> np.float64:
    """
    Function to calculate the F1-score for multi-class classification. F1-score is a measure that combines precision and recall into one value.
    It is calculated as 2 * (precision * recall) / (precision + recall), where:
    - precision is the ratio of correctly predicted positive results out of all predicted
    positive results.
    - recall is the ratio of correctly predicted positive results out of all actual positive
    results.
    F1-score is calculated for each class separately and then averaged.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        Array of true class labels.
    y_predict : NDArray[np.int64]
        Array of predicted class labels.

    Returns
    -------
    np.float64
        Average value of the F1-score.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import f1_score
    >>>
    >>> # Generate example data
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_predict = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    >>>
    >>> # Calculate F1-score
    >>> f1_value = f1_score(y_true, y_predict)
    >>>
    >>> print("F1-score:", f1_value)
    """

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
    """
    Compute the F1 score for each prediction in a 2D array.

    Parameters
    ----------
    y_true : NDArray[np.int64]
        The true labels.
    y_predict2d : NDArray[np.int64]
        A 2D array of predicted labels where each row is a separate prediction.

    Returns
    -------
    NDArray[np.float64]
        An array of F1 scores, one for each prediction.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.metrics import f1_score2d
    >>>
    >>> # Generate example data
    >>> y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> y_predict2d = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 1, 2, 1, 0], [0, 1, 2, 0, 1, 2, 1, 2, 0]])
    >>>
    >>> # Calculate F1 score for each prediction
    >>> f1_values = f1_score2d(y_true, y_predict2d)
    >>>
    >>> print("F1 scores for each prediction:", f1_values)
    """

    size = len(y_predict2d)
    average_f1_values = np.empty(size, dtype=np.float64)

    for i in range(size):
        average_f1_values[i] = f1_score(y_true, y_predict2d[i])

    return average_f1_values
