import numpy as np
from numpy.typing import NDArray


def z(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Helper function used in F13 and F14.

    Combination of three inverse squared terms with different parameters.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array

    Returns
    -------
    NDArray[np.float64]
        Function values
    """
    first = -1 / ((x - 1) ** 2 + 0.2)
    left = -1 / (2 * (x - 2) ** 2 + 0.15)
    right = -1 / (3 * (x - 3) ** 2 + 0.3)
    return first + left + right


def F1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Complex oscillatory function with exponential decay (1D).

    Domain: [-1, 1]
    Variables: 1

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 1)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    firts = 0.05 * (x_1 - 1) * (x_1 - 1)
    exp_x_power_2 = np.exp(-2.77257 * x_1 * x_1)
    left = 3 - 2.9 * exp_x_power_2
    right = 1 - np.cos(x_1 * (4 - 50 * exp_x_power_2))
    return firts + left * right


def F2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multi-frequency cosine composition (1D).

    Domain: [-1, 1]
    Variables: 1

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 1)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    left = 0.5 * np.cos(1.5 * (10 * x_1 - 0.3)) * np.cos(31.4 * x_1)
    right = 0.5 * np.cos(np.sqrt(5) * 10 * x_1) * np.cos(35 * x_1)
    return 1 - left + right


def F3(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quadratic with cosine terms (2D).

    Domain: [-16, 16]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    left = 0.1 * x_1**2 + 0.1 * x_2**2
    right = -4 * np.cos(0.8 * x_1) - 4 * np.cos(0.8 * x_2) + 8
    return left + right


def F4(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scaled version of F3 (2D).

    Domain: [-16, 16]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    left = (0.1 * 1.5 * x_2) ** 2 + (0.1 * 0.8 * x_1) ** 2
    right = -4 * np.cos(0.8 * 1.5 * x_2) - 4 * np.cos(0.8 * 0.8 * x_1) + 8
    return left + right


def F5(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rosenbrock function (2D).

    Classic optimization benchmark with a narrow parabolic valley.

    Domain: [-2, 2]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    return 100 * ((x_2 - x_1**2)) ** 2 + (1 - x_1) ** 2


def F6(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Modified Griewank-like function (2D).

    Domain: [-16, 16]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    left = 0.005 * (x_1**2 + x_2**2)
    right = -np.cos(x_1) * np.cos(x_2 / np.sqrt(2)) + 2
    return -10 / (left + right) + 10


def F7(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse Rosenbrock (2D).

    Domain: [-5, 5]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    down = 100 * (x_1**2 - x_2) + (1 - x_1) ** 2 + 1
    return -100 / down + 100


def F8(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Modified Schaffer function (2D).

    Domain: [-10, 10]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    power_x_y = x_1**2 + x_2**2
    up = 1 - np.sin(np.sqrt(power_x_y)) ** 2
    down = 1 + 0.001 * (power_x_y)
    return up / down


def F9(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quadratic with multi-frequency cosines (2D).

    Domain: [-2.5, 2.5]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    first = 0.5 * (x_1**2 + x_2**2)
    left = 2 * 0.8 + 0.8 * np.cos(1.5 * x_1) * np.cos(3.14 * x_2)
    right = 0.8 * np.cos(np.sqrt(5) * x_1) * np.cos(3.5 * x_2)
    return first * (left + right)


def F10(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Similar to F9 with different domain (2D).

    Domain: [-5, 5]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    first = 0.5 * (x_1**2 + x_2**2)
    left = 2 * 0.8 + 0.8 * np.cos(1.5 * x_1) * np.cos(3.14 * x_2)
    right = 0.8 * np.cos(np.sqrt(5) * x_1) * np.cos(3.5 * x_2)
    return first * (left + right)


def F11(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Absolute sine products with inverse term (2D).

    Domain: [-4, 4]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    left = (x_1**2) * np.abs(np.sin(2 * x_1))
    right = (x_2**2) * np.abs(np.sin(2 * x_2))
    last = -1 / (5 * x_1**2 + 5 * x_2**2 + 0.2) + 5
    return left + right + last


def F12(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cross-term interaction function (2D).

    Domain: [0, 4]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    first = 0.5 * (x_1**2 + x_1 * x_2 + x_2**2)
    left = 1 + 0.5 * np.cos(1.5 * x_1) * np.cos(3.2 * x_1 * x_2) * np.cos(3.14 * x_2)
    right = 0.5 * np.cos(2.2 * x_2) * np.cos(4.8 * x_1 * x_2) * np.cos(3.5 * x_2)
    return first * (left + right)


def F13(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Product of z-functions (2D).

    Domain: [0, 4]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    z_1 = z(x_1)
    z_2 = z(x_2)
    return -z_1 * z_2


def F14(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum of z-functions (2D).

    Domain: [0, 4]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    z_1 = z(x_1)
    z_2 = z(x_2)
    return z_1 + z_2


def F15(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Simple quadratic (2D).

    Domain: [-5, 5]
    Variables: 2

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 2)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    return (x_1 - 2) ** 2 + (x_2 - 1) ** 2


def F16(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sine with quadratic term (1D).

    Domain: [-5, 5]
    Variables: 1

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 1)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    return np.sin(x_1) * x_1 * x_1


def F17(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear sine function (1D).

    Domain: [-5, 5]
    Variables: 1

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of shape (n_samples, 1)

    Returns
    -------
    NDArray[np.float64]
        Function values of shape (n_samples,)
    """
    x_1 = x[:, 0]
    return np.sin(x_1) + x_1


problems_dict = {
    "F1": {"function": F1, "bounds": (-1, 1), "n_vars": 1},
    "F2": {"function": F2, "bounds": (-1, 1), "n_vars": 1},
    "F3": {"function": F3, "bounds": (-16, 16), "n_vars": 2},
    "F4": {"function": F4, "bounds": (-16, 16), "n_vars": 2},
    "F5": {"function": F5, "bounds": (-2, 2), "n_vars": 2},
    "F6": {"function": F6, "bounds": (-16, 16), "n_vars": 2},
    "F7": {"function": F7, "bounds": (-5, 5), "n_vars": 2},
    "F8": {"function": F8, "bounds": (-10, 10), "n_vars": 2},
    "F9": {"function": F9, "bounds": (-2.5, 2.5), "n_vars": 2},
    "F10": {"function": F10, "bounds": (-5, 5), "n_vars": 2},
    "F11": {"function": F11, "bounds": (-4, 4), "n_vars": 2},
    "F12": {"function": F12, "bounds": (0, 4), "n_vars": 2},
    "F13": {"function": F13, "bounds": (0, 4), "n_vars": 2},
    "F14": {"function": F14, "bounds": (0, 4), "n_vars": 2},
    "F15": {"function": F15, "bounds": (-5, 5), "n_vars": 2},
    "F16": {"function": F16, "bounds": (-5, 5), "n_vars": 1},
    "F17": {"function": F17, "bounds": (-5, 5), "n_vars": 1},
}
