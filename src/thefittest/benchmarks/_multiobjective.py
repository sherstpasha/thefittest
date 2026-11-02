"""Unconstrained two-objective problems from Table I of Deb et al. (2002).

All objectives are minimized. doi:10.1109/4235.996017.
"""

from abc import ABC
from abc import abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray


class MultiObjectiveProblem(ABC):
    def __init__(
        self,
        n_variables: int,
        left_border: Union[float, NDArray[np.float64]],
        right_border: Union[float, NDArray[np.float64]],
    ) -> None:
        self.n_variables = n_variables
        self.left_border = np.broadcast_to(left_border, (n_variables,)).copy()
        self.right_border = np.broadcast_to(right_border, (n_variables,)).copy()

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_variables:
            raise ValueError(f"Expected a population with shape (N, {self.n_variables}).")
        return self.f(x)

    @abstractmethod
    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class SCH(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(1, -1000.0, 1000.0)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack((x[:, 0] ** 2, (x[:, 0] - 2.0) ** 2))


class FON(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(3, -4.0, 4.0)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        shift = 1.0 / np.sqrt(self.n_variables)
        return np.column_stack(
            (
                1.0 - np.exp(-np.sum((x - shift) ** 2, axis=1)),
                1.0 - np.exp(-np.sum((x + shift) ** 2, axis=1)),
            )
        )


class POL(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(2, -np.pi, np.pi)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        a1 = 0.5 * np.sin(1.0) - 2.0 * np.cos(1.0) + np.sin(2.0) - 1.5 * np.cos(2.0)
        a2 = 1.5 * np.sin(1.0) - np.cos(1.0) + 2.0 * np.sin(2.0) - 0.5 * np.cos(2.0)
        b1 = 0.5 * np.sin(x[:, 0]) - 2.0 * np.cos(x[:, 0]) + np.sin(x[:, 1]) - 1.5 * np.cos(x[:, 1])
        b2 = 1.5 * np.sin(x[:, 0]) - np.cos(x[:, 0]) + 2.0 * np.sin(x[:, 1]) - 0.5 * np.cos(x[:, 1])
        return np.column_stack(
            (
                1.0 + (a1 - b1) ** 2 + (a2 - b2) ** 2,
                (x[:, 0] + 3.0) ** 2 + (x[:, 1] + 1.0) ** 2,
            )
        )


class KUR(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(3, -5.0, 5.0)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        first = np.sum(-10.0 * np.exp(-0.2 * np.sqrt(x[:, :-1] ** 2 + x[:, 1:] ** 2)), axis=1)
        second = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x**3), axis=1)
        return np.column_stack((first, second))


class ZDT1(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(30, 0.0, 1.0)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        first = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        return np.column_stack((first, g * (1.0 - np.sqrt(first / g))))


class ZDT2(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(30, 0.0, 1.0)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        first = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        return np.column_stack((first, g * (1.0 - (first / g) ** 2)))


class ZDT3(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(30, 0.0, 1.0)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        first = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        second = g * (1.0 - np.sqrt(first / g) - (first / g) * np.sin(10.0 * np.pi * first))
        return np.column_stack((first, second))


class ZDT4(MultiObjectiveProblem):
    def __init__(self) -> None:
        lower = np.array([0.0] + [-5.0] * 9)
        upper = np.array([1.0] + [5.0] * 9)
        super().__init__(10, lower, upper)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        first = x[:, 0]
        g = (
            1.0
            + 10.0 * (self.n_variables - 1)
            + np.sum(x[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[:, 1:]), axis=1)
        )
        return np.column_stack((first, g * (1.0 - np.sqrt(first / g))))


class ZDT6(MultiObjectiveProblem):
    def __init__(self) -> None:
        super().__init__(10, 0.0, 1.0)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        first = 1.0 - np.exp(-4.0 * x[:, 0]) * np.sin(6.0 * np.pi * x[:, 0]) ** 6
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1) ** 0.25
        return np.column_stack((first, g * (1.0 - (first / g) ** 2)))
