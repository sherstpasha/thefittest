from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ..base._gp import BaseGP
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP


class GeneticProgrammingRegressor(RegressorMixin, BaseGP):
    """
    Genetic Programming-based regressor using evolved symbolic expressions.

    This regressor evolves mathematical expressions (trees) to perform symbolic
    regression by learning explicit functional relationships between input features
    and target values.

    Parameters
    ----------
    n_iter : int, optional (default=300)
        Number of iterations (generations) for the GP optimization.
    pop_size : int, optional (default=1000)
        Population size for the genetic programming algorithm.
    functional_set_names : Tuple[str, ...], optional
        Tuple of function names to use in evolved expressions.
        Default: ('cos', 'sin', 'add', 'sub', 'mul', 'div')
        Available functions: 'cos', 'sin', 'add', 'sub', 'mul', 'div',
        'abs', 'logabs', 'exp', 'sqrtabs'.
    optimizer : Type[Union[SelfCGP, GeneticProgramming, PDPGP]], optional (default=SelfCGP)
        Genetic programming optimizer class to use.
        Available: SelfCGP (self-configuring), GeneticProgramming (standard),
        or PDPGP (with dynamic operator probabilities).
    optimizer_args : Optional[dict], optional (default=None)
        Additional arguments passed to the optimizer (excluding n_iter and pop_size).
        Common args: {'show_progress_each': 10, 'max_level': 5}
    random_state : Optional[Union[int, np.random.RandomState]], optional (default=None)
        Random state for reproducibility.
    use_fitness_cache : bool, optional (default=False)
        If True, caches fitness evaluations to avoid redundant computations.

    Attributes
    ----------
    tree_ : Tree
        The evolved tree expression representing the symbolic model.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    **Symbolic Regression with Visualization**

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from thefittest.regressors import GeneticProgrammingRegressor
    >>> from thefittest.optimizers import PDPGP
    >>>
    >>> # Define the true function
    >>> def problem(x):
    ...     return np.sin(x[:, 0] * 3) * x[:, 0] * 0.5
    >>>
    >>> # Generate training data
    >>> n_dimension = 2
    >>> left_border = -4.5
    >>> right_border = 4.5
    >>> sample_size = 100
    >>>
    >>> X = np.array([np.linspace(left_border, right_border, sample_size)
    ...               for _ in range(n_dimension)]).T
    >>> y = problem(X)
    >>>
    >>> # Train the model
    >>> model = GeneticProgrammingRegressor(
    ...     n_iter=500,
    ...     pop_size=1000,
    ...     optimizer=PDPGP,
    ...     optimizer_args={'show_progress_each': 10, 'max_level': 5}
    ... )
    >>> model.fit(X, y)
    >>> predict = model.predict(X)
    >>>
    >>> # Get the evolved symbolic expression
    >>> tree = model.get_tree()
    >>> print('Evolved expression:', tree)
    >>>
    >>> # Visualize results
    >>> fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=1)
    >>> ax[0].plot(X[:, 0], y, label='True y')
    >>> ax[0].plot(X[:, 0], predict, label='Predicted y')
    >>> ax[0].legend()
    >>> tree.plot(ax=ax[1])
    >>> plt.tight_layout()
    >>> plt.show()

    Notes
    -----
    The regressor evolves symbolic expressions that explicitly represent the
    relationship between inputs and outputs. The resulting model is interpretable
    and can be analyzed mathematically.
    """

    def __init__(
        self,
        *,
        n_iter: int = 300,
        pop_size: int = 1000,
        functional_set_names: Tuple[str, ...] = ("cos", "sin", "add", "sub", "mul", "div"),
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        use_fitness_cache: bool = False,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            functional_set_names=functional_set_names,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            random_state=random_state,
            use_fitness_cache=use_fitness_cache,
        )

    def predict(self, X: NDArray[np.float64]):
        """
        Predict target values for X using the evolved symbolic expression.

        Parameters
        ----------
        X : NDArray[np.float64], shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self)

        if hasattr(self, "_validate_data"):
            X = self._validate_data(X, reset=False)
        else:
            X = validate_data(self, X, reset=False)

        n_features = X.shape[1]
        tree_for_predict = self.tree_.set_terminals(**{f"x{i}": X[:, i] for i in range(n_features)})
        y_predict = tree_for_predict() * np.ones(len(X))
        return y_predict
