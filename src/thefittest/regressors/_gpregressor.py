from __future__ import annotations

from copy import deepcopy
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
from ..base._gp import fitness_function_gp_multiobjective
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP
from ..optimizers import NSGPII


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
    optimizer : Type[Union[SelfCGP, GeneticProgramming, PDPGP, NSGPII]], optional (default=SelfCGP)
        Genetic programming optimizer class to use.
        Available: SelfCGP (self-configuring), GeneticProgramming (standard),
        PDPGP (with dynamic operator probabilities), or NSGPII (multiobjective).
    optimizer_args : Optional[dict], optional (default=None)
        Additional arguments passed to the optimizer (excluding n_iter and pop_size).
        Common args: {'show_progress_each': 10, 'max_level': 5}
    random_state : Optional[Union[int, np.random.RandomState]], optional (default=None)
        Random state for reproducibility.
    use_fitness_cache : bool, optional (default=True)
        If True, caches fitness evaluations to avoid redundant computations.
    complexity : str, optional (default="n_nodes")
        Second minimization objective for NSGPII: "n_nodes", "n_levels",
        "n_functions", or "n_terminals". Ignored by single-objective optimizers.
    model_selection : str, optional (default="best_error")
        Select a Pareto model by "best_error" or "simplest". Ties are resolved
        by the other objective. Ignored by single-objective optimizers.

    Attributes
    ----------
    tree_ : Tree
        The evolved tree expression representing the symbolic model.
    pareto_front_ : dict
        Pareto trees and their RMSE and complexity values, available with NSGPII.
    selected_tree_index_ : int
        Index of the active model in pareto_front_, available with NSGPII.
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
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming], Type[NSGPII]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        use_fitness_cache: bool = True,
        complexity: str = "n_nodes",
        model_selection: str = "best_error",
    ) -> None:
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            functional_set_names=functional_set_names,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            random_state=random_state,
            use_fitness_cache=use_fitness_cache,
        )
        self.complexity = complexity
        self.model_selection = model_selection

    def _fit_regression(self, optimizer_args: dict[str, Any], y: NDArray[np.float64]) -> None:
        if not issubclass(self.optimizer, NSGPII):
            super()._fit_regression(optimizer_args, y)
            self.__dict__.pop("pareto_front_", None)
            self.__dict__.pop("selected_tree_index_", None)
            return

        if self.complexity not in ("n_nodes", "n_levels", "n_functions", "n_terminals"):
            raise ValueError(f"Unknown tree complexity criterion: {self.complexity}.")
        if self.model_selection not in ("best_error", "simplest"):
            raise ValueError('model_selection must be "best_error" or "simplest".')
        optimizer_args["fitness_function"] = fitness_function_gp_multiobjective
        optimizer_args["fitness_function_args"] = {
            "y": y,
            "complexity": self.complexity,
            "fitness_cache": self._fitness_cache,
        }
        optimizer_args["minimization"] = (True, True)
        optimizer_ = self.optimizer(**optimizer_args)
        optimizer_.fit()
        self.pareto_front_ = optimizer_.get_pareto_front()
        self.optimizer_stats_ = optimizer_.get_stats()
        errors, complexities = self.pareto_front_["fitness"].T
        if self.model_selection == "best_error":
            order = np.lexsort((complexities, errors))
        else:
            order = np.lexsort((errors, complexities))
        self.select_tree(int(order[0]))

    def get_pareto_front(self) -> dict[str, Any]:
        """Return a detached copy of the fitted Pareto trees and objectives."""
        check_is_fitted(self)
        if not hasattr(self, "pareto_front_"):
            raise ValueError("A Pareto front is available only after fitting with NSGPII.")
        return deepcopy(self.pareto_front_)

    def select_tree(self, index: int) -> GeneticProgrammingRegressor:
        """Activate a Pareto tree for prediction without refitting.

        Parameters
        ----------
        index : int
            Zero-based position in the fitted Pareto front.
        """
        check_is_fitted(self)
        if not hasattr(self, "pareto_front_"):
            raise ValueError("A Pareto front is available only after fitting with NSGPII.")
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer.")
        if not 0 <= index < len(self.pareto_front_["fitness"]):
            raise IndexError("index is outside the Pareto front.")
        self.tree_ = deepcopy(self.pareto_front_["phenotype"][index])
        self.selected_tree_index_ = int(index)
        return self

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
