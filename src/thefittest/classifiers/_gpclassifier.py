from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ..base._gp import BaseGP
from ..base._gp import safe_sigmoid
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP


class GeneticProgrammingClassifier(ClassifierMixin, BaseGP):
    """
    Genetic Programming-based classifier using evolved symbolic expressions.

    This classifier evolves mathematical expressions (trees) to perform classification
    by learning symbolic representations of decision boundaries. It can handle both
    binary and multi-class classification problems.

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


    Examples
    --------
    **Binary Classification**

    >>> from thefittest.classifiers import GeneticProgrammingClassifier
    >>> from thefittest.optimizers import SelfCGP
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # Generate binary classification data
    >>> X, y = make_classification(n_samples=100, n_features=4, n_classes=2)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>>
    >>> # Create and train classifier
    >>> model = GeneticProgrammingClassifier(
    ...     n_iter=100,
    ...     pop_size=500,
    ...     optimizer=SelfCGP,
    ...     optimizer_args={'show_progress_each': 20, 'max_level': 5}
    ... )
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Make predictions
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    >>> tree = model.get_tree()
    >>> print('Evolved expression:', tree)

    Notes
    -----
    The classifier evolves symbolic expressions that map input features to class
    probabilities using sigmoid activation. For multi-class problems, it evolves
    one tree per class using one-vs-rest approach.
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

    def predict_proba(self, X: NDArray[np.float64]):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : NDArray[np.float64], shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : NDArray[np.float64], shape (n_samples, n_classes)
            Class probabilities for each sample.
            For binary classification: [[P(class=0), P(class=1)], ...]
            For multi-class: [[P(class=0), ..., P(class=K-1)], ...]
        """
        check_is_fitted(self)

        if hasattr(self, "_validate_data"):
            X = self._validate_data(X, reset=False)
        else:
            X = validate_data(self, X, reset=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_}."
            )

        def _tree_out(tree):
            tfp = tree.set_terminals(**{f"x{i}": X[:, i] for i in range(self.n_features_in_)})
            return tfp() * np.ones(len(X))

        if (
            hasattr(self, "trees_")
            and isinstance(self.trees_, (list, tuple))
            and len(self.trees_) > 0
        ):
            logits = np.column_stack([_tree_out(t) for t in self.trees_])  # (N, K)
            P = safe_sigmoid(logits)
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            proba = (P / row_sums).astype(np.float64, copy=False)
            return proba

        tree_output = _tree_out(self.tree_)
        p1 = safe_sigmoid(tree_output)
        proba = np.vstack([1.0 - p1, p1]).T.astype(np.float64, copy=False)
        return proba

    def predict(self, X: NDArray[np.float64]):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : NDArray[np.float64], shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)  # ndarray (N, K)
        indeces = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indeces)
