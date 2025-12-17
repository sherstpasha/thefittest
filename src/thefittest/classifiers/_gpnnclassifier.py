from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Type
from typing import Union
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from numpy.typing import ArrayLike

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if TYPE_CHECKING:
        import torch

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ..base._gpnn import BaseGPNN
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SelfCGP


class GeneticProgrammingNeuralNetClassifier(ClassifierMixin, BaseGPNN):
    """
    Genetic Programming-based Neural Network classifier with evolved architecture.

    This classifier evolves both the neural network architecture and weights using
    genetic programming. The network structure is represented as a tree, and weights
    are optimized using evolutionary algorithms.

    Parameters
    ----------
    n_iter : int, optional (default=15)
        Number of iterations (generations) for architecture evolution.
    pop_size : int, optional (default=50)
        Population size for the architecture evolution.
    input_block_size : int, optional (default=1)
        Size of input processing blocks.
    max_hidden_block_size : int, optional (default=9)
        Maximum size of hidden layer blocks.
    offset : bool, optional (default=True)
        If True, adds bias terms to the network.
    test_sample_ratio : float, optional (default=0.5)
        Ratio of data to use for validation during evolution.
    optimizer : Type[Union[SelfCGP, GeneticProgramming, PDPGP]], optional (default=SelfCGP)
        Genetic programming optimizer for evolving architecture.
        Available: SelfCGP, GeneticProgramming, or PDPGP.
    optimizer_args : Optional[dict], optional (default=None)
        Additional arguments for the architecture optimizer (excluding n_iter and pop_size).
    weights_optimizer : Type, optional (default=SHADE)
        Evolutionary algorithm for optimizing network weights.
    weights_optimizer_args : Optional[dict], optional (default=None)
        Additional arguments for the weights optimizer.
        Note: Use 'iters' and 'pop_size' keys for setting iterations and population size.
        Example: {'iters': 150, 'pop_size': 150, 'show_progress_each': 10}
    net_size_penalty : float, optional (default=0.0)
        Penalty coefficient for network complexity (larger = simpler networks).
    random_state : Optional[Union[int, np.random.RandomState]], optional (default=None)
        Random state for reproducibility.
    device : str, optional (default="cpu")
        Device for PyTorch computations: 'cpu' or 'cuda'.
    use_fitness_cache : bool, optional (default=False)
        If True, caches fitness evaluations.
    fitness_cache_size : int, optional (default=1000)
        Maximum size of fitness cache.

    Examples
    --------
    **Multi-class Classification with Evolved Architecture**

    >>> from thefittest.optimizers import SelfCGP, SHAGA
    >>> from thefittest.benchmarks import IrisDataset
    >>> from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import minmax_scale
    >>> from sklearn.metrics import confusion_matrix, f1_score
    >>>
    >>> # Load and prepare data
    >>> data = IrisDataset()
    >>> X = data.get_X()
    >>> y = data.get_y()
    >>> X_scaled = minmax_scale(X)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X_scaled, y, test_size=0.1
    ... )
    >>>
    >>> # Create classifier with evolved architecture
    >>> model = GeneticProgrammingNeuralNetClassifier(
    ...     n_iter=10,
    ...     pop_size=10,
    ...     optimizer=SelfCGP,
    ...     optimizer_args={'show_progress_each': 1},
    ...     weights_optimizer=SHAGA,
    ...     weights_optimizer_args={'iters': 150, 'pop_size': 150}
    ... )
    >>>
    >>> model.fit(X_train, y_train)
    >>> predict = model.predict(X_test)
    >>>
    >>> print("Confusion matrix:\\n", confusion_matrix(y_test, predict))
    >>> print("F1 score:", f1_score(y_test, predict, average="macro"))

    Notes
    -----
    Requires PyTorch. Install with: pip install thefittest[torch]

    This is a two-stage optimization: first, GP evolves the network architecture,
    then for each architecture, an EA optimizes the weights. This can discover
    novel network structures but is computationally intensive.
    """

    def __init__(
        self,
        *,
        n_iter: int = 15,
        pop_size: int = 50,
        input_block_size: int = 1,
        max_hidden_block_size: int = 9,
        offset: bool = True,
        test_sample_ratio: float = 0.5,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        net_size_penalty: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        device: str = "cpu",
        use_fitness_cache: bool = False,
        fitness_cache_size: int = 1000,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "GeneticProgrammingNeuralNetClassifier requires PyTorch. "
                "Install with: pip install thefittest[torch]"
            )
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            input_block_size=input_block_size,
            max_hidden_block_size=max_hidden_block_size,
            offset=offset,
            test_sample_ratio=test_sample_ratio,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
            net_size_penalty=net_size_penalty,
            random_state=random_state,
            device=device,
            use_fitness_cache=use_fitness_cache,
            fitness_cache_size=fitness_cache_size,
        )

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Predict class probabilities using the evolved neural network.

        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : NDArray[np.float64], shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        check_is_fitted(self)

        if hasattr(self, "_validate_data"):
            X = self._validate_data(X, reset=False)
        else:
            X = validate_data(self, X, reset=False)

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device(self.device)
        X_t = torch.as_tensor(X, dtype=torch.float64, device=device)

        with torch.no_grad():
            proba_t = self.net_.forward(X_t)
        proba = proba_t.detach().cpu().to(torch.float64).numpy()
        return np.ascontiguousarray(proba, dtype=np.float64)

    def predict(self, X: ArrayLike):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : ArrayLike, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indeces = np.argmax(proba, axis=1)

        return self._label_encoder.inverse_transform(indeces)
