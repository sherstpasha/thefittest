from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING

import warnings

import numpy as np
from numpy.typing import NDArray
from numpy.typing import ArrayLike

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if TYPE_CHECKING:
        import torch

from ..base._mlp import BaseMLPEA
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import SHADE


class MLPEAClassifier(ClassifierMixin, BaseMLPEA):
    """
    Multi-Layer Perceptron classifier with Evolutionary Algorithm-based training.

    This classifier uses evolutionary algorithms to optimize neural network weights
    instead of traditional gradient-based methods.

    Parameters
    ----------
    n_iter : int, optional (default=100)
        Number of iterations (generations) for weight optimization.
    pop_size : int, optional (default=500)
        Population size for the evolutionary algorithm.
    hidden_layers : Tuple[int, ...], optional (default=(0,))
        Tuple specifying the number of neurons in each hidden layer.
        Empty tuple or (0,) means no hidden layers (linear model).
        Example: (5, 5) creates two hidden layers with 5 neurons each.
    activation : str, optional (default="sigma")
        Activation function for hidden layers.
        Available: 'sigma' (sigmoid), 'relu', 'gauss' (Gaussian), 'tanh',
        'ln' (natural logarithm normalization), 'softmax'.
    offset : bool, optional (default=True)
        If True, adds bias terms to the network.
    weights_optimizer : Type, optional (default=SHADE)
        Evolutionary algorithm class for optimizing weights, or PyTorch optimizer.
        Available EA: SHADE, jDE, DifferentialEvolution, SHAGA, etc.
        Available torch.optim: Adam, SGD, RMSprop, etc.
        Note: When using torch.optim optimizers, pop_size parameter is ignored.
    weights_optimizer_args : Optional[dict], optional (default=None)
        Additional arguments passed to the weights optimizer (excluding n_iter and pop_size).
        For EA optimizers: {'show_progress_each': 10}
        For torch.optim: {'lr': 0.01, 'weight_decay': 0.0001}
        Note: Use 'epochs' or 'iters' to set training iterations for torch.optim.
    random_state : Optional[Union[int, np.random.RandomState]], optional (default=None)
        Random state for reproducibility.
    device : str, optional (default="cpu")
        Device for PyTorch computations: 'cpu' or 'cuda'.


    Examples
    --------
    **Multi-class Classification with Iris Dataset**

    >>> from thefittest.optimizers import SHAGA
    >>> from thefittest.benchmarks import IrisDataset
    >>> from thefittest.classifiers import MLPEAClassifier
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
    >>> # Create and train classifier
    >>> model = MLPEAClassifier(
    ...     n_iter=500,
    ...     pop_size=500,
    ...     hidden_layers=[5, 5],
    ...     weights_optimizer=SHAGA,
    ...     weights_optimizer_args={'show_progress_each': 10}
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

    The classifier uses evolutionary algorithms to find optimal network weights,
    which can be more robust to local minima compared to gradient descent but
    may require more function evaluations.
    """

    def __init__(
        self,
        *,
        n_iter: int = 100,
        pop_size: int = 500,
        hidden_layers: Tuple[int, ...] = (0,),
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "MLPEAClassifier requires PyTorch. " "Install with: pip install thefittest[torch]"
            )
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            hidden_layers=hidden_layers,
            activation=activation,
            offset=offset,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
            random_state=random_state,
            device=device,
        )

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Predict class probabilities for X using the trained neural network.

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

        return proba_t.detach().cpu().numpy().astype(np.float64)

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
