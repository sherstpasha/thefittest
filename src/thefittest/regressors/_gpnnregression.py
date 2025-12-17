from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Type
from typing import Union
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if TYPE_CHECKING:
        import torch

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ..base._gpnn import BaseGPNN
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SelfCGP


class GeneticProgrammingNeuralNetRegressor(RegressorMixin, BaseGPNN):
    """
    Genetic Programming-based Neural Network regressor with evolved architecture.

    This regressor evolves both the neural network architecture and weights using
    genetic programming. The network structure is represented as a tree, and weights
    are optimized using evolutionary algorithms or gradient-based optimizers.

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
        Optimizer for network weights. Can be evolutionary algorithm or torch.optim optimizer.
        Available: SHADE, SHAGA, jDE, or torch.optim.Adam, torch.optim.SGD, etc.
    weights_optimizer_args : Optional[dict], optional (default=None)
        Additional arguments for the weights optimizer.
        For EA optimizers: {'iters': 150, 'pop_size': 150, 'show_progress_each': 10}
        For torch.optim: {'iters': 1000, 'lr': 0.01} (pop_size is ignored)
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

    Attributes
    ----------
    net_ : torch.nn.Module
        The evolved and trained neural network.
    tree_ : Tree
        Tree representation of the evolved architecture.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    **Regression with Evolved Architecture and EA Optimizer**

    >>> import numpy as np
    >>> from thefittest.regressors import GeneticProgrammingNeuralNetRegressor
    >>> from thefittest.optimizers import PDPGP, SHAGA
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import scale
    >>> from sklearn.metrics import r2_score
    >>>
    >>> # Define the problem
    >>> def problem(x):
    ...     return np.sin(x[:, 0] * 3) * x[:, 0] * 0.5
    >>>
    >>> # Generate data
    >>> n_dimension = 1
    >>> sample_size = 100
    >>> X = np.array([np.linspace(-4.5, 4.5, sample_size)
    ...               for _ in range(n_dimension)]).T
    >>> noise = np.random.normal(0, 0.1, size=sample_size)
    >>> y = problem(X) + noise
    >>>
    >>> X_scaled = scale(X)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X_scaled, y, test_size=0.1
    ... )
    >>>
    >>> # Train with evolutionary algorithm for weights
    >>> model = GeneticProgrammingNeuralNetRegressor(
    ...     n_iter=10,
    ...     pop_size=10,
    ...     optimizer=PDPGP,
    ...     optimizer_args={'show_progress_each': 1},
    ...     weights_optimizer=SHAGA,
    ...     weights_optimizer_args={'iters': 150, 'pop_size': 150}
    ... )
    >>> model.fit(X_train, y_train)
    >>> predict = model.predict(X_test)
    >>> print("r2_score:", r2_score(y_test, predict))

    **Using PyTorch Optimizer (Adam) for Weight Training**

    >>> import torch.optim as optim
    >>>
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> model = GeneticProgrammingNeuralNetRegressor(
    ...     n_iter=10,
    ...     pop_size=10,
    ...     optimizer=PDPGP,
    ...     optimizer_args={'show_progress_each': 1},
    ...     weights_optimizer=optim.Adam,
    ...     weights_optimizer_args={'iters': 1000, 'lr': 0.01},
    ...     device=device
    ... )
    >>> model.fit(X_train, y_train)
    >>> predict = model.predict(X_test)

    Notes
    -----
    Requires PyTorch. Install with: pip install thefittest[torch]

    This is a two-stage optimization: first, GP evolves the network architecture,
    then for each architecture, weights are optimized using either evolutionary
    algorithms or gradient-based methods. This can discover novel network
    structures but is computationally intensive.
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
                "GeneticProgrammingNeuralNetRegressor requires PyTorch. "
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

    def predict(self, X: NDArray[np.float64]):
        """
        Predict target values using the evolved neural network.

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

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device(self.device)
        X_t = torch.as_tensor(X, dtype=torch.float64, device=device)

        with torch.no_grad():
            out = self.net_.forward(X_t)

        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().to(torch.float64).numpy()
        if out.ndim == 3 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.ndim == 2 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return np.ascontiguousarray(out.reshape(-1), dtype=np.float64)
