from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from sklearn.base import RegressorMixin
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


class MLPEARegressor(RegressorMixin, BaseMLPEA):
    def __init__(
        self,
        *,
        n_iter: int = 100,
        pop_size: int = 500,
        hidden_layers: Tuple[int, ...] = (100,),
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "MLPEARegressor requires PyTorch. " "Install with: pip install thefittest[torch]"
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

    def predict(self, X: ArrayLike):
        check_is_fitted(self)

        if hasattr(self, "_validate_data"):
            X = self._validate_data(X, reset=False)
        else:
            X = validate_data(self, X, reset=False)
        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device(self.device)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            out = self.net_.forward(X_t)

        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy().astype(np.float64)
        if out.ndim == 3 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.ndim == 2 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out.reshape(-1)
