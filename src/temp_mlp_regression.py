from thefittest.regressors import MLPEARegressor

from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from thefittest.optimizers import SHAGA
import torch

from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"


def problem(x):
    return np.sin(x[:, 0] * 3) * x[:, 0] * 0.5


n_dimension = 1
left_border = -4.5
right_border = 4.5
sample_size = 100


X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
noise = np.random.normal(0, 0.1, size=sample_size)
y = problem(X) + noise

X_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)
X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
y_test = y_test

model = MLPEARegressor(
    n_iter=5000,
    pop_size=500,
    hidden_layers=[100],
    weights_optimizer=Adam,  # ключ: torch оптимизатор-класс
    weights_optimizer_args={"lr": 1e-2, "show_progress_each": 10},  # и его аргументы
    # weights_optimizer=SHAGA,
    # weights_optimizer_args={"show_progress_each": 10},
)

model.fit(X_train, y_train)
predict = model.predict(X_test)

print("r2_score:", r2_score(y_test, predict))
