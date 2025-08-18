from thefittest.regressors import GeneticProgrammingNeuralNetRegressor

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

model = GeneticProgrammingNeuralNetRegressor(
    n_iter=5,
    pop_size=10,
    input_block_size=8,
    weights_optimizer=Adam,  # ключ: torch оптимизатор-класс
    weights_optimizer_args={
        "lr": 1e-2,
        "epochs": 3000,
        "show_progress_each": 100,
    },  # и его аргументы
    optimizer_args={"show_progress_each": 1, 'n_jobs': 10},
    # weights_optimizer=SHADE,
    # weights_optimizer_args={"iters": 30, "pop_size": 30, "show_progress_each": 10},
    device=device,
)

model.fit(X_train, y_train)
predict = model.predict(X_test)

print("r2_score:", r2_score(y_test, predict))
