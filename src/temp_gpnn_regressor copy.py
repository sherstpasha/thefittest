import time
from thefittest.regressors import GeneticProgrammingNeuralNetRegressor
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim import Adam
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Проблема ----
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

# фиксируем random_state для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# ---- Эксперименты ----
n_jobs_list = [None, 2, 4, 6, 10]  # None = не указываем
results = []

for nj in n_jobs_list:
    print(f"\n=== Эксперимент с n_jobs={nj} ===")
    optimizer_args = {"show_progress_each": 1}
    if nj is not None:
        optimizer_args["n_jobs"] = nj

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=5,
        pop_size=10,
        input_block_size=8,
        weights_optimizer=Adam,
        weights_optimizer_args={
            "lr": 1e-2,
            "epochs": 3000,
            "show_progress_each": 100,
        },
        optimizer_args=optimizer_args,
        device=device,
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    predict = model.predict(X_test)
    score = r2_score(y_test, predict)

    print(f"Время: {elapsed:.2f} сек | r2_score: {score:.4f}")
    results.append((nj, elapsed, score))

# ---- Итог ----
print("\nИТОГИ:")
for nj, elapsed, score in results:
    print(f"n_jobs={nj}: время={elapsed:.2f} сек, r2_score={score:.4f}")
