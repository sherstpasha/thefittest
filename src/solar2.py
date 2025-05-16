import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed

from thefittest.benchmarks import SolarBatteryDegradationDataset


# === Метрики ===
def calculate_global_normalized_error(y_true, y_pred):
    s, m = y_true.shape
    total_error = 0.0
    for i in range(m):
        y_max = np.max(y_true[:, i])
        y_min = np.min(y_true[:, i])
        denom = y_max - y_min if y_max != y_min else 1.0
        total_error += np.sum(np.abs(y_true[:, i] - y_pred[:, i])) / denom
    return (100 / s) * (1 / m) * total_error


# === Загрузка данных ===
dataset = SolarBatteryDegradationDataset()
X = dataset.get_X()[:, 6].reshape(-1, 1)

y = dataset.get_y()
target_names = list(dataset.get_y_names())
print(X.shape, y.shape)
X_train, X_test = X[:169], X[169:]
y_train, y_test = y[:169], y[169:]

# === Модели и гиперпараметры ===
models_and_params = {
    "RandomForest": (
        RandomForestRegressor(),
        {"n_estimators": [100, 200], "max_depth": [5, 10]},
    ),
    "MLP": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(max_iter=50000, n_iter_no_change=500)),
            ]
        ),
        {
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate": ["constant", "adaptive"],
        },
    ),
}

# === Подготовка папки ===
os.makedirs("results", exist_ok=True)


# === Обучение одного регрессора ===
def train_model(model_name, model, param_grid, target_index, target_name):
    y_train_i, y_test_i = y_train[:, target_index], y_test[:, target_index]

    search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    search.fit(X_train, y_train_i)

    best_model = search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_all_true = np.concatenate([y_train_i, y_test_i])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])

    result = {
        "Model": model_name,
        "Target": target_name,
        "GNE_train": calculate_global_normalized_error(
            y_train_i.reshape(-1, 1), y_train_pred.reshape(-1, 1)
        ),
        "GNE_test": calculate_global_normalized_error(
            y_test_i.reshape(-1, 1), y_test_pred.reshape(-1, 1)
        ),
        "GNE_full": calculate_global_normalized_error(
            y_all_true.reshape(-1, 1), y_all_pred.reshape(-1, 1)
        ),
    }

    # Сохраняем модель
    model_path = f"results/{model_name}_{target_name}.joblib"
    joblib.dump(best_model, model_path)

    # Сохраняем график с точками
    plt.figure(figsize=(12, 5))
    indices_train = np.arange(len(y_train_i))
    indices_test = np.arange(len(y_train_i), len(y_train_i) + len(y_test_i))

    plt.scatter(indices_train, y_train_i, label="Train True", s=30, alpha=0.7, marker="x")
    plt.scatter(indices_train, y_train_pred, label="Train Pred", s=30, marker="x")
    plt.scatter(indices_test, y_test_i, label="Test True", s=30, alpha=0.7, marker="x")
    plt.scatter(indices_test, y_test_pred, label="Test Pred", s=30, marker="x")

    plt.axvline(len(y_train_i), color="gray", linestyle=":", label="Train/Test Split")
    plt.title(f"{target_name} — {model_name}")
    plt.xlabel("Sample Index")
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_{target_name}.png")
    plt.close()

    return result


# === Параллельное обучение ===
tasks = [
    delayed(train_model)(model_name, model, param_grid, i, target_name)
    for i, target_name in enumerate(target_names)
    for model_name, (model, param_grid) in models_and_params.items()
]
results = Parallel(n_jobs=-1)(tasks)

# === Сохраняем GNE-метрики ===
df_results = pd.DataFrame(results)
df_results.to_csv("results/all_metrics.csv", index=False)

# === Красивый барплот GNE ===
plt.figure(figsize=(14, 6))
df_long = pd.melt(
    df_results,
    id_vars=["Model", "Target"],
    value_vars=["GNE_train", "GNE_test", "GNE_full"],
    var_name="Dataset",
    value_name="GNE (%)",
)
df_long["Label"] = df_long["Model"] + " (T" + df_long["Target"].astype(str) + ")"
sorted_labels = df_long[df_long["Dataset"] == "GNE_test"].sort_values("GNE (%)")["Label"]

sns.barplot(
    data=df_long,
    x="GNE (%)",
    y="Label",
    hue="Dataset",
    order=sorted_labels,
    palette="Set2",
)
plt.title("Global Normalized Error (по выходам и наборам)")
plt.xlabel("GNE (%)")
plt.ylabel("Модель (Выход)")
plt.legend(title="Набор")
plt.tight_layout()
plt.savefig("results/GNE_barplot.png")
plt.close()

print("\n✅ Все модели обучены. Результаты и графики сохранены в 'results/'")
