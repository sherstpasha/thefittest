# Пример использования

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from thefittest.regressors._gpnneregression_one_tree_mo import (
    GeneticProgrammingNeuralNetStackingRegressorMO,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SHADE
from thefittest.benchmarks import SolarBatteryDegradationDataset


# Объединённая функция для вычисления метрик
def calculate_metrics(y_true, y_pred, y_train):
    metrics = {}

    # MASE
    naive_forecast = np.roll(y_train, 1, axis=0)[1:]
    mae_naive = np.mean(np.abs(y_train[1:] - naive_forecast))
    metrics["MASE"] = np.mean(np.abs(y_true - y_pred)) / mae_naive

    # sMAPE
    metrics["sMAPE"] = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )

    # RMSE
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)

    # R²
    metrics["R²"] = r2_score(y_true, y_pred)

    # MSE
    metrics["MSE"] = mean_squared_error(y_true, y_pred)

    # MAPE
    metrics["MAPE"] = 100 * np.mean(np.abs((y_true - y_pred) / y_true))

    return metrics


# Генерируем синтетические данные
# n_samples — количество объектов; n_features — число признаков; n_targets — число выходов
dataset = SolarBatteryDegradationDataset()
X = dataset.get_X()
y = dataset.get_y()


# Разбиваем на train/test
X_train, X_test = X[:169], X[169:]
y_train, y_test = y[:169], y[169:]

# Масштабируем X
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Масштабируем y
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Инициализируем и обучаем модель
model = GeneticProgrammingNeuralNetStackingRegressorMO(
    iters=10,
    pop_size=10,
    input_block_size=1,
    optimizer=PDPSHAGP,
    optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 5},
    weights_optimizer=SHADE,
    weights_optimizer_args={
        "iters": 500,
        "pop_size": 300,
        "no_increase_num": 100,
        "fitness_update_eps": 0.0001,
    },
    test_sample_ratio=0.25,
)
model.fit(X_train_scaled, y_train_scaled)

# Делаем предсказания и возвращаем их к исходному масштабу
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Вывод размеров массивов и метрики
print("Shapes:", y_pred.shape, y_test.shape, y_train.shape)
metrics = calculate_metrics(y_test, y_pred, y_train)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


print(y_test)
print(y_pred)

import matplotlib.pyplot as plt

# Количество выходов
n_outputs = y_test.shape[1]

# Строим графики
for i in range(n_outputs):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:, i], label=f"True Target {i+1}", marker="o")
    plt.plot(y_pred[:, i], label=f"Predicted Target {i+1}", marker="x")
    plt.title(f"Comparison for Target {i+1}")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
