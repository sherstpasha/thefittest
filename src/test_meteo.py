import numpy as np
from sklearn.preprocessing import StandardScaler
from thefittest.regressors._mlpearegressor_multi_out import MLPEARegressorMO
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Объединенная функция для вычисления метрик
def calculate_metrics(y_true, y_pred, y_train):
    metrics = {}

    # MASE
    naive_forecast = np.roll(y_train, 1)[1:]
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


# Загрузка данных из файлов
X_train = np.load(r"C:\Users\pasha\OneDrive\Рабочий стол\meteo_res\data\X_train_3.npy")
y_train = np.load(r"C:\Users\pasha\OneDrive\Рабочий стол\meteo_res\data\y_train_3.npy")
X_test = np.load(r"C:\Users\pasha\OneDrive\Рабочий стол\meteo_res\data\X_test_3.npy")
y_test = np.load(r"C:\Users\pasha\OneDrive\Рабочий стол\meteo_res\data\y_test_3.npy")

# Масштабирование данных X
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Масштабирование данных y
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


model = MLPEARegressorMO(
    iters=200, pop_size=100, hidden_layers=(10,), weights_optimizer_args={"show_progress_each": 1}
)

model.fit(X_train_scaled, y_train_scaled)


y_pred_scaled = model.predict(X_test_scaled)

print(y_pred_scaled.shape)

y_pred = scaler_y.inverse_transform(y_pred_scaled)

print(y_pred.shape, y_test.shape, y_train.shape)

print(calculate_metrics(y_test, y_pred, y_train))
