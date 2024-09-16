import numpy as np
from sklearn.preprocessing import StandardScaler
from thefittest.regressors._gpnnregression_multi_out import (
    GeneticProgrammingNeuralNetRegressorMO,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from thefittest.base._net import Net


def _predict(X, net, offset=True):
    if offset:
        X = np.hstack([X, np.ones((X.shape[0], 1))])

    output = net.forward(X)[0]
    return output


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
X_train = np.load(r"C:\Users\pasha\forecast\forecast\X_train.npy")
y_train = np.load(r"C:\Users\pasha\forecast\forecast\y_train.npy")
X_test = np.load(r"C:\Users\pasha\forecast\forecast\X_test.npy")
y_test = np.load(r"C:\Users\pasha\forecast\forecast\y_test.npy")

# Масштабирование данных X
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Масштабирование данных y
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


net = Net.load_from_file(r"C:\Users\pasha\OneDrive\Рабочий стол\meteo_res\gpnn_model.pkl")

y_pred_scaled = _predict(X_test_scaled, net)

print(y_pred_scaled.shape)

y_pred = scaler_y.inverse_transform(y_pred_scaled)

print(y_pred.shape, y_test.shape, y_train.shape)

print(calculate_metrics(y_test, y_pred, y_train))
