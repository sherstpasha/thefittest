import numpy as np
from sklearn.preprocessing import StandardScaler
from thefittest.regressors._gpnnregression_multi_out import GeneticProgrammingNeuralNetRegressorMO
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from thefittest.tools.print import print_net
import matplotlib.pyplot as plt
import os
import cloudpickle
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


# Объединенная функция для вычисления метрик
def calculate_metrics(y_true, y_pred, y_train):
    metrics = {}

    for i in range(y_true.shape[1]):
        metrics[f"MASE_output_{i}"] = []
        metrics[f"sMAPE_output_{i}"] = []
        metrics[f"RMSE_output_{i}"] = []
        metrics[f"MAE_output_{i}"] = []
        metrics[f"R²_output_{i}"] = []
        metrics[f"MSE_output_{i}"] = []
        metrics[f"MAPE_output_{i}"] = []

    for i in range(y_true.shape[1]):
        naive_forecast = np.roll(y_train[:, i], 1)[1:]
        mae_naive = np.mean(np.abs(y_train[1:, i] - naive_forecast))
        metrics[f"MASE_output_{i}"] = np.mean(np.abs(y_true[:, i] - y_pred[:, i])) / mae_naive

        metrics[f"sMAPE_output_{i}"] = 100 * np.mean(
            2 * np.abs(y_pred[:, i] - y_true[:, i]) / (np.abs(y_true[:, i]) + np.abs(y_pred[:, i]))
        )

        metrics[f"RMSE_output_{i}"] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))

        metrics[f"MAE_output_{i}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])

        metrics[f"R²_output_{i}"] = r2_score(y_true[:, i], y_pred[:, i])

        metrics[f"MSE_output_{i}"] = mean_squared_error(y_true[:, i], y_pred[:, i])

        metrics[f"MAPE_output_{i}"] = 100 * np.mean(
            np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])
        )

    # Calculate averages across all outputs
    metrics["MASE_avg"] = np.mean([metrics[f"MASE_output_{i}"] for i in range(y_true.shape[1])])
    metrics["sMAPE_avg"] = np.mean([metrics[f"sMAPE_output_{i}"] for i in range(y_true.shape[1])])
    metrics["RMSE_avg"] = np.mean([metrics[f"RMSE_output_{i}"] for i in range(y_true.shape[1])])
    metrics["MAE_avg"] = np.mean([metrics[f"MAE_output_{i}"] for i in range(y_true.shape[1])])
    metrics["R²_avg"] = np.mean([metrics[f"R²_output_{i}"] for i in range(y_true.shape[1])])
    metrics["MSE_avg"] = np.mean([metrics[f"MSE_output_{i}"] for i in range(y_true.shape[1])])
    metrics["MAPE_avg"] = np.mean([metrics[f"MAPE_output_{i}"] for i in range(y_true.shape[1])])

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

# Путь для сохранения результатов
results_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\meteo_res\results"
os.makedirs(results_dir, exist_ok=True)


def train_and_save_model(i, output_dir):
    # Создание и обучение модели
    model = GeneticProgrammingNeuralNetRegressorMO(
        iters=2,
        pop_size=10,
        input_block_size=1,
        optimizer_args={"show_progress_each": 1},
        weights_optimizer_args={
            "iters": 10,
            "pop_size": 10,
            "fitness_update_eps": 0.0001,
            "no_increase_num": 50,
        },
    )

    model.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Вычисление метрик
    metrics = calculate_metrics(y_test, y_pred, y_train)
    metrics["Model"] = f"model_{i}"

    # Сохранение модели
    net_file = os.path.join(output_dir, f"gpnn_model_{i}.pkl")
    net = model._optimizer.get_fittest()["phenotype"]
    plt.figure()
    print_net(net)
    plt.savefig(os.path.join(output_dir, f"net_{i}.png"))
    plt.close()

    net.save_to_file(net_file)

    # Запись метрик в файл
    metrics_df = pd.DataFrame([metrics])

    metrics_file = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        metrics_df.to_csv(
            metrics_file, mode="w", header=True, index=False
        )  # Create file with header if it doesn't exist
    else:
        metrics_df.to_csv(
            metrics_file, mode="a", header=False, index=False
        )  # Append to file without header if it exists

    return metrics


def run_multiple_experiments(n_runs, output_dir):
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(train_and_save_model, i, output_dir) for i in range(n_runs)]
        for future in futures:
            future.result()  # Wait for each to complete

    print("Все модели успешно обучены, сохранены и метрики записаны в metrics.csv")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\meteo_res\results"
    n_runs = 5  # Number of runs you want to perform
    run_multiple_experiments(n_runs, output_dir)
