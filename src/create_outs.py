import os
import cloudpickle
import pandas as pd
import numpy as np
import torch
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def predict(X, model, offset=True):
    if offset:
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, device=device, dtype=torch.float32)
    output = model.meta_output(X)
    return output.cpu().numpy()

def calculate_metrics(y_true, y_pred, y_train):
    metrics = {}
    naive = np.roll(y_train, 1, axis=0)[1:]
    mae_naive = np.mean(np.abs(y_train[1:] - naive))
    metrics["MASE"] = np.mean(np.abs(y_true - y_pred)) / mae_naive
    metrics["sMAPE"] = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["R2"] = r2_score(y_true, y_pred)
    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["MAPE"] = 100 * np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None)))
    return metrics

# Путь к корневой папке, где лежат все run_0, run_1, ..., run_20
root_dir = r"C:\Users\USER\Desktop\Расчеты по нейросетям\расчеты сетей метео\GPENN"

# Пути к неизменным данным
X_test_path = "src\\lookback_1h\\X_test.csv"
X_train_path = "src\\lookback_1h\\X_train.csv"
y_test_path = "src\\lookback_1h\\y_test.csv"
y_train_path = "src\\lookback_1h\\y_train.csv"

# Загрузка неизменных данных один раз
X_test = pd.read_csv(X_test_path, index_col=0)
X_train = pd.read_csv(X_train_path, index_col=0)
y_test = pd.read_csv(y_test_path)
y_train = pd.read_csv(y_train_path)

all_input_cols = X_test.columns.tolist()
all_output_cols = y_test.columns.tolist()

scaler_X = StandardScaler().fit(X_train.values)
scaler_y = StandardScaler().fit(y_train)

# Бежим по run_0 ... run_20
for run_number in range(21):
    run_dir = os.path.join(root_dir, f"run_{run_number}")
    ens_path = os.path.join(run_dir, "best_ens.pkl")
    y_test_pred_NN_path = os.path.join(run_dir, "y_pred.txt")
    
    if not os.path.exists(ens_path):
        print(f"Пропускаем {run_dir}, файл best_ens.pkl не найден.")
        continue
    
    print(f"Обрабатываем {run_dir}")

    # Загрузка ансамбля
    with open(ens_path, "rb") as f:
        ensemble = cloudpickle.load(f)

    # Собираем все входы
    all_inputs = set()
    for net in ensemble._nets:
        all_inputs.update(net._inputs)

    # Исключаем 8 и сортируем
    final_inputs = sorted(x for x in all_inputs if x != 8)

    input_feature_cols = all_input_cols[:]  # убираем колонку времени
    print(input_feature_cols, final_inputs)
    selected_input_cols = [input_feature_cols[i] for i in final_inputs]
    used_cols = selected_input_cols

    # Загрузка предсказаний НС
    y_test_pred_NN = pd.read_csv(y_test_pred_NN_path, header=None, delim_whitespace=True)
    y_test_pred_NN.columns = all_output_cols

    # Подготовка признаков
    X_train_NN = X_train[selected_input_cols]
    X_test_NN = X_test[selected_input_cols]

    # Предсказания на обучении
    X_train_s = scaler_X.transform(X_train)
    y_train_NN_s = predict(X_train_s, ensemble)
    y_train_NN = scaler_y.inverse_transform(y_train_NN_s)

    # Сохраняем результаты
    X_test_NN.to_csv(os.path.join(run_dir, "X_test_NN.csv"), index=False)
    X_train_NN.to_csv(os.path.join(run_dir, "X_train_NN.csv"), index=False)
    pd.DataFrame(y_train_NN, columns=all_output_cols).to_csv(os.path.join(run_dir, "y_train_NN.csv"), index=False)
    y_test_pred_NN.to_csv(os.path.join(run_dir, "y_test_NN.csv"), index=False)

    # Вычисляем метрики
    metrics = calculate_metrics(y_train.values, y_train_NN, y_train.values)

    # Сохраняем метрики
    metrics_path = os.path.join(run_dir, "metrics_train.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ {run_dir} успешно обработан.\n")

        # === Вычисляем метрики на тесте ===
    metrics_test = {}

    for i, target_name in enumerate(all_output_cols):
        y_true_i = y_test[target_name].values
        y_pred_i = y_test_pred_NN[target_name].values

        metrics_test[target_name] = calculate_metrics(
            y_true=y_true_i,
            y_pred=y_pred_i,
            y_train=y_train[target_name].values
        )

    # Сохраняем метрики по тесту
    metrics_test_path = os.path.join(run_dir, "metrics_test.json")
    with open(metrics_test_path, "w") as f:
        json.dump(metrics_test, f, indent=4)

    print(f"✅ Метрики на тесте сохранены для {run_dir}.\n")

print("Все расчеты завершены!")
