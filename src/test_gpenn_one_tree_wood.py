import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cloudpickle
import warnings

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.model_selection import train_test_split

from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SHADE
from thefittest.regressors._gpnneregression_one_tree_mo import GeneticProgrammingNeuralNetStackingRegressorMO
from thefittest.tools.print import print_tree, print_ens

warnings.filterwarnings("ignore")

def run_experiment(run_id, output_dir, csv_path, n_outputs=1):
    # Загрузка и проверка данных
    df = pd.read_csv(csv_path)

    if df.shape[0] < 20:
        raise ValueError("Слишком мало строк в датасете. Нужно минимум 20.")

    if df.isnull().any().any():
        raise ValueError("В датасете есть пропущенные значения (NaN).")

    # Разделение на признаки и целевые переменные
    X_raw = df.iloc[:, :-n_outputs].values
    y_raw = df.iloc[:, -n_outputs:].values

    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)

    if np.any(np.isnan(y_raw)) or y_raw.shape[0] == 0:
        raise ValueError("Целевая переменная содержит NaN или пуста.")

    # Масштабирование
    X = minmax_scale(X_raw)
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y_raw)

    # Случайное разбиение
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=42
    )

    if X_train.shape[0] < 10:
        raise ValueError(f"Недостаточно данных для обучения: только {X_train.shape[0]} тренировочных строк.")

    # Обучение модели
    model = GeneticProgrammingNeuralNetStackingRegressorMO(
        iters=30,
        pop_size=10,
        input_block_size=1,
        optimizer=PDPSHAGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 10, "no_increase_num": 5},
        weights_optimizer=SHADE,
        weights_optimizer_args={"iters": 1500, "pop_size": 100, "no_increase_num": 100, "fitness_update_eps": 0.01},
        test_sample_ratio=0.1,
    )

    model.fit(X, y)

    # Предсказания
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train))
    y_test_pred = scaler_y.inverse_transform(model.predict(X_test))

    y_all_true = np.vstack([y_train_raw, y_test_raw])
    y_all_pred = np.vstack([y_train_pred, y_test_pred])

    # Метрики
    r2 = r2_score(y_test_raw, y_test_pred)
    rmse_train = np.sqrt(np.mean((y_train_raw - y_train_pred) ** 2))
    rmse_test = np.sqrt(np.mean((y_test_raw - y_test_pred) ** 2))
    rmse_full = np.sqrt(np.mean((y_all_true - y_all_pred) ** 2))

    # Сохранение результатов
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"R2: {r2:.6f}\nRMSE_train: {rmse_train:.6f}\nRMSE_test: {rmse_test:.6f}\nRMSE_full: {rmse_full:.6f}\n")

    for j in range(y_all_true.shape[1]):
        plt.figure(figsize=(14, 5))
        plt.plot(y_all_true[:, j], label="True", linewidth=2)
        plt.plot(y_all_pred[:, j], label="Pred", linestyle='--')
        plt.axvline(len(y_train_pred), linestyle=':', color='gray')
        plt.title(f"Output {j+1} — Predictions")
        plt.xlabel("Index")
        plt.ylabel(f"Output_{j+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"curve_output_{j+1}.png"))
        plt.close()

    # Сохранение структуры модели
    optimizer = model.get_optimizer()
    fittest = optimizer.get_fittest()

    print_tree(fittest['genotype'])
    plt.savefig(os.path.join(run_dir, "common_tree.png"))
    plt.close()

    print_ens(fittest['phenotype'])
    plt.savefig(os.path.join(run_dir, "ensemble.png"))
    plt.close()

    cloudpickle.dump(fittest['phenotype'], open(os.path.join(run_dir, "ens.pkl"), "wb"))

    # Экспорт используемых признаков
    used_inputs = {i for net in fittest['phenotype']._nets for i in net._inputs}
    bias_idx = X_raw.shape[1]
    used_inputs.discard(bias_idx)
    sel = sorted(used_inputs)
    cols = df.columns[:X_raw.shape[1]]
    df_inputs = pd.DataFrame(X_raw[:, sel], columns=[cols[i] for i in sel])
    for j in range(y_all_pred.shape[1]):
        df_inputs[f"output_{j+1}"] = y_all_pred[:, j]
    df_inputs.to_csv(os.path.join(run_dir, "selected_inputs_and_preds.csv"), index=False)

    print(f"✅ Run {run_id} done | R2={r2:.4f} | RMSE_test={rmse_test:.4f}")
    return r2

def run_multiple_experiments(n_runs, output_dir, csv_path, n_outputs=1):
    scores = []
    for run_id in range(n_runs):
        print(f"\n🔁 Запуск {run_id + 1} из {n_runs}")
        try:
            r2 = run_experiment(run_id=run_id, output_dir=output_dir, csv_path=csv_path, n_outputs=n_outputs)
            scores.append(r2)
        except Exception as e:
            print(f"❌ Ошибка в запуске {run_id}: {e}")

    # Сохраняем средний R2
    if scores:
        avg_r2 = np.mean(scores)
        with open(os.path.join(output_dir, "average_r2.txt"), "w") as f:
            f.write(f"Average R2 across {len(scores)} runs: {avg_r2:.6f}\n")
        print(f"\n✅ Все запуски завершены | Средний R2: {avg_r2:.4f}")
    else:
        print("\n⚠️ Ни один запуск не завершился успешно.")

# Пример запуска
if __name__ == "__main__":
    csv_file_path = r"src\thefittest\benchmarks\cleaned_data.csv"
    output_path = "results_custom_dataset"
    os.makedirs(output_path, exist_ok=True)

    run_multiple_experiments(n_runs=20, output_dir=output_path, csv_path=csv_file_path, n_outputs=1)