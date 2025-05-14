import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from thefittest.regressors._gpnneregression_one_tree_mo import (
    GeneticProgrammingNeuralNetStackingRegressorMO,
)
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SHADE
from thefittest.tools.print import print_tree, print_trees, print_net, print_ens, print_nets
import cloudpickle

# === Параметры ===
excel_path = r"C:\Users\USER\Desktop\Для тезисов\Программа 2\data.xlsx"
sheet_name = "Sheet1"
output_dir = r"output_runs"
n_runs = 1
start_run = 0

def load_excel_data(path, sheet):
    """Load dataset from Excel and return feature matrix X and target vector y."""
    df = pd.read_excel(path, sheet_name=sheet)
    # Drop first ID column
    df = df.drop(columns=[df.columns[0]])
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    return X, y

# === Функция для вычисления метрик ===
def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    metrics = {}
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["R2"] = r2_score(y_true, y_pred)
    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["MAPE"] = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    # Naive forecast metric (MASE)
    naive = np.roll(y_true, 1, axis=0)[1:]
    mae_naive = np.mean(np.abs(y_true[1:] - naive))
    metrics["MASE"] = np.mean(np.abs(y_true - y_pred)) / mae_naive
    metrics["sMAPE"] = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )
    return metrics

# === Один запуск ===
def run_experiment(run_id, X_train, X_test, y_train, y_test, output_dir):
    # Масштабирование
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    X_train_s, X_test_s = scaler_X.transform(X_train), scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train)

    # Обучение модели
    model = GeneticProgrammingNeuralNetStackingRegressorMO(
        iters=100, pop_size=20, input_block_size=1,
        optimizer=PDPSHAGP,
        optimizer_args={"show_progress_each":1, "keep_history":True, "n_jobs": 15, "no_increase_num":100},
        weights_optimizer=SHADE,
        weights_optimizer_args={"iters":100, "pop_size":100, "no_increase_num":100, "fitness_update_eps":1e-1},
        test_sample_ratio=0.1,
    )
    model.fit(X_train_s, y_train_s)

    # Предсказание
    y_pred_train = scaler_y.inverse_transform(model.predict(X_train_s))
    y_pred_test  = scaler_y.inverse_transform(model.predict(X_test_s))

    # Метрики train и test
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_test  = calculate_metrics(y_test,  y_pred_test)

    # Общие метрики на объединенной выборке
    y_all_true = np.vstack([y_train, y_test])
    y_all_pred = np.vstack([y_pred_train, y_pred_test])
    metrics_all = calculate_metrics(y_all_true, y_all_pred)

    # Сохранение
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics_train.json"), "w") as f:
        json.dump(metrics_train, f, indent=2)
    with open(os.path.join(run_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics_test, f, indent=2)
    with open(os.path.join(run_dir, "metrics_all.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    # Остальные сохранения
    np.savetxt(os.path.join(run_dir, "y_pred_train.txt"), y_pred_train)
    np.savetxt(os.path.join(run_dir, "y_pred_test.txt"),  y_pred_test)
    optimizer = model.get_optimizer()
    stats = optimizer.get_stats(); stats["population_ph"]=None
    with open(os.path.join(run_dir, "stats.pkl"), "wb") as f:
        cloudpickle.dump(stats, f)
    best = optimizer.get_fittest(); tree=best["genotype"]; ens=best["phenotype"]
    tree.save_to_file(os.path.join(run_dir, "best_tree.pkl"))
    ens.save_to_file   (os.path.join(run_dir, "best_ens.pkl"))
    print_tree(tree); plt.savefig(os.path.join(run_dir, "1_tree.png")); plt.close()
    print_trees(ens._trees); plt.savefig(os.path.join(run_dir, "2_trees.png")); plt.close()
    print_nets(ens._nets); plt.savefig(os.path.join(run_dir, "3_nets.png")); plt.close()
    print_net(ens._meta_algorithm); plt.savefig(os.path.join(run_dir, "4_meta_net.png")); plt.close()
    print_ens(ens); plt.savefig(os.path.join(run_dir, "5_ens.png")); plt.close()

    print(f"Run {run_id} done. Train RMSE={metrics_train['RMSE']:.4f}, Test RMSE={metrics_test['RMSE']:.4f}, All RMSE={metrics_all['RMSE']:.4f}")
    return metrics_train, metrics_test, metrics_all

# === Множественные прогоны ===
def run_multiple_experiments(n_runs, output_dir, excel_path, sheet_name, start_run=0):
    X, y = load_excel_data(excel_path, sheet_name)
    os.makedirs(output_dir, exist_ok=True)
    all_tr, all_te, all_all = [], [], []
    for i in range(start_run, start_run+n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42+i)
        mt, mte, mta = run_experiment(i, X_tr, X_te, y_tr, y_te, output_dir)
        all_tr.append(mt); all_te.append(mte); all_all.append(mta)
    def avg(dicts): return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}
    avg_tr = avg(all_tr); avg_te = avg(all_te); avg_all = avg(all_all)
    with open(os.path.join(output_dir, "avg_train.json"), "w") as f: json.dump(avg_tr, f, indent=2)
    with open(os.path.join(output_dir, "avg_test.json"),  "w") as f: json.dump(avg_te, f, indent=2)
    with open(os.path.join(output_dir, "avg_all.json"),   "w") as f: json.dump(avg_all, f, indent=2)
    print(f"All done. Avg Train RMSE={avg_tr['RMSE']:.4f}, Avg Test RMSE={avg_te['RMSE']:.4f}, Avg All RMSE={avg_all['RMSE']:.4f}")

if __name__ == "__main__":
    run_multiple_experiments(n_runs, output_dir, excel_path, sheet_name, start_run)
