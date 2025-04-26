import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from thefittest.regressors._gpnneregression_one_tree_mo import GeneticProgrammingNeuralNetStackingRegressorMO
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SHADE
from thefittest.tools.print import print_tree, print_trees, print_net, print_ens, print_nets
import cloudpickle
import matplotlib
matplotlib.use('Agg')

# --- Ваша функция метрик, как раньше ---
def calculate_metrics(y_true, y_pred, y_train):
    metrics = {}
    # MASE
    naive = np.roll(y_train, 1, axis=0)[1:]
    mae_naive = np.mean(np.abs(y_train[1:] - naive))
    metrics["MASE"] = np.mean(np.abs(y_true - y_pred)) / mae_naive
    # sMAPE
    metrics["sMAPE"] = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )
    # RMSE, MAE, R2, MSE, MAPE
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["MAE"]  = mean_absolute_error(y_true, y_pred)
    metrics["R2"]   = r2_score(y_true, y_pred)
    metrics["MSE"]  = mean_squared_error(y_true, y_pred)
    metrics["MAPE"] = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    return metrics

def run_experiment(run_id, output_dir):
    # 1) Генерируем синтетические многовыходные данные
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_targets=3,
        noise=0.1,
        random_state=42  # фиксируем генерацию одних и тех же данных
    )

    # 2) split + масштабирование
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=run_id
    )
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train)
    y_test_s  = scaler_y.transform(y_test)

    # 3) создаём и обучаем модель
    model = GeneticProgrammingNeuralNetStackingRegressorMO(
        iters=10,
        pop_size=10,
        input_block_size=3,
        optimizer=PDPSHAGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 5},
        weights_optimizer=SHADE,
        weights_optimizer_args={
            "iters": 300,
            "pop_size": 100,
            "no_increase_num": 100,
            "fitness_update_eps": 1e-4,
        },
        test_sample_ratio=0.25,
    )
    model.fit(X_train_s, y_train_s)

    # 4) предсказание + обратное масштабирование
    y_pred_s = model.predict(X_test_s)
    y_pred   = scaler_y.inverse_transform(y_pred_s)

    # 5) собираем метрики
    metrics = calculate_metrics(y_test, y_pred, y_train)

    # 6) сохраняем всё в run_{run_id}
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # 6.1) параметры и метрики
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 6.2) данные
    np.savetxt(os.path.join(run_dir, "X_train.txt"), X_train)
    np.savetxt(os.path.join(run_dir, "X_test.txt"),  X_test)
    np.savetxt(os.path.join(run_dir, "y_train.txt"), y_train)
    np.savetxt(os.path.join(run_dir, "y_test.txt"),  y_test)
    np.savetxt(os.path.join(run_dir, "y_pred.txt"),  y_pred)

    # 6.3) оптимизатор и его статистика
    optimizer = model.get_optimizer()
    stats = optimizer.get_stats()
    stats["population_ph"] = None  # чтобы не тащить тяжёлый объект
    with open(os.path.join(run_dir, "stats.pkl"), "wb") as f:
        cloudpickle.dump(stats, f)

    # 6.4) сохраняем лучшую дерево-структуру и ансамбль
    best = optimizer.get_fittest()
    tree = best["genotype"]
    ens  = best["phenotype"]

    tree.save_to_file(os.path.join(run_dir, "best_tree.pkl"))
    ens.save_to_file( os.path.join(run_dir, "best_ens.pkl"))

    # 6.5) визуализации
    #  (print_* рисует в текущий plt.figure())
    print_tree(tree)
    plt.savefig(os.path.join(run_dir, "1_tree.png"))
    plt.close()

    print_trees(ens._trees)
    plt.savefig(os.path.join(run_dir, "2_trees.png"))
    plt.close()

    print_nets(ens._nets)
    plt.savefig(os.path.join(run_dir, "3_nets.png"))
    plt.close()

    print_net(ens._meta_algorithm)
    plt.savefig(os.path.join(run_dir, "4_meta_net.png"))
    plt.close()

    print_ens(ens)
    plt.savefig(os.path.join(run_dir, "5_ens.png"))
    plt.close()

    print(f"Run {run_id} finished. Metrics: {metrics}")
    return metrics

def run_multiple_experiments(n_runs, output_dir, start_run=0):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    all_metrics = []
    for i in range(start_run, start_run + n_runs):
        m = run_experiment(i, output_dir)
        all_metrics.append(m)

    # усредняем по всем запускам
    avg = {}
    keys = all_metrics[0].keys()
    for k in keys:
        avg[k] = float(np.mean([m[k] for m in all_metrics]))
    with open(os.path.join(output_dir, "average_metrics.json"), "w") as f:
        json.dump(avg, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nAll done! {n_runs} runs, elapsed {elapsed:.1f} sec.")
    print("Average metrics:", avg)

if __name__ == "__main__":
    OUTPUT_DIR = r"results_regressor"
    N_RUNS = 5
    run_multiple_experiments(N_RUNS, OUTPUT_DIR, start_run=0)
