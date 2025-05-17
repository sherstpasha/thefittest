import os
import numpy as np
import pandas as pd

# Force non-GUI backend to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cloudpickle
import warnings
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, minmax_scale

from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SHADE
from thefittest.regressors._gpnneregression_one_tree_mo import GeneticProgrammingNeuralNetStackingRegressorMO
from thefittest.benchmarks import SolarBatteryDegradationDataset
from thefittest.tools.print import print_tree, print_ens

warnings.filterwarnings("ignore")

def calculate_global_normalized_error(y_true, y_pred, y_min_global, y_max_global):
    """
    Global normalized error with fixed global ranges:
      (100/s)*(1/m) * sum_i sum_j |y_true[i,j] - y_pred[i,j]| / (y_max_global[j] - y_min_global[j])
    """
    s, m = y_true.shape
    total_error = 0.0
    for j in range(m):
        denom = (y_max_global[j] - y_min_global[j]) or 1.0
        total_error += np.sum(np.abs(y_true[:, j] - y_pred[:, j])) / denom
    return (100.0 / s) * (1.0 / m) * total_error


def run_experiment(run_id, output_dir):
    # === Data ===
    dataset = SolarBatteryDegradationDataset()
    X_raw = dataset.get_X()
    X = minmax_scale(X_raw)
    y_raw = dataset.get_y()

    # Scale y to [0,1]
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y_raw)

    # Global min/max for GNE
    y_min_global = y_raw.min(axis=0)
    y_max_global = y_raw.max(axis=0)

    # Train/test split
    n_train = 169
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    y_train_raw, y_test_raw = y_raw[:n_train], y_raw[n_train:]

    # === Model ===
    model = GeneticProgrammingNeuralNetStackingRegressorMO(
        iters=20,
        pop_size=10,
        input_block_size=1,
        optimizer=PDPSHAGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 10, "no_increase_num": 5},
        weights_optimizer=SHADE,
        weights_optimizer_args={"iters": 1500, "pop_size": 100, "no_increase_num": 100, "fitness_update_eps": 0.01},
        test_sample_ratio=0.33,
    )

    # Fit on full scaled data (stacking uses internal CV)
    model.fit(X, y)

    # Predict and inverse-scale
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train))
    y_test_pred  = scaler_y.inverse_transform(model.predict(X_test))

    # Combine
    y_all_true = np.vstack([y_train_raw, y_test_raw])
    y_all_pred = np.vstack([y_train_pred, y_test_pred])

    # Metrics
    r2 = r2_score(y_test_raw, y_test_pred)
    gne_train = calculate_global_normalized_error(y_train_raw, y_train_pred, y_min_global, y_max_global)
    gne_test  = calculate_global_normalized_error(y_test_raw,  y_test_pred,  y_min_global, y_max_global)
    gne_full  = calculate_global_normalized_error(y_all_true,  y_all_pred,  y_min_global, y_max_global)

    # Output directory
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"R2: {r2:.6f}\nGNE_train: {gne_train:.6f}\nGNE_test: {gne_test:.6f}\nGNE_full: {gne_full:.6f}\n")

    # Save predictions
    np.savetxt(os.path.join(run_dir, "train_pred.txt"), y_train_pred)
    np.savetxt(os.path.join(run_dir, "test_pred.txt"),  y_test_pred)
    np.savetxt(os.path.join(run_dir, "full_true.txt"), y_all_true)
    np.savetxt(os.path.join(run_dir, "full_pred.txt"), y_all_pred)

    # Plot true vs pred curves
    split_idx = len(y_train_pred)
    for j in range(y_all_true.shape[1]):
        plt.figure(figsize=(14, 5))
        plt.plot(y_all_true[:, j], label="True", linewidth=2)
        plt.plot(y_all_pred[:, j], label="Pred", linestyle='--')
        plt.axvline(split_idx, linestyle=':', color='gray')
        plt.title(f"Output {j+1} — Predictions")
        plt.xlabel("Index")
        plt.ylabel(f"Output_{j+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"curve_output_{j+1}.png"))
        plt.close()

    # Save model structure and ensemble
    optimizer = model.get_optimizer()
    fittest = optimizer.get_fittest()
    print_tree(fittest['genotype'])
    plt.savefig(os.path.join(run_dir, "common_tree.png"))
    plt.close()
    print_ens(fittest['phenotype'])
    plt.savefig(os.path.join(run_dir, "ensemble.png"))
    plt.close()
    cloudpickle.dump(fittest['phenotype'], open(os.path.join(run_dir, "ens.pkl"), "wb"))

    # Export selected inputs & preds
    used_inputs = {i for net in fittest['phenotype']._nets for i in net._inputs}
    bias_idx = X_raw.shape[1]
    used_inputs.discard(bias_idx)
    sel = sorted(used_inputs)
    x_names = dataset.get_X_names()
    cols = [x_names[i] for i in sel]
    df = pd.DataFrame(X_raw[:, sel], columns=cols)
    for j, nm in enumerate(dataset.get_y_names()):
        df[nm] = y_all_pred[:, j]
    df.to_csv(os.path.join(run_dir, "selected_inputs_and_preds.csv"), index=False)

    print(f"✅ Run {run_id} done | R2={r2:.4f} | GNE_test={gne_test:.4f}")
    return r2


def run_multiple_experiments(n_runs, output_dir):
    scores = []
    for i in range(n_runs):
        attempt = 0
        while True:
            attempt += 1
            try:
                print(f"=== Run {i}, attempt {attempt} ===")
                r2 = run_experiment(i, output_dir)
                scores.append(r2)
                break
            except Exception as err:
                print(f"Run {i} failed on attempt {attempt}: {err!r}. Retrying…")
    avg = np.mean(scores)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "avg_r2.txt"), "w") as f:
        f.write(f"Average R2: {avg:.6f}\n")
    print(f"✅ All runs done | avg R2={avg:.4f}")


if __name__ == "__main__":
    run_multiple_experiments(n_runs=20, output_dir="results_regression_combined")
