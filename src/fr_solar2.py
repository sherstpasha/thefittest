import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from thefittest.fuzzy import FuzzyRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Root directory containing run_0, run_1, ..., run_19 subfolders
ROOT = r"C:\Users\pasha\OneDrive\Рабочий стол\solar_res\ANN"


def calculate_global_normalized_error(y_true, y_pred, y_min_global, y_max_global):
    s, m = y_true.shape
    total_error = 0.0
    for j in range(m):
        denom = y_max_global[j] - y_min_global[j] or 1.0
        total_error += np.sum(np.abs(y_true[:, j] - y_pred[:, j])) / denom
    return (100.0 / s) * (1.0 / m) * total_error


def run_fuzzy_on_selected(base_dir, input_csv, full_pred_txt, full_true_txt, test_pred_txt):
    os.makedirs(base_dir, exist_ok=True)

    # Load features and targets
    df = pd.read_csv(input_csv)
    n_targets = 4
    feature_names = list(df.columns[:-n_targets])
    target_names = [f"out_{i}" for i in range(n_targets)]
    X_raw = df.iloc[:, :-n_targets].values

    # Load predictions and ground truth
    y_net_all = np.loadtxt(full_pred_txt)
    y_true_all = np.loadtxt(full_true_txt)
    y_net_test = np.loadtxt(test_pred_txt)

    # Split into train/test
    n_train = 169
    X_test = X_raw[n_train:]
    y_true_test = y_true_all[n_train:]
    y_net_all_tail = y_net_all[n_train:]

    # Adjust tail if size mismatch
    if y_net_test.shape != y_net_all_tail.shape:
        y_net_test = y_net_all_tail[-y_net_test.shape[0] :]
        y_true_test = y_true_test[-y_net_test.shape[0] :]

    # Scale features and network predictions
    scaler_X = MinMaxScaler().fit(X_raw)
    scaler_y = MinMaxScaler().fit(y_net_all)
    X_scaled = scaler_X.transform(X_raw)
    y_net_scaled = scaler_y.transform(y_net_all)

    # Initialize and train FuzzyRegressor
    model = FuzzyRegressor(
        iters=3000,
        pop_size=200,
        n_features_fuzzy_sets=[5] * X_scaled.shape[1],
        n_target_fuzzy_sets=[7] * y_net_scaled.shape[1],
        max_rules_in_base=20,
        target_grid_volume=100,
    )
    model.define_sets(
        X_scaled, y_net_scaled, feature_names=feature_names, target_names=target_names
    )
    start = time.time()
    model.fit(X_scaled, y_net_scaled)
    train_time = time.time() - start

    # Predict fuzzy outputs on test set
    X_test_scaled = scaler_X.transform(X_test)
    y_fuzzy_test = scaler_y.inverse_transform(model.predict(X_test_scaled))

    # Compute Global Normalized Errors
    gne_vs_net = calculate_global_normalized_error(
        y_net_test, y_fuzzy_test, y_true_all.min(axis=0), y_true_all.max(axis=0)
    )
    gne_vs_true = calculate_global_normalized_error(
        y_true_test, y_fuzzy_test, y_true_all.min(axis=0), y_true_all.max(axis=0)
    )
    gne_net_true = calculate_global_normalized_error(
        y_true_test, y_net_test, y_true_all.min(axis=0), y_true_all.max(axis=0)
    )

    # Save model and fuzzy rules
    with open(os.path.join(base_dir, "fuzzy_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(base_dir, "fuzzy_rules.txt"), "w", encoding="utf-8") as f:
        f.write(model.get_text_rules(print_intervals=False))

    # Save comparison CSV
    cols = target_names
    df_out = pd.DataFrame(
        np.hstack([y_true_test, y_net_test, y_fuzzy_test]),
        columns=[*cols, *[c + "_net" for c in cols], *[c + "_fuzzy" for c in cols]],
    )
    df_out.to_csv(os.path.join(base_dir, "comparisons_test.csv"), index=False)

    # Plot all samples (train+test) curves
    y_fuzzy_all = scaler_y.inverse_transform(model.predict(scaler_X.transform(X_raw)))
    for j, name in enumerate(target_names):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true_all[:, j], label="True", linewidth=2)
        plt.plot(y_net_all[:, j], label="Net_pred", linestyle="--")
        plt.plot(y_fuzzy_all[:, j], label="Fuzzy_pred", linestyle=":")
        plt.axvline(n_train - 0.5, color="gray", linestyle=":", label="Train/Test split")
        plt.title(f"Output {name}: all data (train vs test)")
        plt.xlabel("Sample index")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"curve_all_{j}.png"))
        plt.close()

    # Save metrics JSON
    metrics = {
        "GNE_vs_net": float(gne_vs_net),
        "GNE_vs_true": float(gne_vs_true),
        "GNE_net_vs_true": float(gne_net_true),
        "Train_time_sec": train_time,
    }
    with open(os.path.join(base_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(
        f"Done on test: run in {train_time:.1f}s — GNE fuzzy vs net: {gne_vs_net:.4f},"
        f" fuzzy vs true: {gne_vs_true:.4f}, net vs true: {gne_net_true:.4f}"
    )


def worker(run_idx: int, trial_idx: int):
    run_name = f"run_{run_idx}"
    base_dir = os.path.join("results_fuzzy_selected", run_name, f"trial_{trial_idx}")
    input_csv = os.path.join(ROOT, run_name, "selected_inputs_and_preds.csv")
    full_pred = os.path.join(ROOT, run_name, "full_pred.txt")
    full_true = os.path.join(ROOT, run_name, "full_true.txt")
    test_pred = os.path.join(ROOT, run_name, "test_pred.txt")
    run_fuzzy_on_selected(base_dir, input_csv, full_pred, full_true, test_pred)
    return run_idx, trial_idx


def main():
    num_runs = 20
    trials_per_run = 5
    max_workers = os.cpu_count() or 4

    tasks = [(i, j) for i in range(num_runs) for j in range(trials_per_run)]

    # Launch tasks in parallel and show progress
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker, run_i, trial_i): (run_i, trial_i) for run_i, trial_i in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            run_i, trial_i = futures[future]
            try:
                future.result()
                tqdm.write(f"[OK]   run_{run_i} trial_{trial_i}")
            except Exception as exc:
                tqdm.write(f"[FAIL] run_{run_i} trial_{trial_i} → {exc!r}")


if __name__ == "__main__":
    main()
