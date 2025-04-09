import os
import numpy as np
import cloudpickle
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from itertools import product


def run_svr_manual_grid(run_id, output_dir, param_grid=None):
    data = load_diabetes()
    X = data.data
    y = data.target

    X_scaled = minmax_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    if param_grid is None:
        param_grid = {
            "C": [0.1, 1],
            "epsilon": [0.01, 0.1, 1],
            "gamma": ["scale", 0.01, 0.1],
        }

    best_r2 = -np.inf
    best_model = None
    best_params = None

    for C, epsilon, gamma in product(param_grid["C"], param_grid["epsilon"], param_grid["gamma"]):
        model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_params = {"C": C, "epsilon": epsilon, "gamma": gamma}

    run_dir = os.path.join(output_dir, "svr_manual_grid", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "r2_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"R2 Score: {best_r2}\n")
        f.write(f"Best Params: {best_params}\n")

    np.savetxt(os.path.join(run_dir, "X_test.txt"), X_test)
    np.savetxt(os.path.join(run_dir, "y_test.txt"), y_test)
    np.savetxt(os.path.join(run_dir, "predictions.txt"), best_model.predict(X_test))

    with open(os.path.join(run_dir, "model.pkl"), "wb") as f_model:
        cloudpickle.dump(best_model, f_model)

    print(f"Run {run_id} done | R²: {best_r2:.4f} | Best Params: {best_params}")
    return best_r2, best_params


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results_diabetes"
    n_runs = 5
    enable_svr = True

    param_grid_svr = {
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 1],
        "gamma": ["scale", 0.01, 0.1],
    }

    if enable_svr:
        all_r2 = []
        all_params = []

        for i in range(n_runs):
            r2, best_params = run_svr_manual_grid(i, output_dir, param_grid_svr)
            all_r2.append(r2)
            all_params.append(best_params)

        avg_r2 = np.mean(all_r2)
        summary_path = os.path.join(output_dir, "svr_manual_grid", "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Average R² over {n_runs} runs: {avg_r2:.4f}\n\n")
            for i, (r2_val, params) in enumerate(zip(all_r2, all_params)):
                f.write(f"Run {i} — R²: {r2_val:.4f} — Params: {params}\n")
