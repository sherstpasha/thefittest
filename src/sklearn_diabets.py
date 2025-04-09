import os
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor


def run_experiment(run_id, output_dir, model_type, model_params=None):
    data = load_diabetes()
    X = data.data
    y = data.target

    X_scaled = minmax_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    if model_params is None:
        model_params = {}

    # Выбор модели
    if model_type == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=model_params.get("hidden_layer_sizes", (100,)),
            max_iter=model_params.get("max_iter", 2000),
            random_state=42,
        )
    elif model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=model_params.get("n_estimators", 100),
            random_state=42,
        )
    elif model_type == "svr":
        model = SVR(kernel=model_params.get("kernel", "rbf"))
    elif model_type == "adaboost":
        model = AdaBoostRegressor(
            n_estimators=model_params.get("n_estimators", 50),
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=model_params.get("n_estimators", 100),
            random_state=42,
        )
    elif model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=model_params.get("n_estimators", 100),
            random_state=42,
        )
    elif model_type == "voting":
        regressors = [
            ("mlp", MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
        model = VotingRegressor(estimators=regressors)
    elif model_type == "stacking":
        base_learners = [
            ("mlp", MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("svr", SVR(kernel="rbf")),
        ]
        model = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
    else:
        raise ValueError("Invalid model type.")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    run_dir = os.path.join(output_dir, model_type, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "r2_score.txt"), "w") as f:
        f.write(f"R2 Score: {r2}\n")

    np.savetxt(os.path.join(run_dir, "X_train.txt"), X_train)
    np.savetxt(os.path.join(run_dir, "X_test.txt"), X_test)
    np.savetxt(os.path.join(run_dir, "y_train.txt"), y_train)
    np.savetxt(os.path.join(run_dir, "y_test.txt"), y_test)
    np.savetxt(os.path.join(run_dir, "predictions.txt"), predictions)

    with open(os.path.join(run_dir, "model.pkl"), "wb") as f_model:
        cloudpickle.dump(model, f_model)

    print(run_id, model_type, "done")
    return r2


def run_multiple_experiments(n_runs, n_processes, output_dir, model_type, model_params=None):
    from concurrent.futures import ProcessPoolExecutor

    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [
            executor.submit(run_experiment, i, output_dir, model_type, model_params)
            for i in range(n_runs)
        ]
        r2_scores = [future.result() for future in futures]

    avg_r2 = np.mean(r2_scores)

    with open(os.path.join(model_dir, "average_r2_score.txt"), "w") as f:
        f.write(f"Average R2 Score: {avg_r2}\n")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results_diabetes"
    n_runs = 30
    n_processes = 1

    model_params_mlp = {"hidden_layer_sizes": (100, 100), "max_iter": 2000}
    model_params_rf = {"n_estimators": 100}
    model_params_svr = {"kernel": "rbf"}
    model_params_adaboost = {"n_estimators": 50}
    model_params_gb = {"n_estimators": 100}
    model_params_xgb = {"n_estimators": 100}

    run_multiple_experiments(n_runs, n_processes, output_dir, "mlp", model_params_mlp)
    run_multiple_experiments(n_runs, n_processes, output_dir, "linear_regression")
    run_multiple_experiments(n_runs, n_processes, output_dir, "random_forest", model_params_rf)
    run_multiple_experiments(n_runs, n_processes, output_dir, "svr", model_params_svr)
    run_multiple_experiments(n_runs, n_processes, output_dir, "adaboost", model_params_adaboost)
    run_multiple_experiments(n_runs, n_processes, output_dir, "gradient_boosting", model_params_gb)
    run_multiple_experiments(n_runs, n_processes, output_dir, "xgboost", model_params_xgb)
    run_multiple_experiments(n_runs, n_processes, output_dir, "voting")
    run_multiple_experiments(n_runs, n_processes, output_dir, "stacking")
