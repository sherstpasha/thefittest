import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import cloudpickle
from thefittest.benchmarks import BanknoteDataset  # Здесь меняем на нужный датасет


def run_experiment(run_id, output_dir, model_type, model_params=None):
    data = BanknoteDataset()  # Используем BanknoteDataset
    X = data.get_X()
    y = data.get_y()

    X_scaled = minmax_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    if model_params is None:
        model_params = {}

    if model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=model_params.get("hidden_layer_sizes", (100,)),
            max_iter=model_params.get("max_iter", 2000),
            random_state=42,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=model_params.get("max_iter", 400), random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100), random_state=42
        )
    elif model_type == "svc":
        model = SVC(kernel=model_params.get("kernel", "rbf"), random_state=42)
    elif model_type == "adaboost":
        model = AdaBoostClassifier(
            n_estimators=model_params.get("n_estimators", 50), random_state=42
        )
    elif model_type == "voting":
        classifiers = [
            ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
            ("lr", LogisticRegression(max_iter=400, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
        model = VotingClassifier(estimators=classifiers, voting="hard")
    elif model_type == "stacking":
        base_learners = [
            ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("svc", SVC(kernel="rbf", random_state=42)),
        ]
        model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=model_params.get("n_estimators", 100), random_state=42
        )
    elif model_type == "xgboost":
        model = XGBClassifier(n_estimators=model_params.get("n_estimators", 100), random_state=42)
    else:
        raise ValueError("Invalid model type.")

    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    f1 = f1_score(y_test, predict, average="macro")

    run_dir = os.path.join(output_dir, model_type, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "f1_score.txt"), "w") as f:
        f.write(f"f1_score: {f1}\n")

    cm = confusion_matrix(y_test, predict)
    np.savetxt(os.path.join(run_dir, "confusion_matrix.txt"), cm, fmt="%d")
    np.savetxt(os.path.join(run_dir, "X_train.txt"), X_train)
    np.savetxt(os.path.join(run_dir, "X_test.txt"), X_test)
    np.savetxt(os.path.join(run_dir, "y_train.txt"), y_train, fmt="%d")
    np.savetxt(os.path.join(run_dir, "y_test.txt"), y_test, fmt="%d")

    with open(os.path.join(run_dir, "model.pkl"), "wb") as model_file:
        cloudpickle.dump(model, model_file)

    print(run_id, model_type, "done")
    return f1


def run_multiple_experiments(n_runs, n_processes, output_dir, model_type, model_params=None):
    from concurrent.futures import ProcessPoolExecutor

    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [
            executor.submit(run_experiment, i, output_dir, model_type, model_params)
            for i in range(n_runs)
        ]
        f1_scores = [future.result() for future in futures]

    avg_f1 = np.mean(f1_scores)

    with open(os.path.join(model_dir, "average_f1_score.txt"), "w") as f:
        f.write(f"Average F1 Score: {avg_f1}\n")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results_banknote"
    n_runs = 30
    n_processes = 1

    model_params_mlp = {"hidden_layer_sizes": (100, 100), "max_iter": 2000}
    model_params_lr = {"max_iter": 400}
    model_params_rf = {"n_estimators": 100}
    model_params_svc = {"kernel": "rbf"}
    model_params_adaboost = {"n_estimators": 50}
    model_params_gradient_boosting = {"n_estimators": 100}
    model_params_xgboost = {"n_estimators": 100}

    run_multiple_experiments(n_runs, n_processes, output_dir, "mlp", model_params_mlp)
    run_multiple_experiments(
        n_runs, n_processes, output_dir, "logistic_regression", model_params_lr
    )
    run_multiple_experiments(n_runs, n_processes, output_dir, "random_forest", model_params_rf)
    run_multiple_experiments(n_runs, n_processes, output_dir, "svc", model_params_svc)
    run_multiple_experiments(n_runs, n_processes, output_dir, "adaboost", model_params_adaboost)
    run_multiple_experiments(n_runs, n_processes, output_dir, "voting")
    run_multiple_experiments(n_runs, n_processes, output_dir, "stacking")
    run_multiple_experiments(
        n_runs, n_processes, output_dir, "gradient_boosting", model_params_gradient_boosting
    )
    run_multiple_experiments(n_runs, n_processes, output_dir, "xgboost", model_params_xgboost)
