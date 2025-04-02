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
from xgboost import XGBClassifier  # Для XGBoost
import numpy as np
import cloudpickle
from thefittest.benchmarks import UserKnowladgeDataset  # Изменено на UserKnowledgeDataset


def run_experiment(run_id, output_dir, model_type, model_params=None):
    data = UserKnowladgeDataset()  # Используем UserKnowledgeDataset
    X = data.get_X()  # Получаем данные признаков
    y = data.get_y()  # Получаем метки классов

    X_scaled = minmax_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)

    # Choose the model based on model_type
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
        # Example of ensemble model: VotingClassifier
        classifiers = [
            ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
            ("lr", LogisticRegression(max_iter=400, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
        model = VotingClassifier(estimators=classifiers, voting="hard")
    elif model_type == "stacking":
        # Example of Stacking Classifier
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
        raise ValueError(
            "Invalid model type. Choose from 'mlp', 'logistic_regression', 'random_forest', 'svc', 'adaboost', 'voting', 'stacking', 'gradient_boosting', 'xgboost'."
        )

    model.fit(X_train, y_train)

    predict = model.predict(X_test)

    # Calculate F1 score
    f1 = f1_score(y_test, predict, average="macro")

    # Create a specific directory for each model type
    run_dir = os.path.join(output_dir, model_type, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "f1_score.txt"), "w") as f:
        f.write(f"f1_score: {f1}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, predict)
    np.savetxt(os.path.join(run_dir, "confusion_matrix.txt"), cm, fmt="%d")

    # Save train and test datasets
    np.savetxt(os.path.join(run_dir, "X_train.txt"), X_train)
    np.savetxt(os.path.join(run_dir, "X_test.txt"), X_test)
    np.savetxt(os.path.join(run_dir, "y_train.txt"), y_train, fmt="%d")
    np.savetxt(os.path.join(run_dir, "y_test.txt"), y_test, fmt="%d")

    # Save the model to a file
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
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results2\mlp_lr_rf_svc_adaboost_voting_stacking_gradient_xgboost_userknowledge"
    n_runs = 30  # Number of runs you want to perform
    n_processes = 1  # Number of processes to use in parallel

    # Define model parameters for different models
    model_params_mlp = {"hidden_layer_sizes": (100, 100), "max_iter": 2000}
    model_params_lr = {"max_iter": 400}
    model_params_rf = {"n_estimators": 100}
    model_params_svc = {"kernel": "rbf"}
    model_params_adaboost = {"n_estimators": 50}
    model_params_gradient_boosting = {"n_estimators": 100}
    model_params_xgboost = {"n_estimators": 100}

    # Run for MLP model
    model_type_mlp = "mlp"
    run_multiple_experiments(n_runs, n_processes, output_dir, model_type_mlp, model_params_mlp)

    # Run for Logistic Regression model
    model_type_lr = "logistic_regression"
    run_multiple_experiments(n_runs, n_processes, output_dir, model_type_lr, model_params_lr)

    # Run for Random Forest model
    model_type_rf = "random_forest"
    run_multiple_experiments(n_runs, n_processes, output_dir, model_type_rf, model_params_rf)

    # Run for Support Vector Classifier model
    model_type_svc = "svc"
    run_multiple_experiments(n_runs, n_processes, output_dir, model_type_svc, model_params_svc)

    # Run for AdaBoost model
    model_type_adaboost = "adaboost"
    run_multiple_experiments(
        n_runs, n_processes, output_dir, model_type_adaboost, model_params_adaboost
    )

    # Run for VotingClassifier model (ensemble)
    model_type_voting = "voting"
    run_multiple_experiments(n_runs, n_processes, output_dir, model_type_voting)

    # Run for Stacking Classifier model
    model_type_stacking = "stacking"
    run_multiple_experiments(n_runs, n_processes, output_dir, model_type_stacking)

    # Run for Gradient Boosting model
    model_type_gradient_boosting = "gradient_boosting"
    run_multiple_experiments(
        n_runs,
        n_processes,
        output_dir,
        model_type_gradient_boosting,
        model_params_gradient_boosting,
    )

    # Run for XGBoost model
    model_type_xgboost = "xgboost"
    run_multiple_experiments(
        n_runs, n_processes, output_dir, model_type_xgboost, model_params_xgboost
    )
