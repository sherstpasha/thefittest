import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cloudpickle
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from concurrent.futures import ProcessPoolExecutor

from thefittest.classifiers._gpnneclassifier_one_tree import (
    GeneticProgrammingNeuralNetStackingClassifier,
)
from thefittest.optimizers import PDPGP, SelfCGA, SHADE
from thefittest.tools.print import (
    print_net,
    print_tree,
    print_nets,
    print_trees,
    print_ens,
)


def preprocess_adult_dataset():
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    df_train = pd.read_csv(url_train, header=None, names=columns, skipinitialspace=True)
    df_test = pd.read_csv(url_test, header=None, names=columns, skipinitialspace=True, skiprows=1)
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

    df_all = pd.concat([df_train, df_test], axis=0)

    df_all.replace("?", pd.NA, inplace=True)
    for col in ["workclass", "occupation", "native-country"]:
        df_all[col] = df_all[col].fillna("Unknown")

    df_all["income"] = df_all["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numeric_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    df_encoded = pd.get_dummies(df_all[categorical_features], drop_first=True).astype(int)
    df_final = pd.concat([df_all[numeric_features], df_encoded], axis=1)
    df_final["income"] = df_all["income"]

    feature_names = df_final.drop("income", axis=1).columns.tolist()

    X = df_final.drop("income", axis=1).values
    y = df_final["income"].values
    X_train = X[: len(df_train)]
    y_train = y[: len(df_train)]
    X_test = X[len(df_train) :]
    y_test = y[len(df_train) :]

    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)

    return X_train, X_test, y_train, y_test, feature_names


def run_experiment(run_id, output_dir):
    X_train, X_test, y_train, y_test, feature_names = preprocess_adult_dataset()

    model = GeneticProgrammingNeuralNetStackingClassifier(
        iters=15,
        pop_size=10,
        input_block_size=10,
        optimizer=PDPGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True},
        weights_optimizer=SHADE,
        weights_optimizer_args={
            "iters": 10000,
            "pop_size": 100,
            "no_increase_num": 50,
        },
        test_sample_ratio=0.25,
    )

    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    optimizer = model.get_optimizer()

    stat = optimizer.get_stats()
    stat["population_ph"] = None

    common_tree = optimizer.get_fittest()["genotype"]
    ens = optimizer.get_fittest()["phenotype"]
    trees = ens._trees

    # Метрики
    f1 = f1_score(y_test, predict, average="macro")
    precision = precision_score(y_test, predict, average="macro")
    recall = recall_score(y_test, predict, average="macro")

    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Сохраняем метрики
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"F1 Score (macro): {f1:.4f}\n")
        f.write(f"Precision (macro): {precision:.4f}\n")
        f.write(f"Recall (macro): {recall:.4f}\n")

    # Save confusion matrix and data
    cm = confusion_matrix(y_test, predict)
    np.savetxt(os.path.join(run_dir, "confusion_matrix.txt"), cm, fmt="%d")
    np.savetxt(os.path.join(run_dir, "X_train.txt"), X_train)
    np.savetxt(os.path.join(run_dir, "X_test.txt"), X_test)
    np.savetxt(os.path.join(run_dir, "y_train.txt"), y_train, fmt="%d")
    np.savetxt(os.path.join(run_dir, "y_test.txt"), y_test, fmt="%d")

    with open(os.path.join(run_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    # Сохраняем визуализации
    print_tree(common_tree)
    plt.savefig(os.path.join(run_dir, "1_common_tree.png"))
    plt.close()

    print_trees(trees)
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

    # Сохраняем объекты
    ens.save_to_file(os.path.join(run_dir, "ens.pkl"))
    common_tree.save_to_file(os.path.join(run_dir, "common_tree.pkl"))

    with open(os.path.join(run_dir, "stat.pkl"), "wb") as file:
        cloudpickle.dump(stat, file)

    print(run_id, "done")
    return f1, precision, recall


def run_multiple_experiments(n_runs, n_processes, output_dir):
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(run_experiment, i, output_dir) for i in range(n_runs)]
        results = [future.result() for future in futures]

    # Распаковка
    f1_scores, precisions, recalls = zip(*results)

    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    with open(os.path.join(output_dir, "average_metrics.txt"), "w") as f:
        f.write(f"Average F1 Score (macro): {avg_f1:.4f}\n")
        f.write(f"Average Precision (macro): {avg_precision:.4f}\n")
        f.write(f"Average Recall (macro): {avg_recall:.4f}\n")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results2\one_tree_adult"
    n_runs = 1
    n_processes = 1
    run_multiple_experiments(n_runs, n_processes, output_dir)
