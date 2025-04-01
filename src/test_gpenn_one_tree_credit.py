import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SelfCGA
from thefittest.benchmarks import CreditRiskDataset
from thefittest.classifiers._gpnneclassifier_one_tree import (
    GeneticProgrammingNeuralNetStackingClassifier,
)
from thefittest.tools.print import print_net
from thefittest.tools.print import print_tree
from thefittest.tools.print import print_nets
from thefittest.tools.print import print_trees
from thefittest.tools.print import print_ens
import cloudpickle


def run_experiment(run_id, output_dir):
    data = CreditRiskDataset()
    X = data.get_X()
    y = data.get_y()

    X_scaled = minmax_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)

    model = GeneticProgrammingNeuralNetStackingClassifier(
        iters=15,
        pop_size=35,
        input_block_size=1,
        optimizer=SelfCGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 10},
        weights_optimizer=SelfCGA,
        weights_optimizer_args={
            "iters": 100000,
            "pop_size": 300,
            "no_increase_num": 50,
            "fitness_update_eps": 0.0001,
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

    f1 = f1_score(y_test, predict, average="macro")

    run_dir = os.path.join(output_dir, f"run_{run_id}")
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

    ens.save_to_file(os.path.join(run_dir, "ens.pkl"))

    common_tree.save_to_file(os.path.join(run_dir, "common_tree.pkl"))

    with open(os.path.join(run_dir, "stat.pkl"), "wb") as file:
        cloudpickle.dump(stat, file)

    print(run_id, "done")
    return f1


def run_multiple_experiments(n_runs, n_processes, output_dir):
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(run_experiment, i, output_dir) for i in range(n_runs)]
        f1_scores = [future.result() for future in futures]

    avg_f1 = np.mean(f1_scores)

    with open(os.path.join(output_dir, "average_f1_score.txt"), "w") as f:
        f.write(f"Average F1 Score: {avg_f1}\n")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results2\one_tree_credit"
    n_runs = 30  # Number of runs you want to perform
    n_processes = 1  # Number of processes to use in parallel
    run_multiple_experiments(n_runs, n_processes, output_dir)
