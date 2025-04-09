import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.datasets import load_diabetes
import cloudpickle

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SelfCGA
from thefittest.regressors._gpnneregression_one_tree import (
    GeneticProgrammingNeuralNetStackingRegressor,
)
from thefittest.tools.print import (
    print_net,
    print_tree,
    print_nets,
    print_trees,
    print_ens,
)


def run_experiment(run_id, output_dir):
    data = load_diabetes()
    X = data.data
    y = data.target

    X_scaled = minmax_scale(X)
    # y_scaled = minmax_scale(y.reshape(-1, 1)).flatten()  # Масштабируем y

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    model = GeneticProgrammingNeuralNetStackingRegressor(
        iters=20,
        pop_size=50,
        input_block_size=1,
        optimizer=SelfCGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 10},
        weights_optimizer=SelfCGA,
        weights_optimizer_args={
            "iters": 100000,
            "pop_size": 400,
            "no_increase_num": 100,
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

    r2 = r2_score(y_test, predict)

    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "r2_score.txt"), "w") as f:
        f.write(f"R2 Score: {r2}\n")

    np.savetxt(os.path.join(run_dir, "X_train.txt"), X_train)
    np.savetxt(os.path.join(run_dir, "X_test.txt"), X_test)
    np.savetxt(os.path.join(run_dir, "y_train.txt"), y_train)
    np.savetxt(os.path.join(run_dir, "y_test.txt"), y_test)
    np.savetxt(os.path.join(run_dir, "predictions.txt"), predict)

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
    return r2


def run_multiple_experiments(n_runs, n_processes, output_dir):
    run_experiment(0, output_dir)
    # with ProcessPoolExecutor(max_workers=n_processes) as executor:
    #     futures = [executor.submit(run_experiment, i, output_dir) for i in range(n_runs)]
    #     r2_scores = [future.result() for future in futures]

    # avg_r2 = np.mean(r2_scores)

    # with open(os.path.join(output_dir, "average_r2_score.txt"), "w", encoding="utf-8") as f:
    #     f.write(f"Average R2 Score: {avg_r2}\n")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results3\one_tree_diabetes"
    n_runs = 1
    n_processes = 1
    run_multiple_experiments(n_runs, n_processes, output_dir)
