import os
import matplotlib.pyplot as plt
import numpy as np
import cloudpickle

from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from thefittest.optimizers import SelfCGP, SelfCGA
from thefittest.regressors._gpnneregression_one_tree import GeneticProgrammingNeuralNetStackingRegressor
from thefittest.tools.print import print_tree, print_trees, print_net, print_nets, print_ens
from thefittest.benchmarks import CombinedCycleDataset





def run_experiment(run_id, output_dir):
    dataset = CombinedCycleDataset()
    X = minmax_scale(dataset.X)
    y = dataset.y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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

    print(f"Run {run_id} done - R2: {r2:.4f}")
    return r2


def run_multiple_experiments(n_runs, output_dir, start_run=0):
    r2_scores = []
    for i in range(start_run, n_runs):
        r2 = run_experiment(i, output_dir)
        r2_scores.append(r2)

    avg_r2 = np.mean(r2_scores)
    with open(os.path.join(output_dir, "average_r2_score.txt"), "w") as f:
        f.write(f"Average R2 Score: {avg_r2}\n")

    print(f"\nAll runs complete. Average R2: {avg_r2:.4f}")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results_regression_combined"
    n_runs = 1
    run_multiple_experiments(n_runs, output_dir)
