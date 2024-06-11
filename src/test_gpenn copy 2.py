import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SHADE, SelfCGA
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.classifiers._gpnneclassifier_one_tree import (
    GeneticProgrammingNeuralNetStackingClassifier,
)
from thefittest.tools.print import print_net
from thefittest.tools.print import print_tree
from thefittest.tools.print import print_nets
from thefittest.tools.print import print_trees
from thefittest.tools.print import print_ens


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# Assuming necessary imports for the BanknoteDataset, GeneticProgrammingNeuralNetStackingClassifier, SelfCGP, SelfCGA
# Also, assuming you have print_tree, print_trees, print_nets, print_net, print_ens functions available

def run_experiment(run_id, output_dir):
    data = BanknoteDataset()
    X = data.get_X()
    y = data.get_y()

    X_scaled = minmax_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

    model = GeneticProgrammingNeuralNetStackingClassifier(
        iters=35,
        pop_size=50,
        optimizer=SelfCGP,
        optimizer_args={"show_progress_each": 1},
        weights_optimizer=SelfCGA,
        weights_optimizer_args={"iters": 100, "pop_size": 100, "no_increase_num": 500},
    )

    model.fit(X_train, y_train)

    predict = model.predict(X_test)
    optimizer = model.get_optimizer()

    common_tree = optimizer.get_fittest()["genotype"]
    ens = optimizer.get_fittest()["phenotype"]
    trees = ens._trees

    f1 = f1_score(y_test, predict, average="macro")
    
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "f1_score.txt"), "w") as f:
        f.write(f"f1_score: {f1}\n")
    
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
    output_dir = r"C:\Users\user\Desktop\эксперименты\banknote"  # Change this to your desired output directory
    n_runs = 30  # Number of runs you want to perform
    n_processes = 15  # Number of processes to use in parallel
    run_multiple_experiments(n_runs, n_processes, output_dir)
