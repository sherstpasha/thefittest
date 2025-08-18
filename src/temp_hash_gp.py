import time
import numpy as np
from thefittest.optimizers import SHADE
from thefittest.benchmarks import DigitsDataset
from thefittest.classifiers import GeneticProgrammingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, f1_score


def run_experiment(use_fitness_cache: bool, n_runs: int = 1):
    times = []
    scores = []

    data = DigitsDataset()
    X = minmax_scale(data.get_X())
    y = data.get_y()

    for i in range(n_runs):
        print(i)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=i
        )

        model = GeneticProgrammingClassifier(
            n_iter=100,
            pop_size=500,
            use_fitness_cache=use_fitness_cache,
        )

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        predict = model.predict(X_test)
        score = f1_score(y_test, predict, average="macro")

        times.append(elapsed)
        scores.append(score)

        print(
            f"Run {i+1}/{n_runs}, use_cache={use_fitness_cache}, "
            f"time={elapsed:.2f}s, f1={score:.3f}"
        )

    return np.mean(times), np.std(times), np.mean(scores), np.std(scores)


# Сравниваем кэш ON vs OFF
mean_time_on, std_time_on, mean_score_on, std_score_on = run_experiment(True)
mean_time_off, std_time_off, mean_score_off, std_score_off = run_experiment(False)

print("\n===== Итоги =====")
print(f"С кэшем:   {mean_time_on:.2f} ± {std_time_on:.2f} сек, f1={mean_score_on:.3f} ± {std_score_on:.3f}")
print(f"Без кэша:  {mean_time_off:.2f} ± {std_time_off:.2f} сек, f1={mean_score_off:.3f} ± {std_score_off:.3f}")
