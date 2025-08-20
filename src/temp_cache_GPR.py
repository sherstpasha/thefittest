import time
import numpy as np
from thefittest.regressors import GeneticProgrammingRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score, root_mean_squared_error


# синтетическая функция
def target_function(X):
    return np.sin(X[:, 0]) + 0.5 * (X[:, 1] ** 2)


def run_experiment(use_fitness_cache: bool, n_runs: int = 1):
    times = []
    scores_r2 = []
    scores_rmse = []

    # генерируем данные
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, size=(1000, 2))
    y = target_function(X)

    X = minmax_scale(X)

    for i in range(n_runs):
        print(i)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )

        model = GeneticProgrammingRegressor(
            n_iter=500,
            pop_size=500,
            use_fitness_cache=use_fitness_cache,
        )

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        predict = model.predict(X_test)
        r2 = r2_score(y_test, predict)
        rmse = root_mean_squared_error(y_test, predict)

        times.append(elapsed)
        scores_r2.append(r2)
        scores_rmse.append(rmse)

        print(
            f"Run {i+1}/{n_runs}, use_cache={use_fitness_cache}, "
            f"time={elapsed:.2f}s, R2={r2:.3f}, RMSE={rmse:.3f}"
        )

    return (
        np.mean(times), np.std(times),
        np.mean(scores_r2), np.std(scores_r2),
        np.mean(scores_rmse), np.std(scores_rmse),
    )


# Сравниваем кэш ON vs OFF
mean_time_on, std_time_on, mean_r2_on, std_r2_on, mean_rmse_on, std_rmse_on = run_experiment(True)
mean_time_off, std_time_off, mean_r2_off, std_r2_off, mean_rmse_off, std_rmse_off = run_experiment(False)

print("\n===== Итоги =====")
print(f"С кэшем:   {mean_time_on:.2f} ± {std_time_on:.2f} сек, "
      f"R2={mean_r2_on:.3f} ± {std_r2_on:.3f}, RMSE={mean_rmse_on:.3f} ± {std_rmse_on:.3f}")
print(f"Без кэша:  {mean_time_off:.2f} ± {std_time_off:.2f} сек, "
      f"R2={mean_r2_off:.3f} ± {std_r2_off:.3f}, RMSE={mean_rmse_off:.3f} ± {std_rmse_off:.3f}")
