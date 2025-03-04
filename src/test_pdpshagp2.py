from collections import defaultdict
from thefittest.optimizers._pdpshagp import PDPSHAGP, SelfCSHAGP
import numpy as np
import matplotlib.pyplot as plt
import os

from thefittest.base import FunctionalNode, TerminalNode, EphemeralNode, UniversalSet
from thefittest.tools.operators import Mul, Add, Div, Neg, Sin, Exp
from thefittest.tools.metrics import root_mean_square_error
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# =======================
# Загружаем данные из файла (первая функция из папки)
# =======================
data_folder = r"C:\Users\pasha\OneDrive\Рабочий стол\Feynman_120"
files_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
if not files_list:
    raise ValueError("В указанной папке не найдено файлов!")
first_file = files_list[0]
file_path = os.path.join(data_folder, first_file)
print("Используем файл:", first_file)

# Загружаем данные (предполагается, что данные в виде числовой матрицы)
data = np.loadtxt(file_path).astype(np.float32)
# Для эксперимента возьмём первые 1000 строк: X – все столбцы, кроме последнего; y – последний столбец
X = data[:1000, :-1]
y = data[:1000, -1]
sample_size = X.shape[0]
n_dimension = X.shape[1]

# =======================
# Определяем универсальное множество (uniset)
# =======================
functional_set = (
    FunctionalNode(Add()),
    FunctionalNode(Mul()),
    FunctionalNode(Neg()),
    FunctionalNode(Div()),
    FunctionalNode(Sin()),
    # Можно добавить FunctionalNode(Exp()) по необходимости
)
terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset = UniversalSet(functional_set, tuple(terminal_set))


# =======================
# Функция приспособленности
# Ожидается, что вызов tree() возвращает массив предсказаний для X
# =======================
def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree() * np.ones(len(y))  # tree() должен вернуть np.array длины sample_size
        fitness.append(root_mean_square_error(y.astype(np.float32), y_pred.astype(np.float32)))
    return np.array(fitness, dtype=np.float32)


# =======================
# Параметры эксперимента
# =======================
number_of_iterations = 1000
population_size = 100
num_runs = 300


# Функция одного запуска оптимизатора
def run_experiment(_):
    optimizer = PDPSHAGP(
        fitness_function=fitness_function,
        uniset=uniset,
        pop_size=population_size,
        iters=number_of_iterations,
        minimization=True,
        keep_history=True,
        elitism=False,
    )
    optimizer.fit()
    stats = optimizer.get_stats()

    # Приводим статистику к нужному виду
    run_max_fitness = np.array(stats["max_fitness"])
    run_H_MR = np.array(stats["H_MR"]).mean(axis=1)
    run_H_CR = np.array(stats["H_CR"]).mean(axis=1)
    best_run = max(stats["max_fitness"])
    run_s_proba = stats["s_proba"]  # список словарей для каждой итерации
    run_c_proba = stats["c_proba"]
    run_m_proba = stats["m_proba"]

    return {
        "max_fitness": run_max_fitness,
        "s_proba": run_s_proba,
        "c_proba": run_c_proba,
        "m_proba": run_m_proba,
        "H_MR": run_H_MR,
        "H_CR": run_H_CR,
        "best": best_run,
    }


# =======================
# Проведение эксперимента (num_runs запусков)
# =======================
if __name__ == "__main__":
    agg_max_fitness = np.zeros(number_of_iterations)
    agg_s_proba = defaultdict(lambda: np.zeros(number_of_iterations))
    agg_c_proba = defaultdict(lambda: np.zeros(number_of_iterations))
    agg_m_proba = defaultdict(lambda: np.zeros(number_of_iterations))
    agg_H_MR = np.zeros(number_of_iterations)
    agg_H_CR = np.zeros(number_of_iterations)
    best_fitnesses = []

    results = []
    with ProcessPoolExecutor() as executor:
        for res in tqdm(
            executor.map(run_experiment, range(num_runs)),
            total=num_runs,
            desc="Запуски эксперимента",
        ):
            results.append(res)

    # Агрегируем статистику по всем запускам
    for res in results:
        agg_max_fitness += res["max_fitness"]
        for i, s_dict in enumerate(res["s_proba"]):
            for key, value in s_dict.items():
                agg_s_proba[key][i] += value
        for i, c_dict in enumerate(res["c_proba"]):
            for key, value in c_dict.items():
                agg_c_proba[key][i] += value
        for i, m_dict in enumerate(res["m_proba"]):
            for key, value in m_dict.items():
                agg_m_proba[key][i] += value
        agg_H_MR += res["H_MR"]
        agg_H_CR += res["H_CR"]
        best_fitnesses.append(res["best"])

    # Усредняем накопленную статистику
    avg_max_fitness = agg_max_fitness / num_runs
    avg_s_proba = {key: value / num_runs for key, value in agg_s_proba.items()}
    avg_c_proba = {key: value / num_runs for key, value in agg_c_proba.items()}
    avg_m_proba = {key: value / num_runs for key, value in agg_m_proba.items()}
    avg_H_MR = agg_H_MR / num_runs
    avg_H_CR = agg_H_CR / num_runs

    print("Средняя лучшая приспособленность за все запуски:", np.mean(best_fitnesses))

    # =======================
    # Построение графиков
    # Создаем фигуру с 3 рядами и 2 колонками графиков
    fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=3)

    # График максимальной приспособленности
    ax[0][0].plot(range(number_of_iterations), avg_max_fitness)
    ax[0][0].set_title("Средняя максимальная приспособленность")
    ax[0][0].set_ylabel("Fitness value")
    ax[0][0].set_xlabel("Итерация")

    # График вероятностей селекции
    for key, value in avg_s_proba.items():
        ax[0][1].plot(range(number_of_iterations), value, label=key)
    ax[0][1].legend()
    ax[0][1].set_title("Вероятности селекции")
    ax[0][1].set_xlabel("Итерация")

    # График вероятностей кроссовера
    for key, value in avg_c_proba.items():
        ax[1][0].plot(range(number_of_iterations), value, label=key)
    ax[1][0].legend()
    ax[1][0].set_title("Вероятности кроссовера")
    ax[1][0].set_xlabel("Итерация")

    # График вероятностей мутации
    for key, value in avg_m_proba.items():
        ax[1][1].plot(range(number_of_iterations), value, label=key)
    ax[1][1].legend()
    ax[1][1].set_title("Вероятности мутации")
    ax[1][1].set_xlabel("Итерация")

    # График статистики H_MR
    ax[2][0].plot(range(number_of_iterations), avg_H_MR)
    ax[2][0].set_title("Среднее H_MR")
    ax[2][0].set_xlabel("Итерация")

    # График статистики H_CR
    ax[2][1].plot(range(number_of_iterations), avg_H_CR)
    ax[2][1].set_title("Среднее H_CR")
    ax[2][1].set_xlabel("Итерация")

    plt.tight_layout()
    plt.savefig("pdpshagp_avg.png", dpi=300, bbox_inches="tight")
    plt.show()
