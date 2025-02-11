from collections import defaultdict
from thefittest.optimizers._pdpshagp import PDPSHAGP, SelfCSHAGP
import numpy as np
import matplotlib.pyplot as plt

from thefittest.base import FunctionalNode, TerminalNode, EphemeralNode, UniversalSet
from thefittest.tools.operators import Mul, Add, Div, Neg, Sin, Exp
from thefittest.tools.metrics import coefficient_determination, root_mean_square_error
from thefittest.tools.print import print_tree
from thefittest.benchmarks.symbolicregression17 import problems_dict

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Выбираем задачу
F = "F3"

def generator1():
    return np.round(np.random.uniform(0, 10), 4)

def generator2():
    return np.random.randint(0, 10)

def problem(x):
    return problems_dict[F]["function"](x)

# Параметры датасета
function = problem
left_border = problems_dict[F]["bounds"][0]
right_border = problems_dict[F]["bounds"][1]
sample_size = 300
n_dimension = problems_dict[F]["n_vars"]

# Формируем выборку
X = np.array([np.linspace(left_border, right_border, sample_size)
              for _ in range(n_dimension)]).T
y = function(X)

# Определяем функциональное и терминальное множество
functional_set = (
    FunctionalNode(Add()),
    FunctionalNode(Mul()),
    FunctionalNode(Neg()),
    FunctionalNode(Div()),
    FunctionalNode(Sin()),
    # FunctionalNode(Exp()),
)

terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset = UniversalSet(functional_set, tuple(terminal_set))

# Функция приспособленности
def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree() * np.ones(len(y))
        fitness.append(root_mean_square_error(y.astype(np.float32),
                                                   y_pred.astype(np.float32)))
    return np.array(fitness, dtype=np.float32)

# Параметры алгоритма и эксперимента
number_of_iterations = 2000
population_size = 100
num_runs = 100

def run_experiment(_):
    """
    Функция одного запуска оптимизатора.
    Возвращает словарь со статистикой по данному запуску.
    """
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

if __name__ == "__main__":
    # Инициализируем накопители для статистики
    agg_max_fitness = np.zeros(number_of_iterations)
    agg_s_proba = defaultdict(lambda: np.zeros(number_of_iterations))
    agg_c_proba = defaultdict(lambda: np.zeros(number_of_iterations))
    agg_m_proba = defaultdict(lambda: np.zeros(number_of_iterations))
    agg_H_MR = np.zeros(number_of_iterations)
    agg_H_CR = np.zeros(number_of_iterations)
    best_fitnesses = []
    
    results = []
    # Распараллеливаем запуски с отображением прогресс-бара
    with ProcessPoolExecutor() as executor:
        # Используем executor.map; tqdm оборачивает итератор для отображения прогресса.
        for res in tqdm(executor.map(run_experiment, range(num_runs)),
                        total=num_runs, desc="Запуски эксперимента"):
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
    
    # Построение графиков
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
    plt.savefig("pdpshagp_avg.png")
    plt.show()
