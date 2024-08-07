from thefittest.optimizers import DifferentialEvolution, jDE, SHADE
from thefittest.optimizers import GeneticAlgorithm
from thefittest.benchmarks import Rastrigin
from thefittest.utils.transformations import GrayCode
import numpy as np

# проследить какие параметры нельзя передавать индивидуальным алгоритмам например минимизацию или критерий остановки


iters = 10
adapt_period = 5
pop_size = 100
num_variables = 5
left_border = -4.5
right_border = 4.5
eps = 0.001

ga_genotype_to_phenotype = GrayCode().fit(
    left_border=left_border,
    right_border=right_border,
    num_variables=num_variables,
    h_per_variable=eps,
)

print(ga_genotype_to_phenotype.get_str_len())

individual_algorithms_params = (
    {
        "optimizer": DifferentialEvolution,
        "name": "DE",
        "params": {
            "mutation": "best_1",
            "F": 0.7,
            "CR": 0.7,
            "left_border": left_border,
            "right_border": right_border,
            "num_variables": num_variables,
            "show_progress_each": 1,
            "minimization": True,
        },
    },
    {
        "optimizer": SHADE,
        "name": "SHADE",
        "params": {
            "left_border": left_border,
            "right_border": right_border,
            "num_variables": num_variables,
            "show_progress_each": 1,
            "minimization": True,
        },
    },
    {
        "optimizer": GeneticAlgorithm,
        "name": "GA1",
        "params": {
            "genotype_to_phenotype": ga_genotype_to_phenotype.transform,
            "str_len": ga_genotype_to_phenotype.get_str_len(),
            "selection": "tournament_5",
            "crossover": "uniform_2",
            "mutation": "weak",
            "elitism": False,
            "show_progress_each": 1,
            "minimization": True,
        },
    },
    {
        "optimizer": GeneticAlgorithm,
        "name": "GA2",
        "params": {
            "genotype_to_phenotype": ga_genotype_to_phenotype.transform,
            "str_len": ga_genotype_to_phenotype.get_str_len(),
            "selection": "tournament_5",
            "crossover": "uniform_rank_2",
            "mutation": "weak",
            "elitism": True,
            "show_progress_each": 1,
            "minimization": True,
        },
    },
)


individual_algorithms = (iap["optimizer"](**iap["params"]) for iap in individual_algorithms_params)
individual_algorithms = {}
for i, iap in enumerate(individual_algorithms_params):
    name = iap["name"]
    optimizer = iap["optimizer"]
    params = iap["params"]

    params["fitness_function"] = Rastrigin()
    params["iters"] = iters
    params["pop_size"] = pop_size
    individual_algorithms[name] = optimizer(**params)


print(individual_algorithms)
# величина интервала адаптации (период работы индивид. алгоритмов),
# размер штрафа (сколько индивидов сокращается)
# и размер "социальной карточки" (минмум индивидов),
# количество и характеры выбираемых алгоритмов и т.д.
# 1. Размер "штрафа" - 15% от размера популяции индивидуального алгоритма, величина "социальной карточки" — 15%.
# 2. В состав коэволюционного алгоритма включать комбинацию из 3-5
# алгоритмов, обладающих различными характерами поиска.
# 3. Интервал адаптации должен составлять 5-6 поколений коэволюции.


# Как происходит обмен. (Best to All, All to All)
# Выбор индивида для передачи (Best, Random)
# Выбор индивида, который будет замещен (Worst, Random)
# BtA BrW
# BtA BrR
# BtA RrW
# BtA RrR
# AtA BrW
# AtA BrR
# AtA RrW
# AtA RrR
# SBSP

for name, algorithm_i in individual_algorithms.items():
    print(name)
    algorithm_i._get_init_population()
    algorithm_i._from_population_g_to_fitness()

for j in range(1):  # нужно итераций чтобы выполнить iters раз
    best_fitness_in = np.empty(shape=(adapt_period, len(individual_algorithms)), dtype=np.float64)
    for k, algorithm_i in enumerate(individual_algorithms.values()):
        for i in range(adapt_period):  # нужно заменить на период адаптации
            algorithm_i._show_progress(i)  # отображение прогресса самой коэволюции
            if algorithm_i._termitation_check():  # проверка остановки самой коэволюии
                pass
            else:
                algorithm_i._get_new_population()
                algorithm_i._from_population_g_to_fitness()

                best_fitness_in[i][k] = np.max(
                    algorithm_i._fitness_i
                )  # тут запись лучших решений для оценки алгоритмов

                if algorithm_i._on_generation is not None:
                    algorithm_i._on_generation(algorithm_i)
    print(individual_algorithms.keys())
    print("best_fitness_in: ", best_fitness_in)
    print("argmax:", np.argmax(best_fitness_in, axis=1))
    # print(algorithm_i._population_g_i, algorithm_i._population_ph_i)
