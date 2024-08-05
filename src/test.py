from thefittest.optimizers import DifferentialEvolution, jDE, SHADE
from thefittest.optimizers import GeneticAlgorithm
from thefittest.benchmarks import Rastrigin
from thefittest.utils.transformations import GrayCode

# проследить какие параметры нельзя передавать индивидуальным алгоритмам например минимизацию или критерий остановки


iters = 10
pop_size = 10
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
        "params": {
            "mutation": "rand_1",
            "F": 0.5,
            "CR": 0.3,
            "left_border": left_border,
            "right_border": right_border,
            "num_variables": num_variables,
            "show_progress_each": 1,
            "minimization": True,
        },
    },
    {
        "optimizer": GeneticAlgorithm,
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
)


individual_algorithms = (iap["optimizer"](**iap["params"]) for iap in individual_algorithms_params)
individual_algorithms = {}
for i, iap in enumerate(individual_algorithms_params):
    optimizer = iap["optimizer"]
    params = iap["params"]

    params["fitness_function"] = Rastrigin()
    params["iters"] = iters
    params["pop_size"] = pop_size
    individual_algorithms[f"algorithm_{i + 1}"] = optimizer(**params)


print(individual_algorithms)
# величина интервала адаптации (период работы индивид. алгоритмов),
# размер штрафа (сколько индивидов сокращается)
# и размер "социальной карточки" (минмум индивидов),
# количество и характеры выбираемых алгоритмов и т.д.
# 1. Размер "штрафа" - 15% от размера популяции индивидуального алгоритма, величина "социальной карточки" — 15%.
# 2. В состав коэволюционного алгоритма включать комбинацию из 3-5
# алгоритмов, обладающих различными характерами поиска.
# 3. Интервал адаптации должен составлять 5-6 поколений коэволюции.


for name, algorithm_i in individual_algorithms.items():
    print(name)
    algorithm_i._get_init_population()
    algorithm_i._from_population_g_to_fitness()
    for i in range(iters - 1):
        algorithm_i._show_progress(i)
        if algorithm_i._termitation_check():
            pass
        else:
            algorithm_i._get_new_population()
            algorithm_i._from_population_g_to_fitness()
            if algorithm_i._on_generation is not None:
                algorithm_i._on_generation(algorithm_i)

    print(algorithm_i._population_g_i, algorithm_i._population_ph_i)
