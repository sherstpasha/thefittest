import pandas as pd

# Загрузка данных из файла
file_path = "ga_cec2005.csv"
data = pd.read_csv(file_path)

# Определение порядка функций (в нужной последовательности)
function_order = [
    "ShiftedRosenbrock", "ShiftedRotatedGriewank", "ShiftedExpandedGriewankRosenbrock",
    "RotatedVersionHybridCompositionFunction1", "RotatedVersionHybridCompositionFunction1Noise",
    "RotatedHybridCompositionFunction", "HybridCompositionFunction3", "HybridCompositionFunction3H",
    "NonContinuousHybridCompositionFunction3", "HybridCompositionFunction4",
    "HybridCompositionFunction4withoutbounds", "Rosenbrock", "ExpandedScaffers_F6",
    "Weierstrass", "ShiftedSphere", "ShiftedSchwefe1_2", "ShiftedSchwefe1_2WithNoise",
    "ShiftedRastrigin", "ShiftedRotatedRastrigin", "HybridCompositionFunction1", "Sphere",
    "HighConditionedElliptic", "Griewank", "Ackley", "Rastrigin"
]

# Присваивание порядковых номеров каждому названию функции
function_mapping = {name: idx + 1 for idx, name in enumerate(function_order)}

# Замена названий функций на их порядковые номера в исходных данных
data['Function'] = data['Function'].map(function_mapping)

# Создание упрощённых меток групп операторов: prop | unif2 | avg
data['Group'] = (
    data['Selection'].map({
        'proportional': 'prop',
        'rank': 'rank',
        'tournament_3': 'tour3',
        'tournament_5': 'tour5',
        'tournament_7': 'tour7'
    }) + ' | ' + data['Crossover'].map({
        'empty': 'empty',
        'one_point': 'one',
        'two_point': 'two',
        'uniform_2': 'unif2',
        'uniform_7': 'unif7',
        'uniform_prop_2': 'up2',
        'uniform_prop_7': 'up7',
        'uniform_rank_2': 'ur2',
        'uniform_rank_7': 'ur7',
        'uniform_tour_3': 'ut3',
        'uniform_tour_7': 'ut7'
    }) + ' | ' + data['Mutation']
)

# Поиск наилучшей комбинации операторов для каждой функции (с максимальной надежностью)
best_combination = data.loc[data.groupby('Function')['find_solution'].idxmax()]

# Создание таблицы: строки — порядковые номера функций, столбец — лучшая комбинация операторов
function_best_combinations = pd.DataFrame({
    'Function_Number': best_combination['Function'],
    'Best_Combination': best_combination['Group']
})

# Сортировка функций по их порядковым номерам
function_best_combinations = function_best_combinations.sort_values('Function_Number')

# Печать итоговой таблицы
print("Таблица с порядковыми номерами функций и лучшими комбинациями операторов:")
print(function_best_combinations)

# Сохранение результата в CSV для дальнейшего анализа
function_best_combinations.to_csv("problems25_function_best_combinations_by_number.csv", index=False)
