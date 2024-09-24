import pandas as pd

# Load the data from the provided path
data = pd.read_csv(
    r"selfcshaga_problems13.csv"
)

# Группируем только по функциям и вычисляем средние значения
grouped_data = (
    data.groupby("Function")
    .agg(
        {
            "find_solution": "mean",  # Средняя доля найденных решений (надежность)
            "generation_found": "mean",  # Среднее значение скорости
            "Pop_Size": "first",  # Используем первое значение размера популяции (оно одинаковое для каждой функции)
            "Iters": "first",  # Используем первое значение количества итераций (оно одинаковое для каждой функции)
        }
    )
    .reset_index()
)

# Переименуем колонки для ясности
grouped_data.rename(
    columns={"find_solution": "Reliability", "generation_found": "Avg_Speed"},
    inplace=True,
)

# Округлим значения до трех знаков после запятой
grouped_data["Reliability"] = grouped_data["Reliability"].round(3)
grouped_data["Avg_Speed"] = grouped_data["Avg_Speed"].round(3)

# Зададим порядок функций в том порядке, который указан на изображении
# function_order = [
#     "ShiftedRosenbrock",
#     "ShiftedRotatedGriewank",
#     "ShiftedExpandedGriewankRosenbrock",
#     "RotatedVersionHybridCompositionFunction1",
#     "RotatedVersionHybridCompositionFunction1Noise",
#     "RotatedHybridCompositionFunction",
#     "HybridCompositionFunction3",
#     "HybridCompositionFunction3H",
#     "NonContinuousHybridCompositionFunction3",
#     "HybridCompositionFunction4",
#     "HybridCompositionFunction4withoutbounds",
#     "Rosenbrock",
#     "ExpandedScaffers_F6",
#     "Weierstrass",
#     "ShiftedSphere",
#     "ShiftedSchwefe1_2",
#     "ShiftedSchwefe1_2WithNoise",
#     "ShiftedRastrigin",
#     "ShiftedRotatedRastrigin",
#     "HybridCompositionFunction1",
#     "Sphere",
#     "HighConditionedElliptic",
#     "Griewank",
#     "Ackley",
#     "Rastrigin",
# ]
function_order = [
    "Func1",
    "Func2",
    "Func3",
    "Func4",
    "Func5",
    "Func6",
    "Func7",
    "Func8",
    "Func9",
    "Func10",
    "Func11",
    "Func12",
    "Func13",
]

# Приведем функции в итоговом DataFrame в указанный порядок
grouped_data["Function"] = pd.Categorical(
    grouped_data["Function"], categories=function_order, ordered=True
)
grouped_data = grouped_data.sort_values("Function").reset_index(drop=True)

# Проверим результат
print(grouped_data)

# Сохраните результат в новый CSV файл
grouped_data.to_csv("mean_selfcshaga_problems13.csv", index=False)
