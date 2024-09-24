import pandas as pd

# Load the data from the provided path
data = pd.read_csv(r"shaga_conf_problems13.csv")

# Filter the data based on the specified conditions
# filtered_data = data[
#     (data["Selection"].isin(["proportional", "rank", "tournament_3"]))
#     & (data["Crossover"].isin(["empty", "one_point", "uniform_2"]))
#     & (data["Mutation"].isin(["weak", "average", "strong"]))
# ]
filtered_data = data


print(len(filtered_data))

# Group by the specified columns and calculate the mean for 'find_solution' and 'generation_found'
grouped_data = (
    filtered_data.groupby(
        [
            "Function",
            "Dimensions",
            "Selection",
            "Crossover",
            # "Mutation",
            "Pop_Size",
            "Iters",
        ]
    )
    .agg({"find_solution": "mean", "generation_found": "mean"})
    .reset_index()
)

# Rename the columns for clarity
grouped_data.rename(
    columns={"find_solution": "Reliability", "generation_found": "Avg_Speed"},
    inplace=True,
)

# Calculate the maximum Reliability, average Speed, and extract Pop_Size and Iters for each Function
result = (
    grouped_data.groupby("Function")
    .agg(
        {
            "Reliability": ["mean", "max"],  # Calculate mean and max of Reliability
            "Avg_Speed": "mean",  # Calculate mean of Avg_Speed
            "Pop_Size": "first",  # Take the first occurrence of Pop_Size
            "Iters": "first",  # Take the first occurrence of Iters
        }
    )
    .reset_index()
)

# Flatten the multi-level columns
result.columns = [
    "Function",
    "Mean_Reliability",
    "Max_Reliability",
    "Avg_Speed",
    "Pop_Size",
    "Iters",
]

# Округлим значения до трех знаков после запятой
result["Mean_Reliability"] = result["Mean_Reliability"].round(3)
result["Max_Reliability"] = result["Max_Reliability"].round(3)
result["Avg_Speed"] = result["Avg_Speed"].round(3)

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
result["Function"] = pd.Categorical(
    result["Function"], categories=function_order, ordered=True
)
result = result.sort_values("Function").reset_index(drop=True)

# Проверим результат
print(result)

# # Сохраните результат в новый CSV файл
result.to_csv("mean_max_shaga_conf_problem13.csv", index=False)
