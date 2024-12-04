import pandas as pd

# Чтение CSV-файла (замените 'ga_combproblems.csv' на имя вашего файла)
df = pd.read_csv("shagaconf_combproblems.csv")

# Фильтрация данных на основе указанных условий
# df = df[
#     (df["Selection"].isin(["proportional", "rank", "tournament_3"]))
#     & (df["Crossover"].isin(["empty", "one_point", "uniform_2"]))
#     & (df["Mutation"].isin(["weak", "average", "strong"]))
# ]

# Сначала группируем по столбцам 'Function', 'Selection', 'Crossover', и 'Mutation' и находим максимум для каждой комбинации
max_per_combination = df.groupby(["Function", "Selection", "Crossover"], as_index=False)[
    "fitness"
].max()

# Добавляем также среднее значение 'fitness' по каждой комбинации
avg_per_combination = df.groupby(["Function", "Selection", "Crossover"], as_index=False)[
    "fitness"
].mean()
max_per_combination["avg_fitness"] = avg_per_combination["fitness"]

# Находим лучшую комбинацию для каждой функции (по максимальному значению 'fitness')
best_combination = max_per_combination.loc[
    max_per_combination.groupby("Function")["fitness"].idxmax()
]

# Группируем по 'Function' и находим среднее значение максимального 'fitness' для каждой функции
average_fitness = max_per_combination.groupby("Function")["avg_fitness"].mean().reset_index()

# Объединяем результаты в один финальный DataFrame
final_df = pd.merge(best_combination, average_fitness, on="Function", suffixes=("_best", "_avg"))

# Переименовываем столбцы для ясности
final_df = final_df.rename(
    columns={"fitness_best": "best_fitness", "avg_fitness_avg": "average_fitness_of_combinations"}
)

# Сохранение результата в новый CSV файл
final_df.to_csv("shagaconf_combproblems_combproblems_grouped_fitness_analysis.csv", index=False)

# Вывод результата
print(final_df)
