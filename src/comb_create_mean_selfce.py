import pandas as pd

# Загрузка данных из файла
df = pd.read_csv("selfcga_all_combproblem.csv")

# Группировка по 'Function' и расчет среднего значения
result_df = df.groupby("Function").mean().reset_index()

# Сохранение результата в новый файл
result_df.to_csv("selfcga_all_combproblem_grouped_averaged_output.csv", index=False)
print(result_df)
print("Группировка и усреднение завершены. Результат сохранен в 'grouped_averaged_output.csv'")
