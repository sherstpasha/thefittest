import os
import json
from glob import glob

# Путь к корневой папке с результатами
base_dir = r"C:\Users\USER\Desktop\solar_res\solar_res\FL"

# Шаблон для поиска всех metrics.json в папках run_*
pattern = os.path.join(base_dir, "run_*", "metrics.json")

# Списки для хранения значений метрик
gne_vs_net_values = []
gne_vs_true_values = []

for metrics_path in glob(pattern):
    run_name = os.path.basename(os.path.dirname(metrics_path))
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gne_vs_net = data.get("GNE_vs_net")
        gne_vs_true = data.get("GNE_vs_true")

        print(f"{run_name}: GNE_vs_net = {gne_vs_net}, GNE_vs_true = {gne_vs_true}")

        if isinstance(gne_vs_net, (int, float)):
            gne_vs_net_values.append(gne_vs_net)
        if isinstance(gne_vs_true, (int, float)):
            gne_vs_true_values.append(gne_vs_true)

    except (json.JSONDecodeError, IOError) as e:
        print(f"{run_name}: ошибка при чтении — {e}")

# Выводим средние значения по каждой метрике
if gne_vs_net_values:
    avg_net = sum(gne_vs_net_values) / len(gne_vs_net_values)
    print(f"\nСреднее GNE_vs_net по всем запускам: {avg_net}")
else:
    print("\nНе найдено ни одного числового значения GNE_vs_net для вычисления среднего.")

if gne_vs_true_values:
    avg_true = sum(gne_vs_true_values) / len(gne_vs_true_values)
    print(f"Среднее GNE_vs_true по всем запускам: {avg_true}")
else:
    print("Не найдено ни одного числового значения GNE_vs_true для вычисления среднего.")
