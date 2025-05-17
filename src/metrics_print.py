import os
import json
from glob import glob

# Путь к корневой папке с результатами
base_dir = r"C:\Users\USER\clones\thefittest\results_fuzzy_solar"

# Шаблон для поиска всех metrics.json в папках run_*
pattern = os.path.join(base_dir, "run_*", "metrics.json")

gne_values = []

for metrics_path in glob(pattern):
    run_name = os.path.basename(os.path.dirname(metrics_path))
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Извлекаем значение по ключу 'GNE'
        gne = data.get("GNE", None)
        print(f"{run_name}: GNE = {gne}")
        # Если значение есть и это число — добавляем в список
        if isinstance(gne, (int, float)):
            gne_values.append(gne)
    except (json.JSONDecodeError, IOError) as e:
        print(f"{run_name}: ошибка при чтении — {e}")

# Вычисляем и выводим среднее, если есть хотя бы одно числовое значение
if gne_values:
    avg_gne = sum(gne_values) / len(gne_values)
    print(f"\nСреднее GNE по всем запускам: {avg_gne}")
else:
    print("\nНе найдено ни одного числового значения GNE для вычисления среднего.")
