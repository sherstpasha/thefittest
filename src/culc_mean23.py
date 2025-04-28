import json
import os
from collections import defaultdict

# Путь к папке с файлами
base_path = r"C:\Users\USER\Desktop\Расчеты по нейросетям\расчеты сетей метео\FLS"

# Инициализация структуры для накопления данных
aggregated_metrics = defaultdict(lambda: defaultdict(list))

# Чтение всех файлов
for i in range(20):
    file_path = os.path.join(base_path, f"run_{i}", "metrics.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for wind_key, metrics in data.items():
        for metric_name, metric_value in metrics.items():
            aggregated_metrics[wind_key][metric_name].append(metric_value)

# Расчет средних значений по каждому WindSpeed_t+i
mean_metrics = {}
for wind_key, metrics in aggregated_metrics.items():
    mean_metrics[wind_key] = {}
    for metric_name, values in metrics.items():
        mean_metrics[wind_key][metric_name] = sum(values) / len(values)

# Сохранение среднего по каждому WindSpeed_t+i
average_metrics_path = os.path.join(base_path, "average_metrics.json")
with open(average_metrics_path, 'w', encoding='utf-8') as f:
    json.dump(mean_metrics, f, indent=4, ensure_ascii=False)

print(f"Усредненные метрики по каждому WindSpeed_t+ сохранены в {average_metrics_path}")

# --- Теперь усредняем по всем WindSpeed_t+1..t+12 ---
final_metrics = defaultdict(list)

for wind_key, metrics in mean_metrics.items():
    for metric_name, value in metrics.items():
        final_metrics[metric_name].append(value)

final_average_metrics = {metric_name: sum(values)/len(values) for metric_name, values in final_metrics.items()}

# Сохранение финального среднего по всем 12 шагам
final_average_metrics_path = os.path.join(base_path, "final_average_metrics.json")
with open(final_average_metrics_path, 'w', encoding='utf-8') as f:
    json.dump(final_average_metrics, f, indent=4, ensure_ascii=False)

print(f"Финальные средние метрики по всем WindSpeed_t+ сохранены в {final_average_metrics_path}")
