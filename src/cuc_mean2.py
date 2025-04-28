import os
import json
import numpy as np

# Путь к папке, где лежат все run_0, run_1, ..., run_19
root_dir = r"C:\Users\USER\Desktop\Расчеты по нейросетям\расчеты сетей метео\GPENN"

# Словарь для сбора всех метрик
metrics_all = {}

# Проходим по всем запускам
for run_number in range(20):
    run_dir = os.path.join(root_dir, f"run_{run_number}")
    metrics_test_path = os.path.join(run_dir, "metrics_test.json")
    
    if not os.path.exists(metrics_test_path):
        print(f"⚠️ Нет metrics_test.json в {run_dir}, пропускаем")
        continue

    with open(metrics_test_path, "r", encoding="utf-8") as f:
        metrics_test = json.load(f)
    
    for target, target_metrics in metrics_test.items():
        if target not in metrics_all:
            metrics_all[target] = {metric_name: [] for metric_name in target_metrics.keys()}
        
        for metric_name, value in target_metrics.items():
            metrics_all[target][metric_name].append(value)

# Теперь считаем среднее по каждому выходу
metrics_average = {}

for target, target_metrics in metrics_all.items():
    metrics_average[target] = {}
    for metric_name, values in target_metrics.items():
        metrics_average[target][metric_name] = float(np.mean(values))

# Сохраняем результат
save_path = os.path.join(root_dir, "metrics_test_average.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(metrics_average, f, indent=4, ensure_ascii=False)

print(f"\n✅ Средние метрики по всем 20 запускам сохранены в {save_path}")