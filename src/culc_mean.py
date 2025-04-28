import os
import json
import numpy as np

# Путь к папке, где лежат все run_0, run_1, ..., run_19
base_dir = r"C:\Users\USER\Desktop\Расчеты по нейросетям\расчеты сетей метео\GPENN"

# Собираем все метрики
all_metrics = []

for run_id in range(20):
    metrics_path = os.path.join(base_dir, f"run_{run_id}", "metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"⚠️ Метрики не найдены для run_{run_id}")
        continue
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        all_metrics.append(metrics)

# Проверяем, что что-то собрано
if not all_metrics:
    print("❌ Не найдено ни одного metrics.json")
    exit()

# Ключи метрик
metric_names = all_metrics[0].keys()

# Среднее значение по всем прогонкам
avg_metrics = {
    metric: np.mean([m[metric] for m in all_metrics])
    for metric in metric_names
}

# Печать результата
print("\nСредние метрики по всем запускам:")
for k, v in avg_metrics.items():
    print(f"{k}: {v:.4f}")

# Сохраняем результат
save_path = os.path.join(base_dir, "average_metrics_all_runs.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(avg_metrics, f, indent=4, ensure_ascii=False)

print(f"\n✅ Средние метрики сохранены в {save_path}")
