import os
from glob import glob
import re

# Путь к корневой папке с результатами
base_dir = r"C:\Users\USER\Desktop\solar_res\solar_res\FL"

# Шаблон для поиска всех fuzzy_rules.txt в папках run_*
pattern = os.path.join(base_dir, "run_*", "rules.txt")

all_avg_conditions = []
all_rule_counts = []

for rules_path in glob(pattern):
    run_name = os.path.basename(os.path.dirname(rules_path))
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = [line.strip() for line in f if line.strip()]

        rule_count = len(rules)
        all_rule_counts.append(rule_count)

        condition_counts = []
        for rule in rules:
            # Считаем количество условий вида (xN is xN_sM)
            conditions = re.findall(r"\(x\d+\s+is\s+x\d+_s\d+\)", rule)
            condition_counts.append(len(conditions))

        avg_conditions = sum(condition_counts) / len(condition_counts)
        min_conditions = min(condition_counts)
        max_conditions = max(condition_counts)
        all_avg_conditions.append(avg_conditions)

        print(f"{run_name}:")
        print(f"  Кол-во правил: {rule_count}")
        print(f"  Среднее предпосылок на правило: {avg_conditions:.2f}")
        print(f"  Мин./Макс. предпосылок: {min_conditions} / {max_conditions}")

    except IOError as e:
        print(f"{run_name}: ошибка при чтении — {e}")

# Глобальные итоги
if all_rule_counts and all_avg_conditions:
    overall_avg_rules = sum(all_rule_counts) / len(all_rule_counts)
    overall_avg_conditions = sum(all_avg_conditions) / len(all_avg_conditions)
    print("\n=== Общая статистика по всем запускам ===")
    print(f"Среднее количество правил: {overall_avg_rules:.2f}")
    print(f"Среднее количество предпосылок на правило: {overall_avg_conditions:.2f}")
else:
    print("\nНе удалось найти или обработать ни одного файла с правилами.")
