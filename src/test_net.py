import os


def calculate_average_f1(base_dir):
    f1_scores = []

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("run_"):
            f1_file_path = os.path.join(folder_path, "f1_score.txt")
            if os.path.exists(f1_file_path):
                with open(f1_file_path, "r") as f:
                    line = f.readline().strip()
                    if line.startswith("f1_score:"):
                        try:
                            score = float(line.split(":")[1].strip())
                            f1_scores.append(score)
                        except ValueError:
                            print(f"Ошибка чтения числа в {f1_file_path}")
            else:
                print(f"Файл f1_score.txt не найден в {folder_path}")

    if f1_scores:
        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"Средний F1-скор: {avg_f1} на основе {len(f1_scores)} запусков.")
        return avg_f1
    else:
        print("Не найдено ни одного корректного f1_score.txt.")
        return None


# Пример использования
base_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results2\final_res\Breast\one_tree_breast"
calculate_average_f1(base_dir)
