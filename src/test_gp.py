import time
import pandas as pd
from tqdm import tqdm  # <-- Библиотека для прогресс-бара

from thefittest.optimizers import SelfCGP, SHAGA
from thefittest.benchmarks import IrisDataset, WineDataset, DigitsDataset
from thefittest.classifiers import MLPEAClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, f1_score

# -- 1. WARM-UP -------------------------------------------------------------- #

# Для прогрева возьмём самый маленький датасет (Iris) и самую быструю комбинацию:
warmup_data = IrisDataset()
X_warmup = warmup_data.get_X()
y_warmup = warmup_data.get_y()

# Масштабируем данные
X_warmup_scaled = minmax_scale(X_warmup)

# Разделяем на train/test
X_warmup_train, X_warmup_test, y_warmup_train, y_warmup_test = train_test_split(
    X_warmup_scaled, y_warmup, test_size=0.1, random_state=42
)

# Создаём модель для «прогрева»
warmup_model = MLPEAClassifier(
    iters=30,
    pop_size=30,
    hidden_layers=(5,),
    weights_optimizer=SHAGA,
)
# Прогоняем fit / predict, чтобы скомпилировать (jit) то, что нужно
warmup_model.fit(X_warmup_train, y_warmup_train)
_ = warmup_model.predict(X_warmup_test)

# -- 2. ОСНОВНОЙ ЭКСПЕРИМЕНТ -------------------------------------------------- #

datasets = [IrisDataset(), WineDataset(), DigitsDataset()]

# Пары (iters, pop_size)
iters_pop_pairs = [(30, 30), (200, 200)]

# Список вариантов скрытых слоёв
hidden_layers_list = [(5,), (10, 10), (100,), (100, 100)]

results = []  # Список для сохранения результатов всех прогонов

# Подсчитаем общее количество экспериментов
total_experiments = len(datasets) * len(iters_pop_pairs) * len(hidden_layers_list)

with tqdm(total=total_experiments, desc="Всего экспериментов") as pbar:
    for data in datasets:
        X = data.get_X()
        y = data.get_y()

        # Масштабируем данные
        X_scaled = minmax_scale(X)

        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42
        )

        # Перебираем пары (iters, pop_size)
        for iters, pop_size in iters_pop_pairs:

            # Перебираем все варианты архитектуры скрытых слоёв
            for hidden_layers in hidden_layers_list:

                # Создаём модель
                model = MLPEAClassifier(
                    iters=iters,
                    pop_size=pop_size,
                    hidden_layers=hidden_layers,
                    weights_optimizer=SHAGA,
                )

                # Засекаем время обучения
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                exec_time = time.time() - start_time

                # Метрики
                cm = confusion_matrix(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                # Сохраняем результаты в общий список
                results.append(
                    {
                        "Dataset": data.__class__.__name__,
                        "Iters": iters,
                        "Pop Size": pop_size,
                        "Hidden Layers": hidden_layers,
                        "Time (s)": round(exec_time, 3),
                        "Confusion Matrix": cm.tolist(),  # Можно сохранить как список
                        "F1 Score (macro)": round(f1, 4),
                    }
                )

                # Обновляем прогресс-бар после каждого эксперимента
                pbar.update(1)

# Формируем DataFrame из списка словарей
df_results = pd.DataFrame(results)

# Сохраняем в Excel (без индексов)
df_results.to_excel("results.xlsx", index=False)

print("Эксперименты завершены. Результаты сохранены в 'results.xlsx'.")
