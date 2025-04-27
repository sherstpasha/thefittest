import os
import time  # <--- добавили
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from thefittest.fuzzy import FuzzyRegressor

# from thefittest.fuzzy_gpu import FuzzyRegressorTorch
import warnings
import pickle
import pandas as pd

# Папка для сохранения результатов
output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results_regressor"

os.makedirs(output_dir, exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

# 1. Генерируем синтетические данные
np.random.seed(42)
n_samples = 200
X = np.random.uniform(-5, 5, size=(n_samples, 2))
y1 = 2 * X[:, 0] + 0.5 * np.random.randn(n_samples)
y2 = -1.5 * X[:, 1] + 0.3 * np.random.randn(n_samples)
y3 = X[:, 0] * X[:, 1] + 0.2 * np.random.randn(n_samples)
Y = np.vstack([y1, y2, y3]).T

# 2. Разбиение
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

labels5 = ["очень маленькое", "маленькое", "среднее", "большое", "очень большое"]
feature_names = ["X1", "X2"]
set_names = {name: labels5 for name in feature_names}
target_names = ["Y1", "Y2", "Y3"]
target_set_names = {name: labels5 for name in target_names}

# 3. Инициализация
model = FuzzyRegressor(
    iters=300,
    pop_size=500,
    n_features_fuzzy_sets=[5, 5],
    n_target_fuzzy_sets=[5, 5, 5],
    max_rules_in_base=12,
    target_grid_volume=100,
)

model.define_sets(
    X_train,
    Y_train,
    feature_names=feature_names,
    set_names=set_names,
    target_names=target_names,
    target_set_names=target_set_names,
)

# 3.5 Замер времени обучения
start_time = time.time()
model.fit(X_train, Y_train)
train_time = time.time() - start_time

# 4. Предсказание и оценка
Y_pred = model.predict(X_test)
r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(3)]
print("R2 scores per output:", r2_scores)
print("Average R2:", np.mean(r2_scores))

# 5. Сохранение результатов
rule_base_path = os.path.join(output_dir, "rule_base.txt")
model_path = os.path.join(output_dir, "fuzzy_model.pkl")
preds_path = os.path.join(output_dir, "predictions.csv")

# 5.1 Правила
rules_text = model.get_text_rules(print_intervals=True)
with open(rule_base_path, "w", encoding="utf-8") as f:
    f.write(rules_text)

# 5.2 Модель
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# 5.3 DataFrame с результатами
df = pd.DataFrame(X_test, columns=feature_names)
df[["Y1_true", "Y2_true", "Y3_true"]] = Y_test
df[["Y1_pred", "Y2_pred", "Y3_pred"]] = Y_pred
df.to_csv(preds_path, index=False)

# 6. Финальный вывод
print(f"Training time: {train_time:.2f} seconds")
print("Все файлы успешно сохранены в", output_dir)
print(" -", rule_base_path)
print(" -", model_path)
print(" -", preds_path)
