# Импорт необходимых библиотек
from thefittest.benchmarks import CombinedCycleDataset
from sklearn.model_selection import train_test_split
from thefittest.regressors._symbolicregressiongp_dual import SymbolicRegressionGP_DUAL
from thefittest.regressors._symbolicregressiongp import SymbolicRegressionGP
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    r2_score,
    root_mean_squared_error,
)

# Загрузка датасета Breast Cancer Wisconsin
data = CombinedCycleDataset()
X = data.get_X()  # признаки
y = data.get_y()  # бинарная целевая переменная

# Разбиение данных на обучающую и тестовую выборки (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели KNN для классификации (например, с 5 соседями)
classifier = SymbolicRegressionGP_DUAL(
    iters=1000, pop_size=100, optimizer_args={"show_progress_each": 1}
)

# classifier = SymbolicRegressionGP(
#     iters=100, pop_size=1000, optimizer_args={"show_progress_each": 1}
# )

# Обучение модели
classifier.fit(X_train, y_train)

# # Предсказание на тестовой выборке
y_pred = classifier.predict(X_test)

print(y_pred)

# Оценка качества модели
accuracy = r2_score(y_test, y_pred)
conf_matrix = root_mean_squared_error(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

print("Accuracy (точность):", accuracy)
print("Confusion Matrix (матрица ошибок):\n", conf_matrix)
# print("Classification Report (отчет по классификации):\n", class_report)
