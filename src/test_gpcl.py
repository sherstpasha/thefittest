# Импорт необходимых библиотек
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from thefittest.classifiers._symbolicclassificationgp import SymbolicClassificationGP
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Загрузка датасета Breast Cancer Wisconsin
data = load_breast_cancer()
X = data.data      # признаки
y = data.target    # бинарная целевая переменная

# Разбиение данных на обучающую и тестовую выборки (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели KNN для классификации (например, с 5 соседями)
classifier = SymbolicClassificationGP(iters=100, pop_size=100)

# Обучение модели
classifier.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = classifier.predict(X_test)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy (точность):", accuracy)
print("Confusion Matrix (матрица ошибок):\n", conf_matrix)
print("Classification Report (отчет по классификации):\n", class_report)
