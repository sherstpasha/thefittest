import numpy as np
from thefittest.optimizers import SHADE
from sklearn.utils.estimator_checks import check_estimator

import numpy as np

from thefittest.optimizers import SHADE
from thefittest.benchmarks import BanknoteDataset, IrisDataset, DigitsDataset
from thefittest.classifiers import MLPEAClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

data = DigitsDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)


# Создание и обучение модели
model = MLPEAClassifier(
    pop_size=500, hidden_layers=(100,), weights_optimizer_args={"show_progress_each": 10}
)  # Замените YourModelClass на класс вашей модели
model.fit(X_train, y_train)

predictions_array = model.predict(X_test)

# Предсказания по одному элементу
predictions_individual = np.array([model.predict(np.array([x]))[0] for x in X_test])


print(np.testing.assert_array_equal(predictions_array, predictions_individual))

print(f1_score(y_test, predictions_array, average="macro"))
