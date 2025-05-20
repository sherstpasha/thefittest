import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from thefittest.benchmarks import BanknoteDataset
from thefittest.optimizers._selfcshaga import SelfCSHAGA
from thefittest.fl2 import FCSelfCGA  # Убедитесь, что путь корректный

# ---------------------------
# Параметры запуска
# ---------------------------
iters = 300
pop_size = 300
n_fsets = 7
n_rules = 20
random_seed = 1

# ---------------------------
# Загрузка и подготовка данных
# ---------------------------
X, y = BanknoteDataset().get_X(), BanknoteDataset().get_y()
X = X.astype(np.float32)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, random_state=random_seed
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Обучение и оценка
# ---------------------------
clf = FCSelfCGA(
    iters=iters,
    pop_size=pop_size,
    n_fsets=n_fsets,
    n_rules=n_rules,
    optimizer=SelfCSHAGA,
)

clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

train_f1 = f1_score(y_train, y_pred_train, average="macro")
test_f1 = f1_score(y_test, y_pred_test, average="macro")

print(f"SelfCSHAGA on BanknoteDataset -> Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
