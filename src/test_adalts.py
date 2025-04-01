import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, f1_score

from thefittest.optimizers import SHADE, SelfCGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.classifiers._gpnneclassifier_one_tree import (
    GeneticProgrammingNeuralNetStackingClassifier,
)
from thefittest.tools.print import (
    print_net,
    print_tree,
    print_nets,
    print_trees,
    print_ens,
)


# ✅ Предобработка Adult Census с оригинального UCI
def preprocess_adult_dataset():
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    df_train = pd.read_csv(url_train, header=None, names=columns, skipinitialspace=True)
    df_test = pd.read_csv(url_test, header=None, names=columns, skipinitialspace=True, skiprows=1)

    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
    df_all = pd.concat([df_train, df_test], axis=0)

    df_all.replace("?", pd.NA, inplace=True)
    for col in ["workclass", "occupation", "native-country"]:
        df_all[col] = df_all[col].fillna("Unknown")

    df_all["income"] = df_all["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numeric_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    df_encoded = pd.get_dummies(df_all[categorical_features], drop_first=True).astype(int)
    df_final = pd.concat([df_all[numeric_features], df_encoded], axis=1)
    df_final["income"] = df_all["income"]

    feature_names = df_final.drop("income", axis=1).columns.tolist()

    X = df_final.drop("income", axis=1).values
    y = df_final["income"].values
    X_train = X[: len(df_train)]
    y_train = y[: len(df_train)]
    X_test = X[len(df_train) :]
    y_test = y[len(df_train) :]

    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)

    return X_train, X_test, y_train, y_test, feature_names


# ✅ Загрузка и подготовка
X_train, X_test, y_train, y_test, feature_names = preprocess_adult_dataset()

# Сохраняем названия признаков
with open("feature_names.txt", "w") as f:
    f.write("\n".join(feature_names))

# ✅ Обучение модели
model = GeneticProgrammingNeuralNetStackingClassifier(
    iters=10,
    pop_size=10,
    optimizer=PDPSHAGP,
    optimizer_args={"show_progress_each": 1, "n_jobs": 1, "max_level": 20, "no_increase_num": 30},
    weights_optimizer=SHADE,
    weights_optimizer_args={"iters": 500, "pop_size": 500, "no_increase_num": 30},
)

model.fit(X_train, y_train)

predict = model.predict(X_test)
optimizer = model.get_optimizer()

trees = optimizer.get_fittest()["genotype"]
nets = optimizer.get_fittest()["phenotype"]

# ✅ Оценка
print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))

# ✅ Визуализация
print(nets._nets)
print_nets(nets._nets)
plt.savefig("print_nets.png")
plt.close()

print_ens(nets)
plt.savefig("print_ens.png")
plt.close()
