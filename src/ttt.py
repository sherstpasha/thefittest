import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split


def preprocess_adult_dataset():
    # Ссылки на оригинальные train/test
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    # Названия колонок (по документации UCI)
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

    # Загружаем данные
    df_train = pd.read_csv(url_train, header=None, names=columns, skipinitialspace=True)
    df_test = pd.read_csv(url_test, header=None, names=columns, skipinitialspace=True, skiprows=1)

    # Удаляем точки из income в тесте
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

    # Объединяем для общей предобработки
    df_all = pd.concat([df_train, df_test], axis=0)

    # Обработка пропусков
    df_all.replace("?", pd.NA, inplace=True)
    for col in ["workclass", "occupation", "native-country"]:
        df_all[col] = df_all[col].fillna("Unknown")

    # Целевая переменная
    df_all["income"] = df_all["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    # Категориальные и числовые признаки
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

    # One-hot encoding
    df_encoded = pd.get_dummies(df_all[categorical_features], drop_first=True).astype(int)

    df_final = pd.concat([df_all[numeric_features], df_encoded], axis=1)
    df_final["income"] = df_all["income"]

    # Разделяем обратно по длине
    X = df_final.drop("income", axis=1).values
    y = df_final["income"].values
    X_train = X[: len(df_train)]
    y_train = y[: len(df_train)]
    X_test = X[len(df_train) :]
    y_test = y[len(df_train) :]

    # Масштабируем
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)

    return X_train, X_test, y_train, y_test, df_final


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = preprocess_adult_dataset()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("\nПример строки:")
    print(df.head(1).T)
