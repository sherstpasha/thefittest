import os
from typing import Dict
from typing import Union

import numpy as np
from numpy.typing import NDArray


path = os.path.dirname(__file__) + "/_data/"

iris_data = np.loadtxt(path + "iris_data.txt", delimiter=",")
wine_data = np.loadtxt(path + "wine_data.txt", delimiter=",")
breast_data = np.loadtxt(path + "breast_cancer_data.txt", delimiter=",")
digits_data = np.loadtxt(path + "handwritten_digits_data.txt", delimiter=",")
credit_data = np.loadtxt(path + "credit_risk_data.txt", delimiter=",")
know_data = np.loadtxt(path + "user_knowledge_data.txt", delimiter=",")
banknote_data = np.loadtxt(path + "banknote_dataset.txt", delimiter=",")


class Dataset:
    def __init__(
        self,
        X: NDArray[np.float64],
        y: NDArray[Union[np.int64, np.float64]],
        X_names: Dict,
        y_names: Dict,
    ):
        self._X = X
        self._y = y
        self._X_names = X_names
        self._y_names = y_names

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

    def get_X_names(self):
        return self._X_names

    def get_y_names(self):
        return self._y_names


class IrisDataset(Dataset):
    """Fisher R. A.. (1988). Iris. UCI Machine Learning Repository.
    https://doi.org/10.24432/C56C76."""

    def __init__(self):
        Dataset.__init__(
            self,
            X=iris_data[:, :-1].astype(np.float64),
            y=iris_data[:, -1].astype(np.int64),
            X_names={
                0: "sepal length in cm",
                1: "sepal width in cm",
                2: "petal length in cm",
                3: "petal width in cm",
            },
            y_names={0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"},
        )


class WineDataset(Dataset):
    """Aeberhard,Stefan and Forina,M.. (1991). Wine. UCI Machine
    Learning Repository. https://doi.org/10.24432/C5PC7J."""

    def __init__(self):
        Dataset.__init__(
            self,
            X=wine_data[:, 1:].astype(np.float64),
            y=wine_data[:, 0].astype(np.int64) - 1,
            X_names={
                0: "Alcohol",
                1: "Malic acid",
                2: "Ash",
                3: "Alcalinity of ash",
                4: "Magnesium",
                5: "Total phenols",
                6: "Flavanoids",
                7: "Nonflavanoid phenols",
                8: "Proanthocyanins",
                9: "Color intensity",
                10: "Hue",
                11: "OD280/OD315 of diluted wines",
                12: "Proline",
            },
            y_names={0: "class 1", 1: "class 2", 2: "class 3"},
        )


class BreastCancerDataset(Dataset):
    """
    Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995).
    Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
    https://doi.org/10.24432/C5DW2B."""

    def __init__(self):
        Dataset.__init__(
            self,
            X=breast_data[:, 2:].astype(np.float64),
            y=breast_data[:, 1].astype(np.int64),
            X_names={
                0: "mean radius",
                1: "mean texture",
                2: "mean perimeter",
                3: "mean area",
                4: "mean smoothness",
                5: "mean compactness",
                6: "mean concavity",
                7: "mean concave points",
                8: "mean symmetry",
                9: "mean fractal dimension",
                10: "radius error",
                11: "texture error",
                12: "perimeter error",
                13: "area error",
                14: "smoothness error",
                15: "compactness error",
                16: "concavity error",
                17: "concave points error",
                18: "symmetry error",
                19: "fractal dimension error",
                20: "worst radius",
                21: "worst texture",
                22: "worst perimeter",
                23: "worst area",
                24: "worst smoothness",
                25: "worst compactness",
                26: "worst concavity",
                27: "worst concave points",
                28: "worst symmetry",
                29: "worst fractal dimension",
            },
            y_names={0: "M", 1: "B"},
        )


class DigitsDataset(Dataset):
    """Alpaydin,E. and Kaynak,C.. (1998). Optical Recognition of Handwritten Digits.
    UCI Machine Learning Repository. https://doi.org/10.24432/C50P49."""

    def __init__(self):
        Dataset.__init__(
            self,
            X=digits_data[:, :-1].astype(np.float64),
            y=digits_data[:, -1].astype(np.int64),
            X_names=dict(zip(range(64), range(64))),
            y_names=dict(zip(range(10), range(10))),
        )


class CreditRiskDataset(Dataset):
    """https://www.kaggle.com/datasets/upadorprofzs/credit-risk"""

    def __init__(self):
        Dataset.__init__(
            self,
            X=credit_data[:, 1:-1].astype(np.float64),
            y=credit_data[:, -1].astype(np.int64),
            X_names={0: "income", 1: "age", 2: "loan"},
            y_names={0: "good client", 1: "bad client"},
        )


class UserKnowladgeDataset(Dataset):
    """Kahraman,Hamdi, Colak,Ilhami, and Sagiroglu,Seref. (2013). User Knowledge Modeling.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5231X."""

    def __init__(self):
        Dataset.__init__(
            self,
            X=know_data[:, :-1].astype(np.float64),
            y=know_data[:, -1].astype(np.int64),
            X_names={
                0: "STG (The degree of study time for goal object materails)",
                1: "SCG (The degree of repetition number of user for goal object materails) ",
                2: "STR (The degree of study time of user for related objects with goal object)",
                3: "LPR (The exam performance of user for related objects with goal object)",
                4: "PEG (The exam performance of user for goal objects)",
            },
            y_names={0: "Very Low", 1: "Low", 2: "Middle", 3: "High "},
        )


class BanknoteDataset(Dataset):
    """
    Lohweg,Volker. (2013). banknote authentication.
    UCI Machine Learning Repository. https://doi.org/10.24432/C55P57."""

    def __init__(self):
        Dataset.__init__(
            self,
            X=banknote_data[:, :-1].astype(np.float64),
            y=banknote_data[:, -1].astype(np.int64),
            X_names={
                0: "variance of Wavelet Transformed image (continuous)",
                1: "skewness of Wavelet Transformed image (continuous)",
                2: "curtosis of Wavelet Transformed image (continuous)",
                3: "entropy of image (continuous)",
            },
            y_names={0: "not original", 1: "original"},
        )
