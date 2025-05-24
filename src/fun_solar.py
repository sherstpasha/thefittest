
from thefittest.fl2 import FCSelfCGA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# data_train = pd.read_csv('6 class csv.csv', delimiter = ',')
# X = data_train.iloc[:,:-1]
# y = data_train.iloc[:,-1]
# X = pd.get_dummies(X, columns = ['Star color'])
# data_train = np.loadtxt('G:/My Drive/5.Документы/1. Публикации/РИНЦ/АПАК. 22. Построение базы нечетких правил для распознавания пульсаров с помощью самоконфигурируемого генетического алгоритма/HTRU_2.csv', delimiter = ',')
# X = data_train[:,:-1]
# y = data_train[:,-1]

from thefittest.benchmarks import SolarBatteryDegradationDataset

data = SolarBatteryDegradationDataset()
X = data.get_X()
y = data.get_y

target_names = data.get_X_names()
feature_names = data.get_y_names()

# feature_names = np.array(['Mean of the integrated profile',
#                           'Standard deviation of the integrated profile',
#                           'Excess kurtosis of the integrated profile',
#                           'Skewness of the integrated profile',
#                           'Mean of the DM-SNR curve',
#                           'Standard deviation of the DM-SNR curve',
#                           'Excess kurtosis of the DM-SNR curve',
#                           'Skewness of the DM-SNR curve'])

# feature_names = np.array(['Среднее значение ИП',
#                           'Стандартное отклонение ИП',
#                           'Избыточный эксцесс ИП',
#                           'Асимметрия ИП',
#                           'Среднее значение DM-SNR',
#                           'Стандартное отклонение DM-SNR',
#                           'Избыточный эксцесс кривой DM-SNR',
#                           'Асимметрия кривой DM-SNR'])

# target_names = np.array(['notPulsar', 'Pulsar'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

acc = []
stat = []
for i in range(30):
    model = FCSelfCGA(iters=200, pop_size = 100, n_fsets = 3, n_rules = 2,
                      rl=0.1, bl = 0.1, tour_size = 5)
    model.fit(X, y)
    base = model.print_rules(set_names = )
    # stats = model.opt_model.stats
    # list_ = model.list_

    y_pred = model.predict(X_test)

    acc.append(f1_score(y_test, y_pred, average='macro'))    
    stat.append(model.opt_model.stats)
    print(i, f1_score(y_test, y_pred, average='macro'))