import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from thefittest.optimizers._cshagp import CSHAGP
from thefittest.optimizers import SelfCGP, PDPGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.regressors._symbolicregressiongp_dual import SymbolicRegressionGP_DUAL
from scipy.stats import ttest_ind
import warnings
import traceback
warnings.filterwarnings("ignore")

# Настройки
number_of_iterations = 1000
population_size = 100
num_runs = 1  # Только один запуск

data_folder = r"C:\Users\USER\Desktop\Feynman120\Feynman120"
files_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
methods = ["SelfCGP", "PDPGP", "SelfCSHAGP", "PDPSHAGP"]
abbr = {"SelfCGP": "SCGP", "PDPGP": "PDPG", "SelfCSHAGP": "SCSH", "PDPSHAGP": "PDPSH"}

results = []

for file_name in files_list[:2]:  # Для отладки можно ограничить одним файлом
    for iteration in range(1, num_runs + 1):
        for method in methods:
            try:
                print(f"Running {file_name}, iteration {iteration}, method {method}")
                file_path = os.path.join(data_folder, file_name)
                data = np.loadtxt(file_path).astype(np.float32)

                X = data[:1000, :-1]
                y = data[:1000, -1]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=0.75, random_state=iteration
                )

                if method == "CSHAGP":
                    optimizer_class = CSHAGP
                elif method == "SelfCGP":
                    optimizer_class = SelfCGP
                elif method == "PDPGP":
                    optimizer_class = PDPGP
                elif method == "SelfCSHAGP":
                    optimizer_class = SelfCSHAGP
                elif method == "PDPSHAGP":
                    optimizer_class = PDPSHAGP
                else:
                    raise ValueError(f"Unknown method: {method}")

                model = SymbolicRegressionGP_DUAL(
                    iters=number_of_iterations,
                    pop_size=population_size,
                    optimizer=optimizer_class,
                    optimizer_args={"elitism": False, "keep_history": True},
                )
                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)

                results.append({
                    "iteration": iteration,
                    "file": file_name,
                    "method": method,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                })

                print(f"Done: test_rmse = {test_rmse:.4f}, test_r2 = {test_r2:.4f}")

            except Exception as e:
                print(f"Error in {file_name}, method {method}: {str(e)}")
                traceback.print_exc()

# Преобразуем результаты в DataFrame и сохраняем
if results:
    df_results = pd.DataFrame(results)
    os.makedirs("results_debug", exist_ok=True)
    df_results.to_csv("results_debug/debug_results.csv", index=False)
    print("Результаты сохранены в 'results_debug/debug_results.csv'")
else:
    print("Нет результатов. Возможно, произошли ошибки при всех запусках.")
