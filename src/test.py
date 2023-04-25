import numpy as np
from thefittest.benchmarks.symbolicregression17 import problems_dict
from thefittest.regressors import SymbolicRegressionGP
from thefittest.optimizers import GeneticProgramming
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


function = problems_dict['F1']['function']
left_border = problems_dict['F1']['bounds'][0]
right_border = problems_dict['F1']['bounds'][1]
sample_size = 300
n_dimension = problems_dict['F1']['n_vars']

X = np.array([np.linspace(left_border, right_border, sample_size)
              for _ in range(n_dimension)]).T
y = function(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


model = SymbolicRegressionGP(iters=300,
                             pop_size=500,
                             show_progress_each=10)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(y_pred)
print(model.optimizer.get_fittest().get()[-1])
print(r2_score(y_test, y_pred))
