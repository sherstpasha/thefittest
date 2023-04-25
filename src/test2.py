import numpy as np
from thefittest.classifiers import MLPClassifierEA
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


data = load_digits()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


model = MLPClassifierEA(iters=300, 
                        pop_size=300,
                        hidden_layers=(150,),
                        show_progress_each=10)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

print(f1_score(y_test, y_pred, average='micro'))
print(confusion_matrix(y_test, y_pred))
