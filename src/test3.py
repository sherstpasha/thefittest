from sklearn.svm import SVC
from thefittest.classifiers import MLPEAClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
                                                    
pipe = Pipeline([('scaler', StandardScaler()), ('mlpeaclassifier', MLPEAClassifier())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
print(pipe.fit(X_train, y_train).score(X_test, y_test))
# An estimator's parameter can be set using '__' syntax
print(pipe.set_params(mlpeaclassifier__offset=False).fit(X_train, y_train).score(X_test, y_test))