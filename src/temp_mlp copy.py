from thefittest.optimizers import SHADE
from thefittest.benchmarks import DigitsDataset
from thefittest.classifiers import GeneticProgrammingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, f1_score

import torch

from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Iris dataset
data = DigitsDataset()
X = data.get_X()
y = data.get_y()

# Scale features to the [0, 1] range
X_scaled = minmax_scale(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)


# Initialize the MLPEAClassifier with SHAGA as the optimizer
model = GeneticProgrammingClassifier(
    n_iter=100,
    pop_size=500,
    optimizer_args={"show_progress_each": 10},
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predict = model.predict(X_test)

# Evaluate the model
print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))
