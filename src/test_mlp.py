from thefittest.optimizers import SHAGA
from thefittest.benchmarks import IrisDataset
from thefittest.classifiers import MLPEAClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, f1_score

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Iris dataset
data = IrisDataset()
X = data.get_X()
y = data.get_y()

# Scale features to the [0, 1] range
X_scaled = minmax_scale(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)

X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
y_test = y_test


# Initialize the MLPEAClassifier with SHAGA as the optimizer
model = MLPEAClassifier(
    n_iter=100,
    pop_size=100,
    hidden_layers=[100, 100],
    weights_optimizer=SHAGA,
    weights_optimizer_args={"show_progress_each": 10},
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predict = model.predict(X_test)

# Evaluate the model
print("confusion_matrix: \n", confusion_matrix(y_test, predict))
print("f1_score: \n", f1_score(y_test, predict, average="macro"))
