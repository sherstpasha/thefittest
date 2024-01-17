import numpy as np
from thefittest.utils._metrics import f1_score2d

# Generate example data
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
y_predict2d = np.array(
    [[0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 1, 2, 1, 0], [0, 1, 2, 0, 1, 2, 1, 2, 0]],
    dtype=np.int64,
)

# Calculate F1 score for each prediction
print("F1 scores for each prediction:", f1_score2d(y_true, y_predict2d))
