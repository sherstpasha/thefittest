from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

# Ваш массив меток
labels = [1, 2, 1, 1, 2, 1]

# Преобразование меток в числовой формат
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Преобразование числовых меток в one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
onehot_labels = onehot_encoder.fit_transform(np.array(numeric_labels).reshape(-1, 1))

print(onehot_labels)
