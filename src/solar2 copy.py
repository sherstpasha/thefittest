import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, normaltest, anderson
from thefittest.benchmarks import SolarBatteryDegradationDataset
import os

# === Подготовка папки ===
os.makedirs("results", exist_ok=True)

# === Загрузка данных ===
dataset = SolarBatteryDegradationDataset()
X = dataset.get_X()
y = dataset.get_y()
target_names = list(dataset.get_y_names())

df_X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
df_y = pd.DataFrame(y, columns=target_names)

# === Информация о данных ===
print("\n🔹 X shape:", df_X.shape)
print("🔹 y shape:", df_y.shape)
print("\n=== X describe ===")
print(df_X.describe())
print("\n=== y describe ===")
print(df_y.describe())

print("\n=== Пропуски ===")
print(df_X.isnull().sum())
print(df_y.isnull().sum())

# === Корреляция признаков и целей ===
corr_pearson = pd.concat([df_X, df_y], axis=1).corr(method="pearson")
corr_spearman = pd.concat([df_X, df_y], axis=1).corr(method="spearman")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_pearson, cmap="coolwarm", annot=False)
plt.title("Матрица корреляций (Pearson)")
plt.tight_layout()
plt.savefig("results/corr_pearson.png")
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_spearman, cmap="coolwarm", annot=False)
plt.title("Матрица корреляций (Spearman)")
plt.tight_layout()
plt.savefig("results/corr_spearman.png")
plt.close()

# === VIF для оценки мультиколлинеарности ===
vif_data = pd.DataFrame()
vif_data["feature"] = df_X.columns
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print("\n=== VIF ===")
print(vif_data)

# === Распределение целей ===
for col in df_y.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_y[col], kde=True)
    plt.title(f"Распределение: {col}")
    plt.tight_layout()
    plt.savefig(f"results/hist_{col}.png")
    plt.close()

    # Проверка на нормальность
    stat, p = shapiro(df_y[col])
    print(f"\nShapiro-Wilk тест для {col}: W={stat:.4f}, p={p:.4e}")

# === PCA-анализ ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c="b", alpha=0.6)
plt.title("PCA (2 компоненты)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/pca_scatter.png")
plt.close()

print("\n=== Доля объяснённой дисперсии (PCA) ===")
print(pca.explained_variance_ratio_)

# === Pairplot признаков (если < 10) ===
if X.shape[1] <= 10:
    df_small = pd.concat([df_X, df_y], axis=1)
    sns.pairplot(df_small)
    plt.savefig("results/pairplot.png")
    plt.close()
