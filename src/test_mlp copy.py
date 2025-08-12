from thefittest.optimizers import SHAGA
from thefittest.benchmarks import DigitsDataset
from thefittest.classifiers import MLPEAClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, f1_score

import numpy as np
import torch, time

# --- reproducibility ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- data ---
data = DigitsDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)
X_train_np, X_test_np, y_train_np, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=SEED, stratify=y
)


def to_device(x_np, device):
    return torch.as_tensor(x_np, dtype=torch.float32, device=device)


# --- experiment helper ---
def run_experiment(device: str, hidden_layers):
    print(f"\n=== device={device}, hidden_layers={hidden_layers} ===")
    X_train = to_device(X_train_np, device)
    X_test = to_device(X_test_np, device)
    # y_train можно оставить как float тензор: внутри fit ты уже энкодишь метки
    y_train = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)

    model = MLPEAClassifier(
        n_iter=100,
        pop_size=100,
        hidden_layers=hidden_layers,
        weights_optimizer=SHAGA,
        weights_optimizer_args={"show_progress_each": 10},
        random_state=SEED,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)  # под капотом сам разберётся с torch/np
    pred_time = time.perf_counter() - t1

    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("confusion_matrix:\n", cm)
    print("f1_macro:", f1)
    print(f"fit_time: {fit_time:.3f}s | predict_time: {pred_time:.3f}s")
    return {
        "device": device,
        "hidden_layers": str(hidden_layers),
        "f1_macro": float(f1),
        "fit_time_s": float(fit_time),
        "predict_time_s": float(pred_time),
    }


# --- run 4 experiments ---
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
hidden_sets = [[5, 5], [100, 100]]

results = []
for hl in hidden_sets:
    for dev in devices:
        results.append(run_experiment(dev, hl))

print("\n=== summary ===")
for r in results:
    print(r)
