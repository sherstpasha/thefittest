# ==== imports & setup ====
from thefittest.optimizers import GeneticProgramming
from thefittest.benchmarks import DigitsDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.base._tree import init_net_uniset

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# воспроизводимость
np.random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)
dtype = torch.float32

# ==== данные ====
data = DigitsDataset()
X = data.get_X()  # (150, 4)
y = data.get_y().astype(np.int64)  # классы 0..2

X_scaled = minmax_scale(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---- тензоры ----
Xtr_t = torch.as_tensor(X_train, dtype=dtype, device=device)
ytr_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
Xte_t = torch.as_tensor(X_test, dtype=dtype, device=device)
yte_t = torch.as_tensor(y_test, dtype=torch.long, device=device)

# ==== генерим архитектуру через GP (одна особь) ====
model = GeneticProgrammingNeuralNetClassifier(n_iter=1, pop_size=1)  # тут GP только ради формы сети
uniset = init_net_uniset(
    n_variables=X.shape[1],
    input_block_size=3,
    max_hidden_block_size=8,
    offset=True,
)

pop = GeneticProgramming.half_and_half(1, uniset, 15)

# строим сеть с ЛИНЕЙНЫМ выходом (логиты)
net = model.genotype_to_phenotype_tree(
    tree=pop[0],
    n_variables=X.shape[1],
    n_outputs=10,
    output_activation="ln",  # <--- важно: без softmax
    offset=True,
)

# переносим на девайс/тип и компилим план
net.to(device=device, dtype=dtype)
net.compile_torch()

# ==== обучение весов ====
# делаем веса обучаемыми
net._weights.requires_grad_(True)

optimizer = torch.optim.Adam([net._weights], lr=1e-2, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

EPOCHS = 3000
PRINT_EVERY = 1
BATCH = None  # мини-батчи, можно поставить None для фулл-батча


def iterate_minibatches(Xt, yt, batch):
    if batch is None or batch >= Xt.shape[0]:
        yield Xt, yt
        return
    idx = torch.randperm(Xt.shape[0], device=Xt.device)
    for i in range(0, Xt.shape[0], batch):
        j = idx[i : i + batch]
        yield Xt[j], yt[j]


@torch.no_grad()
def evaluate():
    # логиты -> метрики
    logits_tr = net.forward(Xtr_t)
    loss_tr = criterion(logits_tr, ytr_t).item()
    pred_tr = logits_tr.argmax(dim=1).detach().cpu().numpy()
    acc_tr = accuracy_score(y_train, pred_tr)
    f1_tr = f1_score(y_train, pred_tr, average="macro")

    logits_te = net.forward(Xte_t)
    loss_te = criterion(logits_te, yte_t).item()
    pred_te = logits_te.argmax(dim=1).detach().cpu().numpy()
    acc_te = accuracy_score(y_test, pred_te)
    f1_te = f1_score(y_test, pred_te, average="macro")
    return (loss_tr, acc_tr, f1_tr), (loss_te, acc_te, f1_te)


for epoch in range(1, EPOCHS + 1):
    net._weights.grad = None
    total_loss = 0.0

    for xb, yb in iterate_minibatches(Xtr_t, ytr_t, BATCH):
        logits = net.forward(xb)  # (B, 3), логиты
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.shape[0]

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        (l_tr, a_tr, f_tr), (l_te, a_te, f_te) = evaluate()
        print(
            f"epoch {epoch:4d} | train loss {l_tr:.4f} acc {a_tr:.3f} f1 {f_tr:.3f} "
            f"| test loss {l_te:.4f} acc {a_te:.3f} f1 {f_te:.3f}"
        )

# ==== финальная оценка + матрица ошибок ====
with torch.no_grad():
    logits_te = net.forward(Xte_t)
    y_pred = logits_te.argmax(dim=1).cpu().numpy()

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_pred))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Test F1-macro:", f1_score(y_test, y_pred, average="macro"))


import matplotlib.pyplot as plt

net.plot()
plt.show()
