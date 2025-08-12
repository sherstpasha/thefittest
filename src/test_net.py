# ==== imports & setup ====
from thefittest.optimizers import GeneticProgramming
from thefittest.benchmarks import IrisDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.base._tree import init_net_uniset

import numpy as np
import torch

# для воспроизводимости (можно убрать)
np.random.seed(0)
torch.manual_seed(0)

# ==== данные (готовим один раз) ====
data = IrisDataset()
X = data.get_X()                   # (150, 4)
y = data.get_y()
from sklearn.preprocessing import minmax_scale
X_scaled = minmax_scale(X)
# тензор для новой сети (CPU, float32); автокаст в net.forward уже есть
X_t_base = torch.as_tensor(X_scaled, dtype=torch.float32, device="cpu")

# ==== модель и универсум (готовим один раз) ====
model = GeneticProgrammingNeuralNetClassifier(n_iter=500, pop_size=500)
uniset = init_net_uniset(
    n_variables=X.shape[1],
    input_block_size=3,
    max_hidden_block_size=8,
    offset=True,
)

# ==== функция одного прогона ====
def one_trial():
    # генерим популяцию и берём первый генотип
    pop = GeneticProgramming.half_and_half(100, uniset, 15)

    # строим сети из одного генотипа
    net = model.genotype_to_phenotype_tree(
        tree=pop[0], n_variables=X.shape[1], n_outputs=3,
        output_activation='softmax', offset=True
    )
    net_old = model.genotype_to_phenotype_tree_old(
        tree=pop[0], n_variables=X.shape[1], n_outputs=3,
        output_activation='softmax', offset=True
    )

    # синхронизируем веса
    net_old._weights = net._weights.detach().cpu().numpy().astype(np.float64)

    # прямой проход
    net.compile_torch()
    with torch.no_grad():
        y_new = net.forward(X_t_base)  # torch, (N, C)

    y_old_np = net_old.forward(X_scaled)               # numpy, (1, N, C)
    y_old = torch.from_numpy(y_old_np).squeeze(0).to(dtype=y_new.dtype, device=y_new.device)

    # метрики расхождения
    abs_diff = (y_new - y_old).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    ok = bool(max_abs < 1e-6)

    # сравнение предсказаний по argmax
    argmax_equal = bool(torch.equal(y_new.argmax(dim=1), y_old.argmax(dim=1)))

    return ok, max_abs, mean_abs, argmax_equal

# ==== многократный прогон ====
N_TRIALS = 10
ok_count = 0
argmax_eq_count = 0
worst = {"max_abs": 0.0, "mean_abs": 0.0, "trial": -1}
mean_abs_accum = 0.0
fails = []  # сохраним первые несколько не-OK

for t in range(N_TRIALS):
    try:
        ok, mx, mn, argmax_eq = one_trial()
    except Exception as e:
        # если вдруг что-то упало, считаем как fail и идём дальше
        ok, mx, mn, argmax_eq = False, float("inf"), float("inf"), False
        fails.append({"trial": t, "error": repr(e)})

    ok_count += int(ok)
    argmax_eq_count += int(argmax_eq)
    mean_abs_accum += (0.0 if not np.isfinite(mn) else mn)

    if mx > worst["max_abs"]:
        worst = {"max_abs": mx, "mean_abs": mn, "trial": t}

    if not ok and len(fails) < 5 and "error" not in (fails[-1] if fails else {}):
        fails.append({"trial": t, "max_abs": mx, "mean_abs": mn})

# ==== сводка ====
print("\n=== Summary over", N_TRIALS, "trials ===")
print("OK (max_abs < 1e-6):", ok_count, f"({ok_count/N_TRIALS:.2%})")
print("Argmax equal:       ", argmax_eq_count, f"({argmax_eq_count/N_TRIALS:.2%})")
print("Worst case:         ", worst)
print("Mean of mean_abs:   ", mean_abs_accum / N_TRIALS)

if fails:
    print("\nExamples of failures (up to 5):")
    for f in fails[:5]:
        print(f)
else:
    print("\nNo failures detected.")
