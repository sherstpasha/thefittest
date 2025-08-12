import numpy as np
import torch
from thefittest.base._net import Net, Net_old

def _sorted_idx_map(sorted_ids, other_ids):
    pos = {n:i for i,n in enumerate(sorted_ids)}
    return [pos[n] for n in other_ids]

@torch.no_grad()
def compare_old_new_once(
    inputs: set[int],
    hidden_layers: list[set[int]],
    outputs: set[int],
    connects: np.ndarray,
    activs: dict[int,int],
    B: int = 8,
    Wcases: int | None = None,
    device: str | torch.device = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict:
    """
    Возвращает словарь с метриками расхождения.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    E = int(connects.shape[0])
    # один и тот же набор весов
    if Wcases is None:
        w_np = np.random.randn(E).astype(np.float64)
        w_t  = torch.tensor(w_np, dtype=torch.float32, device=device)
    else:
        w_np = np.random.randn(Wcases, E).astype(np.float64)
        w_t  = torch.tensor(w_np, dtype=torch.float32, device=device)

    # входы делаем в КАНОНИЧЕСКОМ порядке = sorted(inputs)
    in_sorted = sorted(inputs)
    X_base = torch.randn(B, len(in_sorted), device=device, dtype=torch.float32)

    # --- NEW ---
    net_new = Net(inputs=inputs, hidden_layers=hidden_layers, outputs=outputs,
                  connects=connects, weights=(w_t if Wcases is None else torch.randn(E, device=device)),
                  activs=activs)
    net_new.compile_torch()
    y_new = net_new.forward(X_base, weights=(None if Wcases is None else w_t))  # (B,n_out) or (W,B,n_out)

    # --- OLD ---
    net_old = Net_old(inputs=inputs, hidden_layers=hidden_layers, outputs=outputs,
                      connects=connects, weights=(w_np if Wcases is None else np.random.randn(E)),
                      activs=activs)
    # старый forward сам вызовет _get_order при первом запуске, но нам нужно знать его внутренний порядок
    _ = net_old.forward(np.zeros((1, len(in_sorted)), dtype=np.float64))  # прогреем порядок
    old_in_order  = list(net_old._numpy_inputs)      # порядок входов у старой модели
    old_out_order = list(net_old._numpy_outputs)     # порядок выходов у старой модели

    # переставим столбцы X_base под порядок старой
    cols_for_old = _sorted_idx_map(in_sorted, old_in_order)
    X_old = X_base[:, cols_for_old].cpu().numpy().astype(np.float64)

    # прогоним старую (с теми же весами)
    y_old = net_old.forward(X_old, weights=(w_np if Wcases is not None else None))  # (B,n_out) или (W,B,n_out)
    # приведём к torch и к порядку ВЫХОДОВ как у новой (у новой — sorted(outputs))
    out_sorted = sorted(outputs)
    cols_old2sorted = _sorted_idx_map(old_out_order, out_sorted)
    if y_old.ndim == 2:     # (B,n_out)
        y_old_sorted = torch.from_numpy(y_old[:, cols_old2sorted]).to(device=device, dtype=y_new.dtype)
    else:                   # (W,B,n_out)
        y_old_sorted = torch.from_numpy(y_old[:, :, cols_old2sorted]).to(device=device, dtype=y_new.dtype)

    # --- сравнение ---
    diff = (y_new - y_old_sorted).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ok = torch.allclose(y_new, y_old_sorted, atol=atol, rtol=rtol)

    return {
        "ok": ok,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "shape_new": tuple(y_new.shape),
        "shape_old": tuple(y_old_sorted.shape),
    }
# 1) ромб + скипы
inputs  = {0}
h1, h2  = {1,2}, {3,4,5}
outputs = {6,7}
connects = np.array([
    [0,1],[0,2],
    [1,3],[1,4],
    [2,4],[2,5],
    [0,5],          # skip
    [3,6],[4,6],[5,7],
    [0,7],          # skip
], dtype=np.int64)
RELU, TANH, LN = 1, 3, 4
activs = {1:RELU,2:TANH,3:TANH,4:RELU,5:TANH,6:LN,7:LN}
print("diamond:", compare_old_new_once(inputs,[h1,h2],outputs,connects,activs, B=8))

# 2) softmax-группа (два узла с одинаковым множеством родителей)
SOFTMAX=5
inputs  = {0,1}
h1      = {2,3}
outputs = {4}
connects = np.array([
    [0,2],[1,2],
    [0,3],[1,3],   # одинаковые родители → одна группа softmax
    [2,4],[3,4],
], dtype=np.int64)
activs = {2:SOFTMAX, 3:SOFTMAX, 4:LN}
print("softmax:", compare_old_new_once(inputs,[h1],outputs,connects,activs, B=5))

# 3) MLP 1-16-16-1 с батчем весов
def build_mlp_graph(n_in, hidden, n_out, hidden_act=TANH, out_act=LN):
    node=0
    inputs=set(range(node,node+n_in)); node+=n_in
    hs=[]
    for h in hidden:
        layer=set(range(node,node+h)); node+=h
        hs.append(layer)
    outputs=set(range(node,node+n_out)); node+=n_out
    layers=[inputs]+hs+[outputs]
    edges=[]
    for L,R in zip(layers[:-1],layers[1:]):
        for u in L:
            for v in R:
                edges.append((u,v))
    connects=np.array(edges,dtype=np.int64)
    activs={}
    for L in hs:
        for u in L: activs[u]=hidden_act
    for u in outputs: activs[u]=out_act
    return inputs,hs,outputs,connects,activs

inputs, hs, outputs, connects, activs = build_mlp_graph(1,[16,16],1)
print("mlp W=3:", compare_old_new_once(inputs,hs,outputs,connects,activs, B=7, Wcases=3))
