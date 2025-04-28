from typing import Optional, Union, List, Dict
import numpy as np
import torch
from sklearn.metrics import r2_score
from thefittest.tools.transformations import SamplingGrid
from thefittest.optimizers._selfcshaga import SelfCSHAGA


def to_tensor(x, device):
    return torch.from_numpy(x).to(device)


class FuzzyBaseTorch:
    def __init__(
        self,
        iters: int,
        pop_size: int,
        n_features_fuzzy_sets: List[int],
        max_rules_in_base: int,
    ):
        self.iters = iters
        self.pop_size = pop_size
        self.n_features_fuzzy_sets = n_features_fuzzy_sets
        self.max_rules_in_base = max_rules_in_base
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_number_bits(self, value: int) -> int:
        return int(np.ceil(np.log2(value)))

    def create_features_terms(self, X: np.ndarray):
        self.features_terms_point = []
        self.ignore_terms_id = []
        for x_col, n_sets in zip(X.T, self.n_features_fuzzy_sets):
            temp = np.full((n_sets + 1, 3), np.nan)
            self.ignore_terms_id.append(n_sets)
            x_min, x_max = x_col.min(), x_col.max()
            cuts, h = np.linspace(x_min, x_max, n_sets, retstep=True)
            temp[:n_sets, 1] = cuts
            temp[:n_sets, 0] = cuts - h
            temp[:n_sets, 2] = cuts + h
            temp[0, 0] = -np.inf
            temp[n_sets - 1, 2] = np.inf
            self.features_terms_point.append(temp)
        # move to torch
        self.ignore_terms_id = torch.tensor(self.ignore_terms_id, device=self.device)
        self.features_terms_point = [
            to_tensor(arr, self.device) for arr in self.features_terms_point
        ]

    def triangular_mem(self, x: torch.Tensor, term: torch.Tensor) -> torch.Tensor:
        left, center, right = term[0], term[1], term[2]
        mem = torch.zeros_like(x)
        lmask = (x >= left) & (x <= center)
        rmask = (x > center) & (x <= right)
        # handle infinite edges: infinite bounds yield full membership on that side
        # check left bound
        if not bool(torch.isfinite(left)) and left < 0:
            mem[lmask] = 1.0
        else:
            denom_l = (center - left).clamp(min=1e-6)
            mem[lmask] = (x[lmask] - left) / denom_l
        # check right bound
        if not bool(torch.isfinite(right)) and right > 0:
            mem[rmask] = 1.0
        else:
            denom_r = (right - center).clamp(min=1e-6)
            mem[rmask] = (right - x[rmask]) / denom_r
        return mem

    def fuzzification(self, x: torch.Tensor, antecedent: torch.Tensor) -> torch.Tensor:
        mems = []
        for j in range(x.shape[1]):
            term_id = int(antecedent[j].item())
            if term_id == int(self.ignore_terms_id[j].item()):
                mems.append(torch.ones(x.size(0), device=self.device))
            else:
                term = self.features_terms_point[j][term_id]
                mems.append(self.triangular_mem(x[:, j], term))
        return torch.stack(mems, dim=1)

    def inference(self, fuzzy_feats: torch.Tensor) -> torch.Tensor:
        return fuzzy_feats.min(dim=1).values


class FuzzyRegressorTorch(FuzzyBaseTorch):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        n_features_fuzzy_sets: List[int],
        n_target_fuzzy_sets: Union[int, List[int]],
        max_rules_in_base: int,
        target_grid_volume: Optional[int] = None,
    ):
        super().__init__(iters, pop_size, n_features_fuzzy_sets, max_rules_in_base)
        self.n_target_fuzzy_sets = (
            [n_target_fuzzy_sets] if isinstance(n_target_fuzzy_sets, int) else n_target_fuzzy_sets
        )
        self.target_grid_volume = target_grid_volume

    def get_text_rules(
        self,
        print_intervals: bool = False,
        set_names: Optional[Dict[str, List[str]]] = None,
        target_set_names: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """Генерирует текстовое представление правил нечеткой системы.
        
        Args:
            print_intervals: Если True, добавляет интервалы термов для выходных переменных
            set_names: Словарь с именами термов для входных переменных
            target_set_names: Словарь с именами термов для выходных переменных
            
        Returns:
            Строка с текстовым представлением правил
        """
        # Используем сохраненные имена, если не переданы явно
        feat_names = set_names or self.set_names
        targ_names = target_set_names or self.target_set_names
        
        texts = []
        ignore_terms_np = self.ignore_terms_id.cpu().numpy()
        
        for rule in self.base:
            ant = rule[:self.n_features]
            cons = rule[self.n_features:]
            
            # Формируем антецедент
            parts = []
            for j, t in enumerate(ant):
                if t == ignore_terms_np[j]:
                    continue
                fname = self.feature_names[j]
                label = feat_names.get(fname, [])[t] if fname in feat_names else f"s{j}"
                parts.append(f"({fname} is {label})")
            
            head = " and ".join(parts) or "always"
            
            # Формируем консеквент
            tails = []
            for d, c in enumerate(cons):
                tname = self.target_names[d]
                label = targ_names.get(tname, [])[c] if tname in targ_names else f"s{c}"
                
                if print_intervals:
                    inter = self.target_terms_point[d][c].cpu().numpy()
                    tails.append(f"{tname} ≈ {label} ({inter[0]:.3f},{inter[2]:.3f})")
                else:
                    tails.append(f"{tname} is {label}")
            
            texts.append(f"if {head} then " + ", ".join(tails))
        
        return "\n".join(texts)

    def count_antecedents(self) -> List[int]:
        """Подсчитывает количество активных условий в антецедентах правил.
        
        Returns:
            Список с количеством активных условий для каждого правила
        """
        ignore_terms_np = self.ignore_terms_id.cpu().numpy()
        counts = []
        
        for rule in self.base:
            ant = rule[:self.n_features]
            count = np.sum(ant != ignore_terms_np)
            counts.append(int(count))
            
        return counts

    def define_sets(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        set_names: Optional[Dict[str, List[str]]] = None,
        target_names: Optional[List[str]] = None,
        target_set_names: Optional[Dict[str, List[str]]] = None,
    ):
        self.n_samples, self.n_features = X.shape
        self.n_outputs = y.shape[1]
        self.feature_names = feature_names or [f"X_{i}" for i in range(self.n_features)]
        self.target_names = target_names or [f"Y_{d}" for d in range(self.n_outputs)]
        self.set_names = set_names or {
            fn: [f"{fn}_s{j}" for j in range(ns)]
            for fn, ns in zip(self.feature_names, self.n_features_fuzzy_sets)
        }
        self.target_set_names = target_set_names or {
            tn: [f"{tn}_s{j}" for j in range(ns)]
            for tn, ns in zip(self.target_names, self.n_target_fuzzy_sets)
        }
        self.create_features_terms(X)
        self.create_target_terms(y)

    def create_target_terms(self, y: np.ndarray):
        self.target_terms_point = []
        self.target_grids = []
        for ys, n_sets in zip(y.T, self.n_target_fuzzy_sets):
            y_min, y_max = ys.min(), ys.max()
            cuts, h = np.linspace(y_min, y_max, n_sets, retstep=True)
            temp = np.vstack([cuts - h, cuts, cuts + h]).T
            temp[0, 0] = y_min
            temp[-1, 2] = y_max
            self.target_terms_point.append(to_tensor(temp, self.device))
            vol = self.target_grid_volume or len(ys)
            self.target_grids.append(to_tensor(np.linspace(y_min, y_max, vol), self.device))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyRegressorTorch":
        y_np = y.astype(float)
        self._y_mean = torch.tensor(np.mean(y_np, axis=0), device=self.device)
        self.interpolate_memberships = []
        for terms, grid in zip(self.target_terms_point, self.target_grids):
            self.interpolate_memberships.append(
                torch.stack(
                    [self.triangular_mem(grid, terms[i]) for i in range(terms.size(0))], dim=0
                )
            )
        sets_all = self.n_features_fuzzy_sets + self.n_target_fuzzy_sets + [1]
        bits_unit = [self.find_number_bits(s + 1) for s in sets_all]
        bits = np.array(bits_unit * self.max_rules_in_base, int)
        sampler = SamplingGrid(fit_by="parts").fit(
            left=np.zeros_like(bits, float), right=2**bits - 1, arg=bits
        )
        num_vars = self.n_features + self.n_outputs
        genes = num_vars + 1

        def genotype_to_phenotype(pop_g):
            pop_ph = []
            int_g = sampler.transform(pop_g).astype(int)
            ignore_np = self.ignore_terms_id.cpu().numpy()
            for indiv in int_g:
                mat = indiv.reshape(self.max_rules_in_base, genes)
                rule_ids = mat[:, :num_vars]
                switch = mat[:, -1]
                rule_ids = rule_ids[switch == 1]
                for j in range(num_vars):
                    mod = (
                        self.n_features_fuzzy_sets[j]
                        if j < self.n_features
                        else self.n_target_fuzzy_sets[j - self.n_features]
                    )
                    rule_ids[:, j] %= mod
                uniq = np.unique(rule_ids, axis=0)
                valid = [r for r in uniq if not np.all(r[: self.n_features] == ignore_np)]
                pop_ph.append(np.array(valid, int))
            return np.array(pop_ph, object)

        def fitness(pop_ph):
            fits = []
            X_t = torch.tensor(X, device=self.device)
            for rules in pop_ph:
                if len(rules) == 0:
                    fits.append(-np.inf)
                    continue
                ant = torch.tensor(rules[:, : self.n_features], device=self.device)
                cons = torch.tensor(rules[:, self.n_features :], device=self.device)
                acts = torch.stack([self.inference(self.fuzzification(X_t, a)) for a in ant], dim=0)
                preds = []
                for d in range(self.n_outputs):
                    interp = self.interpolate_memberships[d]
                    gridv = self.target_grids[d]
                    mem = interp[cons[:, d]]  # (n_rules, grid_pts)
                    cut = torch.min(acts[:, :, None], mem[:, None, :])  # correct dims
                    agg = cut.max(dim=0).values
                    num = (agg * gridv).sum(dim=1)
                    den = agg.sum(dim=1)
                    y_pred = torch.where(den != 0, num / den, self._y_mean[d])
                    preds.append(y_pred)
                preds_t = torch.stack(preds, dim=1)
                r2 = r2_score(y_np, preds_t.cpu().numpy(), multioutput="uniform_average")
                fits.append(r2 - (len(rules) / self.max_rules_in_base) * 1e-10)
            return np.array(fits)

        opt = SelfCSHAGA(
            fitness_function=fitness,
            genotype_to_phenotype=genotype_to_phenotype,
            iters=self.iters,
            pop_size=self.pop_size,
            str_len=sum(sampler.parts),
            show_progress_each=1,
            no_increase_num=100,
        )
        opt.fit()
        self.base = opt.get_fittest()["phenotype"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, device=self.device)
        ant = torch.tensor(self.base[:, : self.n_features], device=self.device)
        cons_dim = self.base[:, self.n_features :]
        acts = torch.stack([self.inference(self.fuzzification(X_t, a)) for a in ant], dim=0)
        preds = []
        for d in range(self.n_outputs):
            interp = self.interpolate_memberships[d]
            gridv = self.target_grids[d]
            cons = torch.tensor(cons_dim[:, d], device=self.device)
            mem = interp[cons]
            cut = torch.min(acts[:, :, None], mem[:, None, :])  # correct dims
            agg = cut.max(dim=0).values
            num = (agg * gridv).sum(dim=1)
            den = agg.sum(dim=1)
            y_pred = torch.where(den != 0, num / den, self._y_mean[d])
            preds.append(y_pred)
        return torch.stack(preds, dim=1).cpu().numpy()
