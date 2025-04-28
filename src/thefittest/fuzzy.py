from typing import Optional, Union, List, Dict
import numpy as np
from sklearn.metrics import r2_score
from thefittest.tools.transformations import SamplingGrid
from thefittest.optimizers._selfcshaga import SelfCSHAGA


class FuzzyBase:
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

    def find_number_bits(self, value: int) -> int:
        return int(np.ceil(np.log2(value)))

    def create_features_terms(self, X: np.ndarray):
        self.features_terms_point = []
        self.ignore_terms_id = []
        for idx, x_col in enumerate(X.T):
            n_sets = self.n_features_fuzzy_sets[idx]
            temp = np.full((n_sets + 1, 3), np.nan)
            self.ignore_terms_id.append(n_sets)
            x_min, x_max = x_col.min(), x_col.max()
            cuts, h = np.linspace(x_min, x_max, n_sets, retstep=True)
            temp[:n_sets, 1] = cuts
            temp[:n_sets, 0] = cuts - h
            temp[:n_sets, 2] = cuts + h
            temp[0, 0] = x_min
            temp[n_sets - 1, 2] = x_max
            self.features_terms_point.append(temp)
        self.ignore_terms_id = np.array(self.ignore_terms_id, dtype=int)
        self.features_terms_point = tuple(self.features_terms_point)

    def triangular_mem(self, x: np.ndarray, term: np.ndarray) -> np.ndarray:
        left, center, right = term
        mem = np.zeros_like(x, dtype=float)
        l_mask = (x >= left) & (x <= center)
        r_mask = (x > center) & (x <= right)
        denom_l = center - left if center != left else 1
        denom_r = right - center if right != center else 1
        mem[l_mask] = (x[l_mask] - left) / denom_l
        mem[r_mask] = (right - x[r_mask]) / denom_r
        return mem

    def fuzzification(self, X: np.ndarray, antecedent: np.ndarray) -> np.ndarray:
        m = np.zeros_like(X, dtype=float)
        for j, term_id in enumerate(antecedent):
            if term_id == self.ignore_terms_id[j]:
                continue
            term = self.features_terms_point[j][term_id]
            m[:, j] = self.triangular_mem(X[:, j], term)
        return m

    def inference(self, fuzzy_feats: np.ndarray) -> np.ndarray:
        return np.min(fuzzy_feats, axis=1)


class FuzzyRegressor(FuzzyBase):
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
        if isinstance(n_target_fuzzy_sets, int):
            self.n_target_fuzzy_sets = [n_target_fuzzy_sets]
        else:
            self.n_target_fuzzy_sets = n_target_fuzzy_sets
        self.target_grid_volume = target_grid_volume
        self.set_names: Dict[str, List[str]] = {}
        self.target_set_names: Dict[str, List[str]] = {}

    def define_sets(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        set_names: Optional[Dict[str, List[str]]] = None,
        target_names: Optional[List[str]] = None,
        target_set_names: Optional[Dict[str, List[str]]] = None,
    ):
        # Размерности
        self.n_samples, self.n_features = X.shape
        self.n_outputs = y.shape[1]
        assert self.n_outputs == len(
            self.n_target_fuzzy_sets
        ), "n_target_fuzzy_sets length must match number of outputs"
        # Имена признаков и целей
        self.feature_names = feature_names or [f"X_{i}" for i in range(self.n_features)]
        self.target_names = target_names or [f"Y_{d}" for d in range(self.n_outputs)]
        # Названия нечетких множеств для признаков
        for i, fname in enumerate(self.feature_names):
            n_sets = self.n_features_fuzzy_sets[i]
            if set_names and fname in set_names:
                labels = set_names[fname]
            else:
                labels = [f"{fname}_s{j}" for j in range(n_sets)]
            self.set_names[fname] = labels
        # Названия нечетких множеств для целей
        for d, tname in enumerate(self.target_names):
            n_sets = self.n_target_fuzzy_sets[d]
            if target_set_names and tname in target_set_names:
                labels = target_set_names[tname]
            else:
                labels = [f"{tname}_s{j}" for j in range(n_sets)]
            self.target_set_names[tname] = labels
        # Построение термов
        self.create_features_terms(X)
        self.create_target_terms(y)

    def create_target_terms(self, y: np.ndarray):
        self.target_terms_point = []
        self.target_grids = []
        for d, n_sets in enumerate(self.n_target_fuzzy_sets):
            ys = y[:, d]
            y_min, y_max = ys.min(), ys.max()
            cuts, h = np.linspace(y_min, y_max, n_sets, retstep=True)
            temp = np.zeros((n_sets, 3))
            temp[:, 1] = cuts
            temp[:, 0] = cuts - h
            temp[:, 2] = cuts + h
            temp[0, 0] = y_min
            temp[-1, 2] = y_max
            self.target_terms_point.append(temp)
            vol = self.target_grid_volume or len(ys)
            self.target_grids.append(np.linspace(y_min, y_max, vol))
        self.target_terms_point = tuple(self.target_terms_point)
        self.target_grids = tuple(self.target_grids)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyRegressor":
        y = y.astype(float)
        self.interpolate_memberships = []
        for terms, grid in zip(self.target_terms_point, self.target_grids):
            interp = np.array([self.triangular_mem(grid, t) for t in terms])
            self.interpolate_memberships.append(interp)

        sets_all = self.n_features_fuzzy_sets + self.n_target_fuzzy_sets + [1]
        bits_unit = [self.find_number_bits(s + 1) for s in sets_all]
        bits = np.array(bits_unit * self.max_rules_in_base, dtype=int)
        left = np.zeros_like(bits, dtype=float)
        right = 2**bits - 1
        grid = SamplingGrid(fit_by="parts").fit(left=left, right=right, arg=bits)

        num_vars = self.n_features + self.n_outputs
        genes = num_vars + 1

        def genotype_to_phenotype(pop_g: np.ndarray) -> np.ndarray:
            pop_ph = []
            int_g = grid.transform(pop_g).astype(int)
            for indiv in int_g:
                mat = indiv.reshape(self.max_rules_in_base, genes)
                rule_ids = mat[:, :num_vars]
                switch = mat[:, -1]
                rule_ids = rule_ids[switch == 1]
                for j in range(num_vars):
                    if j < self.n_features:
                        mod = self.n_features_fuzzy_sets[j] + 1
                    else:
                        mod = self.n_target_fuzzy_sets[j - self.n_features]
                    rule_ids[:, j] %= mod
                uniq = np.unique(rule_ids, axis=0)
                non_empty = [
                    r for r in uniq if not np.all(r[: self.n_features] == self.ignore_terms_id)
                ]
                pop_ph.append(np.array(non_empty, dtype=int))
            return np.array(pop_ph, dtype=object)

        def fitness(ph_population: List[np.ndarray]) -> np.ndarray:
            fit = np.zeros(len(ph_population), dtype=float)
            for i, rules in enumerate(ph_population):
                if len(rules) < 1:
                    fit[i] = -np.inf
                    continue
                ant = rules[:, : self.n_features]
                cons = rules[:, self.n_features :].astype(int)
                acts = np.array([self.inference(self.fuzzification(X, a)) for a in ant])
                preds = np.zeros((self.n_samples, self.n_outputs), dtype=float)
                for d in range(self.n_outputs):
                    interp = self.interpolate_memberships[d]
                    grid_vals = self.target_grids[d]
                    mem = interp[cons[:, d]]
                    cut = np.minimum(acts[:, :, None], mem[:, None, :])
                    agg = np.max(cut, axis=0)
                    num = (agg * grid_vals).sum(axis=1)
                    den = agg.sum(axis=1)
                    den[den == 0] = 1
                    preds[:, d] = num / den
                score = r2_score(y, preds, multioutput="uniform_average")
                penalty = (len(rules) / self.max_rules_in_base) * 1e-10
                fit[i] = score - penalty
            return fit

        optimizer = SelfCSHAGA(
            fitness_function=fitness,
            genotype_to_phenotype=genotype_to_phenotype,
            iters=self.iters,
            pop_size=self.pop_size,
            str_len=sum(grid.parts),
            show_progress_each=1,
            no_increase_num=100,
        )
        optimizer.fit()
        self.base = optimizer.get_fittest()["phenotype"]

        self._y_mean = np.mean(y, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Дефаззификация с заменой на среднее для не покрытых точек.
        """
        num_vars = self.n_features + self.n_outputs
        ant = self.base[:, : self.n_features]
        cons = self.base[:, self.n_features :].astype(int)

        # степени активации правил (n_rules, n_samples)
        acts = np.array([self.inference(self.fuzzification(X, a)) for a in ant])

        preds = np.zeros((X.shape[0], self.n_outputs), dtype=float)
        for d in range(self.n_outputs):
            interp = self.interpolate_memberships[d]  # (n_sets, grid_pts)
            grid_vals = self.target_grids[d]  # (grid_pts,)

            # для каждого правила r и каждой точки i: min(act[r,i], mem[r,grid])
            mem = interp[cons[:, d]]  # (n_rules, grid_pts)
            cut = np.minimum(acts[:, :, None], mem[:, None, :])  # (n_rules, n_samples, grid_pts)
            agg = np.max(cut, axis=0)  # (n_samples, grid_pts)

            # числитель и знаменатель для центроида
            num = (agg * grid_vals).sum(axis=1)  # (n_samples,)
            den = agg.sum(axis=1)  # (n_samples,)

            # где ничего не сработало (den==0) — подставляем среднее по y
            mask_valid = den != 0
            # обычная дефаззификация
            preds[mask_valid, d] = num[mask_valid] / den[mask_valid]
            # для «пустых» точек
            preds[~mask_valid, d] = self._y_mean[d]

        return preds

    def get_text_rules(
        self,
        print_intervals: bool = False,
        set_names: Optional[Dict[str, List[str]]] = None,
        target_set_names: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        # Локальные или сохраненные имена
        feat_names = set_names or self.set_names
        targ_names = target_set_names or self.target_set_names
        texts = []
        for rule in self.base:
            ant, cons = rule[: self.n_features], rule[self.n_features :]
            parts = []
            for j, t in enumerate(ant):
                if t == self.ignore_terms_id[j]:
                    continue
                fname = self.feature_names[j]
                label = feat_names.get(fname, [])[t] if fname in feat_names else f"s{j}"
                parts.append(f"({fname} is {label})")
            head = " and ".join(parts) or "always"
            tails = []
            for d, c in enumerate(cons):
                tname = self.target_names[d]
                label = targ_names.get(tname, [])[c] if tname in targ_names else f"s{c}"
                if print_intervals:
                    inter = self.target_terms_point[d][c]
                    tails.append(f"{tname} ≈ {label} ({inter[0]:.3f},{inter[2]:.3f})")
                else:
                    tails.append(f"{tname} is {label}")
            texts.append(f"if {head} then " + ", ".join(tails))
        return "\n".join(texts)

    def count_antecedents(self) -> List[int]:
        return [int(np.sum(rule[: self.n_features] != self.ignore_terms_id)) for rule in self.base]
