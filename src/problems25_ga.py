import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from thefittest.benchmarks import (
    BanknoteDataset,
    BreastCancerDataset,
    CreditRiskDataset,
    TwoNormDataset,
    RingNormDataset,
)
from thefittest.optimizers import SelfCGA
from thefittest.tools.transformations import GrayCode
import warnings
warnings.filterwarnings("ignore")

# Папка для результатов
os.makedirs("results", exist_ok=True)

# Датасеты
datasets = {
    "Banknote":     (BanknoteDataset().get_X(),     BanknoteDataset().get_y()),
    "BreastCancer": (BreastCancerDataset().get_X(), BreastCancerDataset().get_y()),
    "CreditRisk":   (CreditRiskDataset().get_X(),   CreditRiskDataset().get_y()),
    "TwoNorm":      (TwoNormDataset().get_X(),      TwoNormDataset().get_y()),
    "RingNorm":     (RingNormDataset().get_X(),     RingNormDataset().get_y()),
}

# Методы и их пространства гиперпараметров
# Методы и их пространства гиперпараметров
methods = {
    "RandomForest": {
        "cls": RandomForestClassifier,
        "param_space": {
            "n_estimators":     ("int",         10, 200),
            "max_depth":        ("int",          1, 20),
            "max_features":     ("categorical", ["auto","sqrt","log2"]),
            "min_samples_split":("int",          2, 20),
        },
        "gc_bounds": {
            "left":  np.array([10.0, 1.0, 0.0, 2.0]),
            "right": np.array([200.0,20.0,2.0,20.0]),
            # шаг: 1 для всех int/категорий
            "arg":   np.array([1.0, 1.0, 1.0, 1.0])
        }
    },
    "GradientBoosting": {
        "cls": GradientBoostingClassifier,
        "param_space": {
            "n_estimators":  ("int",   50, 200),
            "learning_rate":("float", 0.01, 1.0),
            "max_depth":    ("int",     1, 10),
            "subsample":    ("float",  0.5, 1.0),
        },
        "gc_bounds": {
            "left":  np.array([50.0, 0.01, 1.0, 0.5]),
            "right": np.array([200.0,1.0,10.0,1.0]),
            # шаг: 1 для n_estimators, 0.01 для learning_rate, 1 для max_depth, 0.01 для subsample
            "arg":   np.array([1.0, 0.01, 1.0, 0.01])
        }
    },
    "MLP": {
        "cls": MLPClassifier,
        "param_space": {
            "hidden_layer_sizes": ("categorical", [(50,),(100,),(50,50),(100,50)]),
            "alpha":              ("float",       1e-5,   1e-1),
            "learning_rate_init": ("float",       1e-4,   1.0),
            "solver":             ("categorical", ["lbfgs","sgd","adam"]),
        },
        "gc_bounds": {
            "left":  np.array([0.0, 1e-5, 1e-4, 0.0]),
            "right": np.array([3.0, 1e-1, 1.0, 2.0]),
            # шаг: 1 для categorical hidden_layer_sizes, 0.0001 для alpha, 0.001 для lr_init, 1 для solver
            "arg":   np.array([1.0, 1e-4, 0.001, 1.0])
        }
    },
}


def make_vector_tools(param_space):
    idx = {name:(i,i+1) for i,name in enumerate(param_space)}
    def vector_to_params(x):
        p = {}
        for name,spec in param_space.items():
            lo,_ = idx[name]
            raw = x[lo]
            if spec[0]=="float":
                mn,mx = spec[1],spec[2]
                p[name]=float(np.clip(raw,mn,mx))
            elif spec[0]=="int":
                mn,mx = spec[1],spec[2]
                vi=int(round(raw))
                p[name]=int(np.clip(vi,mn,mx))
            else:
                choices=spec[1]
                ci=int(round(raw))
                ci=int(np.clip(ci,0,len(choices)-1))
                p[name]=choices[ci]
        return p
    return idx, vector_to_params

def run_hp(method_cfg, ds_name, iteration, X_params, mode='train'):
    X,y = datasets[ds_name]
    X=X.astype(np.float32); y=y.astype(int)
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,train_size=0.75,random_state=iteration)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test =scaler.transform(X_test)

    _, to_params = make_vector_tools(method_cfg["param_space"])
    results=[]
    for x in X_params:
        params=to_params(x)
        clf=method_cfg["cls"](random_state=iteration,**params)
        if mode=='train':
            cv=KFold(n_splits=2,shuffle=True,random_state=iteration)
            scores=cross_val_score(clf,X_train,y_train,cv=cv,scoring='f1_macro')
            results.append(scores.mean())
        else:
            clf.fit(X_train,y_train)
            results.append(f1_score(y_test,clf.predict(X_test),average='macro'))
    return np.array(results)

def optimize_single(args):
    ds_name, m_name, iteration = args
    cfg = methods[m_name]
    # Split & scale once for baseline
    X,y = datasets[ds_name]
    X_train,X_test,y_train,y_test = train_test_split(
        X.astype(np.float32), y.astype(int),
        train_size=0.75, random_state=iteration
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Baseline default params
    base_clf = cfg["cls"](random_state=iteration)
    # train baseline CV
    cv = KFold(n_splits=2, shuffle=True, random_state=iteration)
    base_train = cross_val_score(base_clf, X_train, y_train, cv=cv, scoring='f1_macro').mean()
    # test baseline
    base_clf.fit(X_train, y_train)
    base_test = f1_score(y_test, base_clf.predict(X_test), average='macro')

    # Optimize
    gb = cfg["gc_bounds"]
    g2p = GrayCode(fit_by="h").fit(left=gb["left"], right=gb["right"], arg=gb["arg"])
    str_len = sum(g2p.parts)
    fitness_fn = lambda pop: run_hp(cfg, ds_name, iteration, pop, 'train')
    opt = SelfCGA(
        fitness_function      = fitness_fn,
        genotype_to_phenotype = g2p.transform,
        iters                 = 30,
        pop_size              = 30,
        str_len               = str_len,
        optimal_value         = 1.0,
        no_increase_num=5,
    )
    opt.fit()

    best_raw = opt.get_fittest()["phenotype"]
    _, to_params = make_vector_tools(cfg["param_space"])
    best_params = to_params(best_raw)

    # Metrics for best
    raw_batch = np.array([best_raw])
    train_f1 = run_hp(cfg, ds_name, iteration, raw_batch, 'train')[0]
    test_f1  = run_hp(cfg, ds_name, iteration, raw_batch, 'test')[0]

    result = {
        "dataset":        ds_name,
        "method":         m_name,
        "iteration":      iteration,
        # best
        **best_params,
        "train_f1_cv":    train_f1,
        "test_f1":        test_f1,
        # baseline
        "base_train_cv":  base_train,
        "base_test":      base_test,
    }
    return result

if __name__ == "__main__":
    tasks = [(ds, m, i) for ds in datasets for m in methods for i in range(1, 21)]
    summary = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(optimize_single, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            summary.append(res)
            print(f"Done: {res['dataset']}-{res['method']} iter {res['iteration']}")

    df = pd.DataFrame(summary)
    df.to_excel("results/hpo_summary_all_methods.xlsx", index=False)
    print("All results saved to 'results/hpo_summary_all_methods.xlsx'")
