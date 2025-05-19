from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import f1_score

from thefittest.optimizers import SelfCGA
from thefittest.tools.transformations import SamplingGrid

class FCSelfCGA(BaseEstimator, ClassifierMixin):
    def __init__(self, iters, pop_size, n_fsets, n_rules, rl=0, bl=0, 
                 tour_size=3, keep_best=True, K=0.5, threshold=0.1, optimizer=SelfCGA):
        self.iters = iters
        self.pop_size = pop_size
        self.n_fsets = n_fsets
        self.n_rules = n_rules
        self.rl = rl
        self.bl = bl
        self.tour_size = tour_size
        self.keep_best = keep_best
        self.K = K
        self.threshold = threshold
        
        self.n_bin = np.ceil(np.log2(self.n_fsets+1)).astype(int)
        
        self.n_bin_c = None
        self.n_vars = None
        self.n_classes = None
        self.len_ = None
        self.term_dict = None
        self.ignore_dict = None
        self.list_ = np.array([], dtype = int)
        
        self.borders = {'left':np.array([0]),
                        'right':np.array([1])}
        self.parts = np.array([1], dtype = int)
        self.grid_model = None
        
        self.base = None
        self.opt_model = None
        self.optimizer = optimizer

    def create_terms(self, X, y):
        self.n_vars = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.n_bin_c = np.ceil(np.log2(self.n_classes)).astype(int)
        self.len_ = (1 + self.n_bin_c + X.shape[1]*self.n_bin)*self.n_rules
        
        self.term_dict = np.full((self.n_fsets*X.shape[1] + X.shape[1], 4),
                                 np.nan)
        
        for i, x_i in enumerate(X.T):
            x_i_min = x_i.min()
            x_i_max = x_i.max()
            cuts, h = np.linspace(x_i_min, x_i_max, self.n_fsets,
                                  retstep = True)
            l = self.n_fsets*(i)
            r = self.n_fsets*(i+1)
            self.borders['left'] = np.append(self.borders['left'],
                                             l + i)
            self.borders['right'] = np.append(self.borders['right'],
                                              l + i + 2**self.n_bin - 1)
            self.term_dict[:,1][l+i:r+i] = cuts
            self.term_dict[:,0][l+i:r+i] = self.term_dict[:,1][l+i:r+i] - h
            self.term_dict[:,2][l+i:r+i] = self.term_dict[:,1][l+i:r+i] + h
            self.term_dict[:,3][l+i:r+i] = i
            self.term_dict[:,0][l+i] = x_i_min
            self.term_dict[:,2][r+i-1] = x_i_max
            self.term_dict[r+i][3] = i
            self.parts = np.append(self.parts, self.n_bin)
        self.list_ = np.arange((self.n_fsets+1)*X.shape[1]).reshape(X.shape[1], -1)
        self.ignore_dict = self.list_[:,-1]
    
        self.borders['left'] = np.append(self.borders['left'], 0)
        self.borders['right'] = np.append(self.borders['right'],
                                          2**self.n_bin_c - 1)
        self.parts = np.append(self.parts, self.n_bin_c)
        self.grid_model = SamplingGrid(fit_by="parts").fit(left=self.borders['left'], right=self.borders['right'], arg=self.parts)
    
    def rule_membership(self, x, rule):
        result = self.culc_triangular_r(x, rule)
        if np.isnan(result).all():
            return np.full(x.shape[0], 0)
        else:
            return np.nanmean(result, axis = 1)     
    
    def culc_triangular_r(self, x, rule_id):
        result = np.full(x.shape, np.nan)
        
        left = self.term_dict[rule_id][:,0]
        center = self.term_dict[rule_id][:,1]
        right = self.term_dict[rule_id][:,2]
        
        l_mask = np.all([left <= x, x <= center], axis = 0)
        r_mask = np.all([center < x, x <= right], axis = 0)
        else_mask = np.invert(l_mask) & np.invert(r_mask)
    
        isnanall = np.any(np.isnan(self.term_dict[rule_id]),
                          axis = 1)

        else_mask[:,isnanall] = False
        l_down = center - left
        l_down[l_down == 0] = 1
        r_down = right - center
        r_down[r_down == 0] = 1

        result[l_mask] = (1 - (center - x)/l_down)[l_mask]
        result[r_mask] = (1 - (x - center)/r_down)[r_mask]
        result[else_mask] = 0

        if len(rule_id) == 1:
            return result[:,0]
        return result  
    
    def grid_and_cut(self, some_x):
        x_float = self.grid_model.transform(some_x).astype(int)
        cond = x_float[:,1:-1] > self.ignore_dict
        cond_c = x_float[:,-1] > (self.n_classes - 1)
        x_float[:,1:-1][cond] = (x_float - self.n_fsets)[:,1:-1][cond]
        x_float[:,-1][cond_c] = (x_float - (self.n_classes - 1))[:,-1][cond_c]
        return x_float
    
    def fitness_function(self, X, y, some_x):
        result = np.array([])
        for x_i in some_x:
            x_i = x_i.reshape(self.n_rules, -1)
            x_float = self.grid_and_cut(x_i)
            
            x_float = x_float[x_float[:,0] == 1][:,1:]
            query = np.array([self.rule_membership(X, rule_i[:-1])
                              for rule_i in x_float])
            if len(query) == 0:
                result = np.append(result, 0)
                continue
            
            argmax = np.nanargmax(query.reshape(-1, X.shape[0]), axis = 0)
            fitness = f1_score(y, x_float[:,-1][argmax], average='macro')
            fine_rlen = self.rl*(len(x_float)/self.n_rules)
            fine_blen = self.bl*(np.sum(x_float[:,:-1] != self.ignore_dict)/\
                x_float[:,:-1].size)
            result = np.append(result, fitness - fine_rlen - fine_blen)
        return result
    
    @staticmethod
    def __str_rules(features, terms, class_):
        text = 'if '
        for f, t in zip(features, terms):
            text += '(' + f + ' is ' + t + ') and '
        text = text[:-4] + 'then ' + class_
        return text
    
    def print_rules(self, set_names, feature_names, target_names):
        text_rules = np.array([], dtype = object)
        for i, rule_c in enumerate(self.base):
            rule = rule_c[:-1]
            c = rule_c[-1]
            index = np.arange(len(rule), dtype = int)
            rule_term_id = rule - index*(self.n_fsets + 1)
            remove_ignore = rule != self.ignore_dict            
            features = feature_names[remove_ignore]
            class_ = target_names[int(c)]
            terms = [set_names[i] for i in rule_term_id[remove_ignore]]
            text_rules = np.append(text_rules,
                                   self.__str_rules(features, terms, class_))
        return text_rules
    
    def fit(self, X, y):
        self.create_terms(X, y)
        opt_func = lambda x: self.fitness_function(X, y, x)
        self.opt_model = self.optimizer(fitness_function=opt_func, iters=self.iters,
                                          pop_size=self.pop_size, str_len=self.len_,
                                          show_progress_each=1,
                                          )
        self.opt_model.fit()
        self.base = self.grid_and_cut(
            self.opt_model.get_fittest()['genotype'].reshape(self.n_rules, -1))
        self.base = self.base[self.base[:,0] == 1][:,1:]
        return self
    
    def predict(self, X):
        query = np.array([self.rule_membership(X, rule_i[:-1])
                          for rule_i in self.base]) 
        argmax = np.nanargmax(query.reshape(-1, X.shape[0]), axis = 0)
        return self.base[:,-1][argmax]