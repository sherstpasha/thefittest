import numpy as np


class TheFittest:
    def __init__(self, genotype, phenotype, fitness):
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness

    def update(self, population_g, population_ph, fitness):
        temp_best_id = np.argmin(fitness)
        temp_best_fitness = fitness[temp_best_id].copy()
        if temp_best_fitness < self.fitness:
            self.genotype = population_g[temp_best_id].copy()
            self.phenotype = population_ph[temp_best_id].copy()
            self.fitness = temp_best_fitness

        return self

    def __str__(self):
        return 'genotype: ' + str(self.genotype) + '\n' + \
            'phenotype: ' + str(self.phenotype) + '\n' + \
            'fitness: ' + str(self.fitness)

    def get(self):
        return self.genotype.copy(), self.phenotype.copy(), self.fitness.copy()


class Statictic:
    def __init__(self):
        self.population = np.array([], dtype=np.byte)
        self.fitness = np.array([], dtype=float)
        self.s_proba = np.array([], dtype=float)
        self.c_proba = np.array([], dtype=float)
        self.m_proba = np.array([], dtype=float)

    def update(self, population_i, fitness_i, s_proba_i, c_proba_i, m_proba_i):
        shape = (-1, population_i.shape[0], population_i.shape[1])
        shape_in = (1, population_i.shape[0], population_i.shape[1])
        self.population = np.vstack([self.population.reshape(shape),
                                     population_i.copy().reshape(shape_in)])
        self.fitness = np.append(self.fitness, np.min(fitness_i))
        self.s_proba = np.vstack([self.s_proba.reshape(-1, len(s_proba_i)),
                                  s_proba_i.copy()])
        self.c_proba = np.vstack([self.c_proba.reshape(-1, len(c_proba_i)),
                                  c_proba_i.copy()])
        self.m_proba = np.vstack([self.m_proba.reshape(-1, len(m_proba_i)),
                                  m_proba_i.copy()])
        return self
