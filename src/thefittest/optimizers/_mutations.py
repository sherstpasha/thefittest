import numpy as np


def flip_mutation(individ, proba):
    mask = np.random.random(size=individ.shape) < proba
    individ[mask] = 1 - individ[mask]
    return individ


def best_1(current, population, F_value):
    best = population[-1]
    r1, r2 = np.random.choice(range(len(population)), size=2, replace=False)
    return best + F_value*(population[r1] - population[r2])


def rand_1(current, population, F_value):
    r1, r2, r3 = np.random.choice(
        range(len(population)), size=3, replace=False)
    return population[r3] + F_value*(population[r1] - population[r2])


def rand_to_best1(current, population, F_value):
    best = population[-1]
    r1, r2, r3 = np.random.choice(
        range(len(population)), size=3, replace=False)
    return population[r1] + F_value*(
        best - population[r1]) + F_value*(population[r2] - population[r3])


def current_to_best_1(current, population, F_value):
    best = population[-1]
    r1, r2 = np.random.choice(range(len(population)), size=2, replace=False)
    return current + F_value*(best - current) + F_value*(population[r1] - population[r2])


def best_2(current, population, F_value):
    best = population[-1]
    r1, r2, r3, r4 = np.random.choice(
        range(len(population)), size=4, replace=False)
    return best + F_value*(population[r1] - population[r2]) + F_value*(population[r3] - population[r4])


def rand_2(current, population, F_value):
    r1, r2, r3, r4, r5 = np.random.choice(
        range(len(population)), size=5, replace=False)
    return population[r5] + F_value*(population[r1] - population[r2]) + F_value*(population[r3] - population[r4])


def current_to_pbest_1(current, population, F_value):
    p_min = 2/len(population)
    p_i = np.random.uniform(p_min, 0.2)

    value = int(p_i*len(population))
    pbest = population[-value:]
    p_best_ind = np.random.randint(0, len(pbest))

    best = pbest[p_best_ind]

    r1, r2 = np.random.choice(range(len(population)), size=2, replace=False)
    return current + F_value*(best - current) + F_value*(population[r1] - population[r2])


def current_to_pbest_1_archive(current, population, F_value, pop_archive):
    p_min = 2/len(population)
    p_i = np.random.uniform(p_min, 0.2)

    value = int(p_i*len(population))
    pbest = population[-value:]
    p_best_ind = np.random.randint(0, len(pbest))

    best = pbest[p_best_ind]

    r1 = np.random.choice(range(len(population)), size=1, replace=False)[0]
    r2 = np.random.choice(range(len(pop_archive)), size=1, replace=False)[0]
    return current + F_value*(best - current) + F_value*(population[r1] - pop_archive[r2])

def current_to_rand_1(current, population, F_value):
    r1, r2, r3 = np.random.choice(
        range(len(population)), size=3, replace=False)
    return population[r1] + F_value*(population[r3] - current) + F_value*(population[r1] - population[r2])