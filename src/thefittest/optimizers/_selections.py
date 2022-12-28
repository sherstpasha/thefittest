import numpy as np
from ..tools import rank_data


def proportional_selection(population, fitness, tour_size, quantity):
    probability = fitness/fitness.sum()
    choosen = np.random.choice(range(len(population)),
                               size=quantity, p=probability)
    return choosen


def rank_selection(population, fitness, tour_size, quantity):
    ranks = rank_data(fitness)
    probability = ranks/np.sum(ranks)
    choosen = np.random.choice(range(len(population)),
                               size=quantity, p=probability)
    return choosen


def tournament_selection(population, fitness, tour_size, quantity):
    tournament = np.random.choice(
        range(len(population)), tour_size*quantity)
    tournament = tournament.reshape(-1, tour_size)
    max_fit_id = np.argmin(fitness[tournament], axis=1)
    choosen = np.diag(tournament[:, max_fit_id])
    return choosen
