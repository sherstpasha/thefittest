import numpy as np


def proportional_selection(fitness, ranks,
                           tour_size, quantity):
    probability = fitness/fitness.sum()
    choosen = np.random.choice(range(len(fitness)),
                               size=quantity, p=probability)
    return choosen


def rank_selection(fitness, ranks,
                   tour_size, quantity):
    probability = ranks/np.sum(ranks)
    choosen = np.random.choice(range(len(fitness)),
                               size=quantity, p=probability)
    return choosen


def tournament_selection(fitness, ranks,
                         tour_size, quantity):
    tournament = np.random.choice(
        range(len(fitness)), tour_size*quantity)
    tournament = tournament.reshape(-1, tour_size)
    max_fit_id = np.argmax(fitness[tournament], axis=1)
    choosen = np.diag(tournament[:, max_fit_id])
    return choosen