from ..base._ea import TheFittest
from ..base._ea import Statistics
from ..base._net import Net
import numpy as np


def test_TheFittest():
    net_1 = Net()
    net_2 = Net()
    net_3 = Net()

    population_g = np.array([net_1, net_2, net_3], dtype=object)
    population_ph = population_g
    fitness = np.array([1, 2, 3], dtype=np.float64)

    thefittest_ = TheFittest()
    thefittest_._update(population_g, population_ph, fitness)
    assert thefittest_._fitness == 3
    assert thefittest_._no_update_counter == 0
    assert thefittest_._fitness is not fitness[2]
    assert thefittest_._genotype is not population_g[2]
    assert thefittest_._phenotype is not population_ph[2]

    fitness = np.array([1, 2, 2], dtype=np.float64)
    thefittest_._update(population_g, population_ph, fitness)
    assert thefittest_._no_update_counter == 1
    assert thefittest_._fitness == 3

    fitness = np.array([1, 2, 4], dtype=np.float64)
    thefittest_._update(population_g, population_ph, fitness)
    assert thefittest_._no_update_counter == 0
    assert thefittest_._fitness == 4

    fittest = thefittest_.get()
    assert type(fittest) is dict
    assert len(fittest) == 3


def test_Statistic():
    net_1 = Net()
    net_2 = Net()
    net_3 = Net()
    population_g = np.array([net_1, net_2, net_3], dtype=object)
    population_ph = population_g
    fitness = np.array([1, 2, 3], dtype=np.float64)

    statistics_ = Statistics()
    fitness_max = np.max(fitness)
    statistics_._update({'population_g': population_g,
                        'population_ph': population_ph,
                         'fitness_max': fitness_max})

    assert statistics_['fitness_max'][0] == fitness_max
    assert np.all(statistics_['population_g'][0] == population_g)
    assert np.all(statistics_['population_ph'][0] == population_ph)

    statistics_._update({'population_g': population_g,
                        'population_ph': population_ph,
                         'fitness_max': fitness_max})

    assert len(statistics_['fitness_max']) == 2
    assert len(statistics_['population_g']) == 2
    assert len(statistics_['population_ph']) == 2
