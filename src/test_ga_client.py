import multiprocessing
import socket
import pickle
import numpy as np
import pandas as pd
from opfunu.cec_based import cec2005
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.transformations import GrayCode
from tqdm import tqdm

def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        if np.all(np.abs(solution - true_solution) <= precision):
            return i + 1
    return None

def run_optimization(function, eps, iters, pop_size, selection, crossover, mutation):
    reliability = 0.
    speed_sum = 0
    range_left = np.nan
    range_right = np.nan
    find_count = 0

    left = np.array(function.bounds[:, 0], dtype=np.float64)
    right = np.array(function.bounds[:, 1], dtype=np.float64)
    h = np.array([eps] * function.ndim, dtype=np.float64)

    genotype_to_phenotype = GrayCode().fit(left, right, h)
    str_len = genotype_to_phenotype.parts.sum()

    def fitness_function(population_g):
        fitness = np.array(list(map(function.evaluate, population_g)), dtype=np.float64)
        return fitness

    optimizer = GeneticAlgorithm(fitness_function=fitness_function,
                                 genotype_to_phenotype=genotype_to_phenotype.transform,
                                 iters=iters,
                                 pop_size=pop_size,
                                 str_len=str_len,
                                 elitism=False,
                                 selection=selection,
                                 crossover=crossover,
                                 mutation=mutation,
                                 keep_history=True,
                                 minimization=True)
    optimizer.fit()
    stat = optimizer.get_stats()
    speed_i = find_solution_with_precision(stat["max_ph"], function.f_global, h)

    if speed_i is not None:
        reliability = 1
        speed_sum = speed_i
        range_left = speed_i
        range_right = speed_i
        find_count = 1

    return reliability, speed_sum, range_left, range_right, find_count

def client_program():
    host = socket.gethostname()  # Или укажите IP сервера
    port = 5000  # Должен совпадать с портом сервера

    client_socket = socket.socket()
    client_socket.connect((host, port))

    # Получаем задачи от сервера
    while True:
        data = client_socket.recv(4096)
        if not data:
            break
        task = pickle.loads(data)
        function, eps, iters, selection, crossover, mutation = task

        # Выполняем задачу
        result = run_optimization(function, eps, iters, selection, crossover, mutation)

        # Отправляем результат обратно на сервер
        client_socket.send(pickle.dumps(result))

    # Закрываем соединение
    client_socket.close()

if __name__ == '__main__':
    client_program()
