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

def server_program():
    task_queue = multiprocessing.SimpleQueue()
    result_queue = multiprocessing.SimpleQueue()

    # Генерация задач
    eps = 0.0001
    initial_iters_pop = 50
    max_iters_pop = 10000
    increment_step = 50
    n_runs = 10
    ndim = 10
    functions = [cec2005.F12005(ndim=ndim),
                          cec2005.F22005(ndim=ndim),
                          cec2005.F32005(ndim=ndim),
                          cec2005.F42005(ndim=ndim),
                          cec2005.F52005(ndim=ndim),
                          cec2005.F62005(ndim=ndim),
                          cec2005.F72005(ndim=ndim),
                          cec2005.F82005(ndim=ndim),
                          cec2005.F92005(ndim=ndim),
                          cec2005.F102005(ndim=ndim),
                          cec2005.F112005(ndim=ndim),
                          cec2005.F122005(ndim=ndim),
                          cec2005.F132005(ndim=ndim),
                          cec2005.F142005(ndim=ndim),
                          cec2005.F152005(ndim=ndim),
                          cec2005.F162005(ndim=ndim),
                          cec2005.F172005(ndim=ndim),
                          cec2005.F182005(ndim=ndim),
                          cec2005.F192005(ndim=ndim),
                          cec2005.F202005(ndim=ndim),
                          cec2005.F212005(ndim=ndim),
                          cec2005.F222005(ndim=ndim),
                          cec2005.F232005(ndim=ndim),
                          cec2005.F242005(ndim=ndim),
                          cec2005.F252005(ndim=ndim)]  # Укажите ваши функции

    for function in functions:
        for iters_pop in range(initial_iters_pop, max_iters_pop + increment_step, increment_step):
            for selection in ["proportional", "rank", "tournament_3"]:
                for crossover in ["one_point", "two_point", "uniform_2"]:
                    for mutation in ["weak", "average", "strong"]:
                        for _ in range(n_runs):
                            task_queue.put((function, eps, iters_pop, selection, crossover, mutation))

    # Настройка сокета для соединения
    host = socket.gethostname()
    port = 5000  # Используйте свободный порт

    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(2)  # Слушаем максимум два соединения

    conn, address = server_socket.accept()  # Принимаем соединение
    print(f"Connection from: {address}")

    # Отправляем задачи на клиент
    while not task_queue.empty():
        task = task_queue.get()
        conn.send(pickle.dumps(task))

    # Получаем результаты от клиента
    while True:
        data = conn.recv(4096)
        if not data:
            break
        result = pickle.loads(data)
        result_queue.put(result)

    # Закрываем соединение
    conn.close()

    # Обработка результатов
    while not result_queue.empty():
        result = result_queue.get()
        print("Received result:", result)

if __name__ == '__main__':
    server_program()
