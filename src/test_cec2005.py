import numpy as np
from thefittest.benchmarks.CEC2005 import problems_dict
from opfunu.cec_based import cec2005


N = 10

problems_dict_thefittest = problems_dict

problems_dict_opfunu = (
    cec2005.F12005(ndim=N),
    cec2005.F22005(ndim=N),
    cec2005.F32005(ndim=N),
    cec2005.F42005(ndim=N),
    cec2005.F52005(ndim=N),
    cec2005.F62005(ndim=N),
    cec2005.F72005(ndim=N),
    cec2005.F82005(ndim=N),
    cec2005.F92005(ndim=N),
    cec2005.F102005(ndim=N),
    cec2005.F112005(ndim=N),
    cec2005.F122005(ndim=N),
    cec2005.F132005(ndim=N),
    cec2005.F142005(ndim=N),
    cec2005.F152005(ndim=N),
    cec2005.F162005(ndim=N),
    cec2005.F172005(ndim=N),
    cec2005.F182005(ndim=N),
    cec2005.F192005(ndim=N),
    cec2005.F202005(ndim=N),
    cec2005.F212005(ndim=N),
    cec2005.F222005(ndim=N),
    cec2005.F232005(ndim=N),
    cec2005.F242005(ndim=N),
    cec2005.F252005(ndim=N),
)

equal = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    # 25,
]

i = 1
for problem_thefittest, problem_opfunu in zip(
    problems_dict_thefittest.values(), problems_dict_opfunu
):
    if i in equal:
        i += 1
        continue
    if i == 25:

        population_ph = np.array(
            [
                np.random.uniform(
                    problem_thefittest["bounds"][0], problem_thefittest["bounds"][1], size=10
                )
                for _ in range(N)
            ],
            dtype=np.float32,
        ).T
        fitness_thefittest = problem_thefittest["function"]()(population_ph)
        fitness_opfunu = np.array(
            [problem_opfunu.evaluate(population_ph_i) for population_ph_i in population_ph]
        )
        fitness_equal_cond = np.all(np.isclose(fitness_thefittest, fitness_opfunu))

        try:
            x_equal_cond = np.isclose(
                problem_thefittest["function"]().x_shift[:N], problem_opfunu.x_global
            )

        except ValueError:
            x_equal_cond = np.isclose(
                problem_thefittest["function"]().x_shift[0][:N], problem_opfunu.x_global
            )

        bounds_equal_cond = np.all(
            [
                np.all(problem_thefittest["bounds"][0] == problem_opfunu.bounds[:, 0]),
                np.all(problem_thefittest["bounds"][1] == problem_opfunu.bounds[:, 1]),
            ]
        )

        print("function:", i)
        print(problem_thefittest["function"]().x_shift[0][:N])
        print(problem_opfunu.x_global)
        print("x_equal_cond:", x_equal_cond)
        print("bounds_equal_cond:", bounds_equal_cond)
        print("fitness_equal_cond:", fitness_equal_cond)
        print(fitness_thefittest)
        print(fitness_opfunu)

    i += 1
