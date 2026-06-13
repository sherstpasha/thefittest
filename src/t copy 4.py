import traceback

from thefittest.benchmarks import Rastrigin
from thefittest.optimizers import SHAGA
from thefittest.utils.transformations import GrayCode


n_dimension = 10
left_border = -5.0
right_border = 5.0
number_of_generations = 2
population_size = 500

fixed_selection = "tournament_5"
fixed_crossover = "one_point_1"

selections = (
    "proportional",
    "rank",
    "tournament_k",
    "tournament_2",
    "tournament_3",
    "tournament_5",
    "tournament_7",
)

crossovers = (
    "empty",
    "one_point_1",
    "one_point_2",
    "one_point_7",
    "one_point_k",
    "one_point_prop_2",
    "one_point_prop_7",
    "one_point_prop_k",
    "one_point_rank_2",
    "one_point_rank_7",
    "one_point_rank_k",
    "one_point_tour_3",
    "one_point_tour_7",
    "one_point_tour_k",
    "uniform_1",
    "uniform_2",
    "uniform_7",
    "uniform_k",
    "uniform_prop_2",
    "uniform_prop_7",
    "uniform_prop_k",
    "uniform_rank_2",
    "uniform_rank_7",
    "uniform_rank_k",
    "uniform_tour_3",
    "uniform_tour_7",
    "uniform_tour_k",
)

genotype_to_phenotype = GrayCode()
genotype_to_phenotype.fit(
    left_border=left_border,
    right_border=right_border,
    num_variables=n_dimension,
    h_per_variable=0.001,
)
num_bits = genotype_to_phenotype.get_bits_per_variable().sum()


def run_case(selection, crossover):
    optimizer = SHAGA(
        fitness_function=Rastrigin(),
        genotype_to_phenotype=genotype_to_phenotype.transform,
        iters=number_of_generations,
        pop_size=population_size,
        str_len=num_bits,
        selection=selection,
        crossover=crossover,
        show_progress_each=None,
        minimization=True,
        optimal_value=0.0,
        keep_history=False,
        random_state=42,
    )
    optimizer.fit()
    return optimizer.get_fittest()["fitness"]


def check_cases(title, cases):
    print(title)
    failed = []
    for selection, crossover in cases:
        name = f"selection={selection}, crossover={crossover}"
        try:
            fitness = run_case(selection, crossover)
        except Exception as exc:
            failed.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"[OK]   {name}: best fitness = {fitness}")

    if failed:
        print("\nFailed cases:")
        for name in failed:
            print(f"  {name}")
    else:
        print("\nAll cases passed.")

    return failed


selection_cases = [(selection, fixed_crossover) for selection in selections]
crossover_cases = [(fixed_selection, crossover) for crossover in crossovers]

failed_selection_cases = check_cases("Checking selections with fixed crossover", selection_cases)
print()
failed_crossover_cases = check_cases("Checking crossovers with fixed selection", crossover_cases)

if failed_selection_cases or failed_crossover_cases:
    raise SystemExit(1)
