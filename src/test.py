from thefittest.utils.selections import proportional_selection
import numpy as np

# Example
fitness_values = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
rank_values = np.array([3, 2, 4, 1], dtype=np.float64)
tournament_size = 2
num_selected = 2
selected_individuals = proportional_selection(
    fitness_values, rank_values, tournament_size, num_selected
)
print("Selected Individuals:", selected_individuals)
