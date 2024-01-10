import numpy as np

from thefittest.utils.transformations import SamplingGrid


def test_sampling_grid_fit():
    # Test fitting the sampling grid with step size
    grid = SamplingGrid()
    grid.fit(left_border=-1.0, right_border=1.0, num_variables=3, h_per_variable=0.1)
    assert np.allclose(grid.get_left_border(), np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    assert np.allclose(grid.get_right_border(), np.array([1.0, 1.0, 1.0], dtype=np.float64))
    assert grid.get_num_variables() == 3
    assert np.allclose(grid.get_h_per_variable(), np.array([0.06451613, 0.06451613, 0.06451613]))
    assert np.allclose(grid.get_bits_per_variable(), np.array([5, 5, 5], dtype=np.int64))

    # # Test fitting the sampling grid with bits per variable
    # grid.fit(left_border=-1.0, right_border=1.0, num_variables=3, bits_per_variable=10)
    # assert np.array_equal(grid.get_left_border(), np.array([-1.0, -1.0, -1.0]))
    # assert np.array_equal(grid.get_right_border(), np.array([1.0, 1.0, 1.0]))
    # assert grid.get_num_variables() == 3
    # assert np.array_equal(grid.get_h_per_variable(), np.array([0.00097656, 0.00097656, 0.00097656]))
    # assert np.array_equal(grid.get_bits_per_variable(), np.array([10, 10, 10]))


# def test_sampling_grid_transform():
#     # Test transforming binary population to floating-point array
#     grid = SamplingGrid()
#     grid.fit(left_border=-1.0, right_border=1.0, num_variables=3, h_per_variable=0.1)
#     binary_population = np.random.randint(
#         2, size=(5, grid.get_bits_per_variable().sum()), dtype=np.int8
#     )
#     transformed_population = grid.transform(binary_population)
#     assert transformed_population.shape == (5, 3)


# def test_sampling_grid_inverse_transform():
#     # Test inverse transforming floating-point population to binary array
#     grid = SamplingGrid()
#     grid.fit(left_border=-1.0, right_border=1.0, num_variables=3, h_per_variable=0.1)
#     floating_population = np.random.rand(5, 3)
#     inverse_transformed_population = grid.inverse_transform(floating_population)
#     assert inverse_transformed_population.shape == (5, grid.get_bits_per_variable().sum())
