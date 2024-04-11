from __future__ import annotations

from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..utils import if_single_or_array_to_float_array
from ..utils import if_single_or_array_to_int_array


def minmax_scale(data: NDArray[Union[np.int64, np.float64]]) -> NDArray[np.float64]:
    """
    Scale the values of a NumPy array between 0 and 1.

    Parameters
    ----------
    data : NDArray[Union[np.int64, np.float64]]
        Input array containing numerical values to be scaled.

    Returns
    -------
    NDArray[np.float64]
        Scaled array with values between 0 and 1.

    Notes
    -----
    This function scales the values of the input array between 0 and 1 using min-max scaling.
    If the minimum and maximum values in the array are equal, the function returns an array of ones.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.transformations import minmax_scale
    >>>
    >>> # Example data
    >>> example_data = np.array([2, 5, 10, 8, 3], dtype=np.int64)
    >>>
    >>> # Scale the data using the minmax_scale function
    >>> scaled_data = minmax_scale(example_data)
    >>>
    >>> # Display original and scaled data
    >>> print("Original Data:", example_data)
    Original Data: [ 2  5 10  8  3]
    >>> print("Scaled Data:", scaled_data)
    Scaled Data: [0.    0.375 1.    0.75  0.125]
    """
    data_copy = data.copy()
    max_value = data_copy.max()
    min_value = data_copy.min()
    if max_value == min_value:
        scaled_data = np.ones_like(data_copy, dtype=np.float64)
    else:
        scaled_data = ((data_copy - min_value) / (max_value - min_value)).astype(np.float64)
    return scaled_data


class SamplingGrid:
    """
    SamplingGrid class for transforming populations between binary and floating-point representations.

    This class provides functionality to fit, transform, and inverse transform populations using a specified
    sampling grid. The grid is defined by the left and right borders for each variable, and either the step size (h)
    or the number of bits for each variable.

    Attributes
    ----------
    _left_border : NDArray[np.float64]
        Left border values for each variable in the sampling grid.
    _right_border : NDArray[np.float64]
        Right border values for each variable in the sampling grid.
    _num_variables : int
        Number of variables in the sampling grid.
    _h_per_variable : NDArray[np.float64]
        Step size for each variable in the sampling grid.
    _bits_per_variable : NDArray[np.int64]
        Number of bits for each variable in the sampling grid.
    _reversed_powers : NDArray[np.int64]
        Reversed powers of 2 used for converting binary representations to integers.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.transformations import SamplingGrid
    >>>
    >>> # Fit the sampling grid
    >>> grid = SamplingGrid()
    >>> grid.fit(left_border=-5.0, right_border=5.0, num_variables=3, h_per_variable=0.1)
    <thefittest.utils.transformations.SamplingGrid object at ...>
    >>>
    >>> # Generate a binary population
    >>> string_length = grid.get_bits_per_variable().sum()
    >>> binary_population = np.random.randint(2, size=(5, string_length), dtype=np.int8)
    >>> print("Binary Population:", binary_population)
    Binary Population: ...
    >>>
    >>> # Transform the binary population to a floating-point array
    >>> transformed_population = grid.transform(binary_population)
    >>> print("Transformed Population:", transformed_population)
    Transformed Population:...

    >>> import numpy as np
    >>> from thefittest.utils.transformations import SamplingGrid
    >>>
    >>> # Fit the sampling grid
    >>> grid = SamplingGrid()
    >>> grid.fit(left_border=0.0, right_border=1.0, num_variables=3, h_per_variable=0.1)
    <thefittest.utils.transformations.SamplingGrid object at ...>
    >>>
    >>> # Generate a floating-point population
    >>> floating_population = np.random.rand(5, 3)
    >>> print("Floating-point Population:", floating_population)
    Floating-point Population:...
    >>>
    >>> # Inverse transform the floating-point population to a binary array
    >>> inverse_transformed_population = grid.inverse_transform(floating_population)
    >>> print("Inverse Transformed Population:", inverse_transformed_population)
    Inverse Transformed Population: ...

    Methods
    -------
    fit(
        left_border: Union[float, NDArray[np.float64]],
        right_border: Union[float, NDArray[np.float64]],
        num_variables: int,
        h_per_variable: Optional[Union[float, NDArray[np.float64]]] = None,
        bits_per_variable: Optional[Union[int, NDArray[np.int64]]] = None,
    ) -> "SamplingGrid":
        Fit the sampling grid using specified parameters.

    get_left_border() -> NDArray[np.float64]:
        Get the left border values for each variable.

    get_right_border() -> NDArray[np.float64]:
        Get the right border values for each variable.

    get_num_variables() -> int:
        Get the number of variables.

    get_h_per_variable() -> NDArray[np.float64]:
        Get the step size values for each variable.

    get_bits_per_variable() -> NDArray[np.int64]:
        Get the number of bits per variable.

    transform(population: NDArray[np.int8]) -> NDArray[np.float64]:
        Transform a binary population into a floating-point array based on the SamplingGrid parameters.

    inverse_transform(population: NDArray[np.float64]) -> NDArray[np.int8]:
        Inverse transform a floating-point population into a binary array based on the SamplingGrid parameters.
    """

    def __init__(
        self,
    ) -> None:
        self._left_border: NDArray[np.float64]
        self._right_border: NDArray[np.float64]
        self._num_variables: int
        self._h_per_variable: NDArray[np.float64]
        self._bits_per_variable: NDArray[np.int64]
        self._reversed_powers: NDArray[np.int64]

    def get_left_border(self) -> NDArray[np.float64]:
        """
        Get the left border values for each variable.

        Returns
        -------
        NDArray[np.float64]
            Left border values for each variable.
        """
        return self._left_border

    def get_right_border(self) -> NDArray[np.float64]:
        """
        Get the right border values for each variable.

        Returns
        -------
        NDArray[np.float64]
            Right border values for each variable.
        """
        return self._right_border

    def get_num_variables(self) -> int:
        """
        Get the number of variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self._num_variables

    def get_h_per_variable(self) -> NDArray[np.float64]:
        """
        Get the step size values for each variable.

        Returns
        -------
        NDArray[np.float64]
            Step size values for each variable.
        """
        return self._h_per_variable

    def get_bits_per_variable(self) -> NDArray[np.int64]:
        """
        Get the number of bits per variable.

        Returns
        -------
        NDArray[np.int64]
            Number of bits per variable.
        """
        return self._bits_per_variable

    def fit(
        self,
        left_border: Union[float, NDArray[np.float64]],
        right_border: Union[float, NDArray[np.float64]],
        num_variables: int,
        h_per_variable: Optional[Union[float, NDArray[np.float64]]] = None,
        bits_per_variable: Optional[Union[int, NDArray[np.int64]]] = None,
    ):
        """
        Fit the sampling grid using specified parameters.

        Parameters
        ----------
        left_border : Union[float, NDArray[np.float64]]
            Left border values for each variable.
        right_border : Union[float, NDArray[np.float64]]
            Right border values for each variable.
        num_variables : int
            Number of variables.
        h_per_variable : Optional[Union[float, NDArray[np.float64]]], optional
            Step size values for each variable. Either `h_per_variable` or `bits_per_variable` should be provided.
        bits_per_variable : Optional[Union[int, NDArray[np.int64]]], optional
            Number of bits per variable. Either `h_per_variable` or `bits_per_variable` should be provided.

        Returns
        -------
        SamplingGrid
            The fitted SamplingGrid instance.

        Raises
        ------
        AssertionError
            If both `h_per_variable` and `bits_per_variable` are provided or if neither is provided.

        Notes
        -----
        This method fits the sampling grid using the specified parameters. If `h_per_variable` is provided,
        it calculates the corresponding `bits_per_variable`. If `bits_per_variable` is provided, it calculates
        the corresponding `h_per_variable`. The powers of 2 used for conversion are also calculated and stored.

        Examples
        --------
        >>> import numpy as np
        >>> from thefittest.utils.transformations import SamplingGrid
        >>>
        >>> grid = SamplingGrid()
        >>> grid.fit(left_border=0.0, right_border=1.0, num_variables=3, h_per_variable=0.1)
        <thefittest.utils.transformations.SamplingGrid object at ...>
        >>> print("Grid Left Border:", grid.get_left_border())
        Grid Left Border: [0. 0. 0.]
        >>> print("Grid Right Border:", grid.get_right_border())
        Grid Right Border: [1. 1. 1.]
        >>> print("Number of Variables:", grid.get_num_variables())
        Number of Variables: 3
        >>> print("Step Size per Variable:", grid.get_h_per_variable())
        Step Size per Variable: [0.06666667 0.06666667 0.06666667]
        >>> print("Bits per Variable:", grid.get_bits_per_variable())
        Bits per Variable: [4 4 4]
        >>>
        >>> grid = SamplingGrid()
        >>> grid.fit(left_border=-1.0, right_border=1.0, num_variables=2, bits_per_variable=4)
        <thefittest.utils.transformations.SamplingGrid object at ...>
        >>> print("Grid Left Border:", grid.get_left_border())
        Grid Left Border: [-1. -1.]
        >>> print("Grid Right Border:", grid.get_right_border())
        Grid Right Border: [1. 1.]
        >>> print("Number of Variables:", grid.get_num_variables())
        Number of Variables: 2
        >>> print("Step Size per Variable:", grid.get_h_per_variable())
        Step Size per Variable: [0.13333333 0.13333333]
        >>> print("Bits per Variable:", grid.get_bits_per_variable())
        Bits per Variable: [4 4]
        >>>
        >>> grid = SamplingGrid()
        >>> grid.fit(
        ...     left_border=np.array([-1.0, 0.5, -2.0], dtype=np.float64),
        ...     right_border=np.array([1.0, 5.0, 2.0], dtype=np.float64),
        ...     num_variables=3,
        ...     h_per_variable=np.array([0.05, 1.0, 0.1], dtype=np.float64),
        ... )
        <thefittest.utils.transformations.SamplingGrid object at ...>
        >>> print("Grid Left Border:", grid.get_left_border())
        Grid Left Border: [-1.   0.5 -2. ]
        >>> print("Grid Right Border:", grid.get_right_border())
        Grid Right Border: [1. 5. 2.]
        >>> print("Number of Variables:", grid.get_num_variables())
        Number of Variables: 3
        >>> print("Step Size per Variable:", grid.get_h_per_variable())
        Step Size per Variable: [0.03174603 0.64285714 0.06349206]
        >>> print("Bits per Variable:", grid.get_bits_per_variable())
        Bits per Variable: [6 3 6]
        >>>
        >>> grid = SamplingGrid()
        >>> grid.fit(
        ...     left_border=np.array([-3.5, -2.0, 10.0, 0.9], dtype=np.float64),
        ...     right_border=np.array([3.5, 7.0, 25.0, 1.5], dtype=np.float64),
        ...     num_variables=4,
        ...     bits_per_variable=np.array([8, 16, 3, 40], dtype=np.int64),
        ... )
        <thefittest.utils.transformations.SamplingGrid object at ...>
        >>> print("Grid Left Border:", grid.get_left_border())
        Grid Left Border: [-3.5 -2.  10.   0.9]
        >>> print("Grid Right Border:", grid.get_right_border())
        Grid Right Border: [ 3.5  7.  25.   1.5]
        >>> print("Number of Variables:", grid.get_num_variables())
        Number of Variables: 4
        >>> print("Step Size per Variable:", grid.get_h_per_variable())
        Step Size per Variable: [2.74509804e-02 1.37331197e-04 2.14285714e+00 5.45696821e-13]
        >>> print("Bits per Variable:", grid.get_bits_per_variable())
        Bits per Variable: [ 8 16  3 40]
        """
        self._num_variables = num_variables
        self._left_border = if_single_or_array_to_float_array(left_border, self._num_variables)
        self._right_border = if_single_or_array_to_float_array(right_border, self._num_variables)

        if bits_per_variable is None and h_per_variable is not None:
            self._h_per_variable = if_single_or_array_to_float_array(h_per_variable)
            self._culc_num_bits_from_h()
            self._culc_h_from_num_bits()
        elif bits_per_variable is not None and h_per_variable is None:
            self._bits_per_variable = if_single_or_array_to_int_array(
                bits_per_variable, self._num_variables
            )
            self._culc_h_from_num_bits()
        else:
            raise ValueError(
                "Either bits_per_variable or h_per_variable must be defined, but not both."
            )

        self._powers = 2 ** np.arange(self._bits_per_variable.max(), dtype=np.int64)

        return self

    def _culc_h_from_num_bits(self) -> None:
        """
        Calculate the step size (h) for each variable based on the number of bits.

        This method calculates the step size (h) for each variable using the specified number of bits per variable,
        the left and right borders of the sampling grid, and updates the internal state of the SamplingGrid.

        Notes
        -----
        This method should be called internally during the fitting process after setting the number of bits per variable.
        """
        self._h_per_variable = (self._right_border - self._left_border) / (
            2.0**self._bits_per_variable - 1
        )

    def _culc_num_bits_from_h(self) -> None:
        """
        Calculate the number of bits for each variable based on the step size (h).

        This method calculates the number of bits required for each variable based on the specified step size (h),
        the left and right borders of the sampling grid, and updates the internal state of the SamplingGrid.

        Notes
        -----
        This method should be called internally during the fitting process after setting the step size (h) per variable.
        """
        self._bits_per_variable = np.ceil(
            np.log2((self._right_border - self._left_border) / self._h_per_variable + 1)
        ).astype(int)

    @staticmethod
    def bit_to_int(
        bit_array: NDArray[np.int64], powers: Optional[NDArray[np.int64]] = None
    ) -> NDArray[np.int64]:
        """
        Convert a binary array to an integer array using specified powers.

        Parameters
        ----------
        bit_array : NDArray[np.int64]
            2D array where each row represents a binary number.
        powers : Optional[NDArray[np.int64]], optional
            1D array of powers of 2 corresponding to the binary places. If provided, avoids recalculation.

        Returns
        -------
        NDArray[np.int64]
            1D array representing the integer values converted from binary.

        Examples
        --------
        >>> import numpy as np
        >>> from thefittest.utils.transformations import SamplingGrid
        >>>
        >>> # Example 1: Convert binary array to integer array
        >>> binary_array = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int64)
        >>> result = SamplingGrid.bit_to_int(binary_array)
        >>> print("Converted Integer Array:", result)
        Converted Integer Array: [5 3]
        >>>
        >>> # Example 2: Convert binary array to integer array with  powers
        >>> custom_powers = np.array([1, 2, 4], dtype=np.int64)
        >>> result_custom_powers = SamplingGrid.bit_to_int(binary_array, powers=custom_powers)
        >>> print("Converted Integer Array (Define Powers):", result_custom_powers)
        Converted Integer Array (Define Powers): [5 3]
        """
        num_bits = bit_array.shape[1]
        if powers is None:
            powers = 2 ** np.arange(num_bits, dtype=np.int64)
        reversed_powers = np.flip(powers[:num_bits])
        int_array = np.dot(bit_array, reversed_powers)
        return int_array

    @staticmethod
    def int_to_bit(
        int_array: NDArray[np.int64], powers: Optional[NDArray[np.int64]] = None
    ) -> NDArray[np.int8]:
        """
        Convert a 1D integer array to a 2D binary array.

        Parameters
        ----------
        int_array : NDArray[np.int64]
            1D array of integers to be converted to binary.
        powers : Optional[NDArray[np.int64]], optional
            1D array of powers of 2 corresponding to the binary places. If provided, avoids recalculation.

        Returns
        -------
        NDArray[np.byte]
            2D binary array converted from the 1D integer array.

        Examples
        --------
        >>> import numpy as np
        >>> from thefittest.utils.transformations import SamplingGrid
        >>>
        >>> # Example 1: Convert one-dimensional integer array to binary array
        >>> integer_array = np.array([5, 3], dtype=np.int64)
        >>> result = SamplingGrid.int_to_bit(integer_array)
        >>> print("Converted Binary Array:", result)
        Converted Binary Array: ...
        >>>
        >>> # Example 2: Convert one-dimensional integer array to binary array with powers
        >>> custom_powers = np.array([1, 2, 4], dtype=np.int64)
        >>> integer_array = np.array([5, 3], dtype=np.int64)
        >>> result_custom_powers = SamplingGrid.int_to_bit(integer_array, powers=custom_powers)
        >>> print("Converted Binary Array (Define Powers):", result_custom_powers)
        Converted Binary Array (Define Powers): ...
        """
        num_bits = int(np.ceil(np.log2(np.max(int_array) + 1)))
        bit_array = np.empty(shape=(int_array.shape[0], num_bits), dtype=np.int8)

        if powers is None:
            powers = 2 ** np.arange(num_bits, dtype=np.int64)

        reversed_powers = np.flip(powers[:num_bits])

        int_array = int_array.astype(np.int64)

        for i, reversed_power_i in enumerate(reversed_powers):
            bit_array[:, i] = np.int8((int_array & reversed_power_i) > 0)
        return bit_array

    def _float_to_bit(
        self, float_array: NDArray[np.float64], left: NDArray[np.float64], h: NDArray[np.float64]
    ) -> NDArray[np.int8]:
        """
        Convert a 1D floating-point array to a 2D binary array based on the SamplingGrid parameters.

        Parameters
        ----------
        float_array : NDArray[np.float64]
            1D floating-point array to be converted to binary.
        left : NDArray[np.float64]
            1D array representing the left boundary of the sampling grid used in the conversion process.
        h : NDArray[np.float64]
            1D array representing the step size of the sampling grid used in the conversion process.

        Returns
        -------
        NDArray[np.byte]
            2D binary array converted from the 1D floating-point array.

        Notes
        -----
        This function internally relies on the `int_to_bit` method to convert floating-point numbers to their binary representation.
        """
        grid_number = (float_array - left) / h
        int_array = np.rint(grid_number)
        bit_array = self.int_to_bit(int_array, self._powers)
        return bit_array

    def _decode(self, bit_array_i: NDArray[np.byte]) -> NDArray[np.int64]:
        """
        Decode a 2D binary array representing a single variable into a 1D integer array.

        Parameters
        ----------
        bit_array_i : NDArray[np.byte]
            2D binary array representing the variable. Each row corresponds to a binary representation.

        Returns
        -------
        NDArray[np.int64]
            1D integer array decoded from the 2D binary representation.

        Notes
        -----
        This function internally relies on the `bit_to_int` method for converting the binary representation
        of a variable to its integer counterpart.

        """
        int_convert = self.bit_to_int(bit_array_i, self._powers)
        return int_convert

    def transform(self, population: NDArray[np.int8]) -> NDArray[np.float64]:
        """
        Transform a binary population into a floating-point array based on the SamplingGrid parameters.

        Parameters
        ----------
        population : NDArray[np.int8]
            Population matrix where each row represents a binary array.

        Returns
        -------
        NDArray[np.float64]
            Floating-point array representing the transformed population.

        Notes
        -----
        This function divides the input population into individual variables, decodes each variable from binary to integer,
        and then calculates the corresponding floating-point values using the SamplingGrid parameters.

        Examples
        --------
        >>> import numpy as np
        >>> from thefittest.utils.transformations import SamplingGrid
        >>>
        >>> # Fit the sampling grid
        >>> grid = SamplingGrid()
        >>> grid.fit(left_border=-5.0, right_border=5.0, num_variables=3, h_per_variable=0.1)
        <thefittest.utils.transformations.SamplingGrid object at ...>
        >>>
        >>> # Generate a binary population
        >>> string_length = grid.get_bits_per_variable().sum()
        >>> binary_population = np.random.randint(2, size=(5, string_length), dtype=np.int8)
        >>> print("Binary Population:", binary_population)
        Binary Population: ...
        >>>
        >>> # Transform the binary population to a floating-point array
        >>> transformed_population = grid.transform(binary_population)
        >>> print("Transformed Population:", transformed_population)
        Transformed Population: ...
        """
        splits = np.add.accumulate(self._bits_per_variable)
        p_parts = np.split(population, splits[:-1], axis=1)

        int_array = np.array(list(map(self._decode, p_parts))).T
        float_array = (
            self._left_border[np.newaxis, :] + self._h_per_variable[np.newaxis, :] * int_array
        )
        return float_array

    def inverse_transform(self, population: NDArray[np.float64]) -> NDArray[np.int8]:
        """
        Inverse transform a floating-point population into a binary array based on the SamplingGrid parameters.

        Parameters
        ----------
        population : NDArray[np.float64]
            Population matrix where each row represents a floating-point array.

        Returns
        -------
        NDArray[np.int8]
            Binary array representing the inverse transformed population.

        Notes
        -----
        This function encodes each variable from floating-point to binary, and then combines the binary representations
        to form the binary array of the inverse transformed population.

        Examples
        --------
        >>> import numpy as np
        >>> from thefittest.utils.transformations import SamplingGrid
        >>>
        >>> # Fit the sampling grid
        >>> grid = SamplingGrid()
        >>> grid.fit(left_border=0.0, right_border=1.0, num_variables=3, h_per_variable=0.1)
        <thefittest.utils.transformations.SamplingGrid object at ...>
        >>>
        >>> # Generate a floating-point population
        >>> floating_population = np.random.rand(5, 3)
        >>> print("Floating-point Population:", floating_population)
        Floating-point Population: ...
        >>>
        >>> # Inverse transform the floating-point population to a binary array
        >>> inverse_transformed_population = grid.inverse_transform(floating_population)
        >>> print("Inverse Transformed Population:", inverse_transformed_population)
        Inverse Transformed Population: ...
        """
        map_ = map(self._float_to_bit, population.T, self._left_border, self._h_per_variable)
        bit_array = np.hstack(list(map_))
        return bit_array


class GrayCode(SamplingGrid):
    """
    GrayCode class for transforming populations between gray code and floating-point representations.

    This class extends the functionality of the SamplingGrid for gray code transformations. Gray code is a binary numeral
    system where two successive values differ in only one bit. GrayCode provides methods to convert between binary and gray
    code representations, as well as transforming populations between gray code and floating-point representations using
    the specified sampling grid.

    Attributes
    ----------
    Inherits attributes from SamplingGrid.

    Methods
    -------
    gray_to_bit(gray_array: NDArray[np.byte]) -> NDArray[np.byte]:
        Convert a gray code array to a binary array.

    bit_to_gray(bit_array: NDArray[np.byte]) -> NDArray[np.byte]:
        Convert a binary array to a gray code array.

    _decode(gray_array_i: NDArray[np.byte]) -> NDArray[np.int64]:
        Decode a 2D gray code array representing a single variable into a 1D integer array.

    _float_to_bit(float_array: NDArray[np.float64], left: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.byte]:
        Convert a 1D floating-point array to a 2D gray code array based on the SamplingGrid parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.transformations import GrayCode
    >>>
    >>> # Fit the sampling grid with gray code transformation
    >>> grid = GrayCode()
    >>> grid.fit(left_border=-5.0, right_border=5.0, num_variables=3, h_per_variable=0.1)
    <thefittest.utils.transformations.GrayCode object at ...>
    >>>
    >>> # Generate a binary population using gray code
    >>> string_length = grid.get_bits_per_variable().sum()
    >>> gray_population = np.random.randint(2, size=(5, string_length), dtype=np.byte)
    >>> print("Gray Code Population:", gray_population)
    Gray Code Population: ...
    >>>
    >>> # Transform the gray code population to a floating-point array
    >>> transformed_population = grid.transform(gray_population)
    >>> print("Transformed Population:", transformed_population)
    Transformed Population: ...

    >>> # Generate a floating-point population
    >>> floating_population = np.random.rand(5, 3)
    >>> print("Floating-point Population:", floating_population)
    Floating-point Population: ...
    >>>
    >>> # Inverse transform the floating-point population to a gray code array
    >>> inverse_transformed_population = grid.inverse_transform(floating_population)
    >>> print("Inverse Transformed Population (Gray Code):", inverse_transformed_population)
    Inverse Transformed Population (Gray Code): ...
    """

    def __init__(self) -> None:
        SamplingGrid.__init__(self)

    @staticmethod
    def gray_to_bit(gray_array: NDArray[np.byte]) -> NDArray[np.byte]:
        """
        Convert a gray code array to a binary array.

        Parameters
        ----------
        gray_array : NDArray[np.byte]
            2D array where each row represents a gray code number.

        Returns
        -------
        NDArray[np.byte]
            2D binary array converted from the gray code array.

        Examples
        --------
        >>> import numpy as np
        >>> from thefittest.utils.transformations import GrayCode
        >>>
        >>> # Example: Convert gray code array to binary array using GrayCode.gray_to_bit method
        >>> gray_array = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.byte)
        >>> result = GrayCode.gray_to_bit(gray_array)
        >>> print("Converted Binary Array:", result)
        Converted Binary Array: ...
        """
        bit_array = np.logical_xor.accumulate(gray_array, axis=-1).astype(np.byte)
        return bit_array

    @staticmethod
    def bit_to_gray(bit_array: NDArray[np.byte]) -> NDArray[np.byte]:
        """
        Convert a binary array to a gray code array.

        Parameters
        ----------
        bit_array : NDArray[np.byte]
            2D binary array where each row represents a binary number.

        Returns
        -------
        NDArray[np.byte]
            2D gray code array converted from the binary array.

        Examples
        --------
        >>> import numpy as np
        >>> from thefittest.utils.transformations import GrayCode
        >>>
        >>> # Example: Convert binary array to gray code array using GrayCode.bit_to_gray method
        >>> binary_array = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.byte)
        >>> result = GrayCode.bit_to_gray(binary_array)
        >>> print("Converted Gray Code Array:", result)
        Converted Gray Code Array: ...
        """
        cut_gray = np.logical_xor(bit_array[:, :-1], bit_array[:, 1:])
        gray_array = np.hstack([bit_array[:, 0].reshape(-1, 1), cut_gray])
        return gray_array

    def _decode(self, gray_array_i: NDArray[np.byte]) -> NDArray[np.int64]:
        """
        Decode a 2D gray code array representing a single variable into a 1D integer array.

        Parameters
        ----------
        gray_array_i : NDArray[np.byte]
            2D gray code array representing the variable. Each row corresponds to a gray code representation.

        Returns
        -------
        NDArray[np.int64]
            1D integer array decoded from the 2D gray code representation.

        """
        bit_array_i = self.gray_to_bit(gray_array_i)
        int_convert = self.bit_to_int(bit_array_i, self._powers)
        return int_convert

    def _float_to_bit(
        self, float_array: NDArray[np.float64], left: NDArray[np.float64], h: NDArray[np.float64]
    ) -> NDArray[np.byte]:
        """
        Convert a 1D floating-point array to a 2D gray code array based on the GrayCode parameters.

        Parameters
        ----------
        float_array : NDArray[np.float64]
            1D floating-point array to be converted to gray code.
        left : NDArray[np.float64]
            1D array representing the left boundary of the sampling grid used in the conversion process.
        h : NDArray[np.float64]
            1D array representing the step size of the sampling grid used in the conversion process.

        Returns
        -------
        NDArray[np.byte]
            2D gray code array converted from the 1D floating-point array.
        """
        grid_number = (float_array - left) / h
        int_array = np.rint(grid_number)
        bit_array = self.int_to_bit(int_array)
        gray_array = self.bit_to_gray(bit_array)
        return gray_array
