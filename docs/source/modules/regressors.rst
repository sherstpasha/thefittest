regressors
==========

The library provides several regressor implementations based on evolutionary algorithms. These regressors can perform symbolic regression, optimize neural network weights, and evolve network architectures for continuous value prediction.

Genetic Programming Regressors
--------------------------------

Genetic Programming regressors evolve symbolic expressions or tree structures to perform regression. They can discover interpretable mathematical models and handle complex non-linear relationships.

*Reference:* Koza, J. R. (1993). Genetic Programming - On the Programming of Computers by Means of Natural Selection. Complex Adaptive Systems.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Regressor
     - Description
   * - :class:`~thefittest.regressors.GeneticProgrammingRegressor`
     - GP-based regressor evolving symbolic expressions for explicit functional relationships

GeneticProgrammingRegressor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.regressors.GeneticProgrammingRegressor
   :members:
   :inherited-members:
   :show-inheritance:

Neural Network Regressors
--------------------------

Neural network regressors combine traditional neural architectures with evolutionary optimization. Instead of gradient descent, they use evolutionary algorithms to train networks or evolve architectures.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Regressor
     - Description
   * - :class:`~thefittest.regressors.MLPEARegressor`
     - Multi-Layer Perceptron with evolutionary algorithm-based weight optimization (`Cotta et al., 2002 <http://dx.doi.org/10.1007/978-1-4615-1539-5_18>`_)
   * - :class:`~thefittest.regressors.GeneticProgrammingNeuralNetRegressor`
     - Neural network with GP-evolved architecture and EA-optimized weights (Lipinsky & Semenkin, 2006)

MLPEARegressor
~~~~~~~~~~~~~~

.. autoclass:: thefittest.regressors.MLPEARegressor
   :members:
   :inherited-members:
   :show-inheritance:

GeneticProgrammingNeuralNetRegressor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.regressors.GeneticProgrammingNeuralNetRegressor
   :members:
   :inherited-members:
   :show-inheritance: