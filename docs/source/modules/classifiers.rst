classifiers
===========

The library provides several classifier implementations based on evolutionary algorithms. These classifiers can learn complex decision boundaries, evolve neural network architectures, and optimize network weights using evolutionary strategies.

Genetic Programming Classifiers
---------------------------------

Genetic Programming classifiers evolve symbolic expressions or tree structures to perform classification. They can discover interpretable decision rules and handle non-linear separations.

*Reference:* Koza, J. R. (1993). Genetic Programming - On the Programming of Computers by Means of Natural Selection. Complex Adaptive Systems.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Classifier
     - Description
   * - :class:`~thefittest.classifiers.GeneticProgrammingClassifier`
     - GP-based classifier evolving symbolic expressions for decision boundaries

GeneticProgrammingClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.classifiers.GeneticProgrammingClassifier
   :members:
   :inherited-members:
   :show-inheritance:

Neural Network Classifiers
---------------------------

Neural network classifiers combine traditional neural architectures with evolutionary optimization. Instead of gradient descent, they use evolutionary algorithms to train networks or evolve architectures.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Classifier
     - Description
   * - :class:`~thefittest.classifiers.MLPEAClassifier`
     - Multi-Layer Perceptron with evolutionary algorithm-based weight optimization (`Cotta et al., 2002 <http://dx.doi.org/10.1007/978-1-4615-1539-5_18>`_)
   * - :class:`~thefittest.classifiers.GeneticProgrammingNeuralNetClassifier`
     - Neural network with GP-evolved architecture and EA-optimized weights (Lipinsky & Semenkin, 2006)

MLPEAClassifier
~~~~~~~~~~~~~~~

.. autoclass:: thefittest.classifiers.MLPEAClassifier
   :members:
   :inherited-members:
   :show-inheritance:

GeneticProgrammingNeuralNetClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.classifiers.GeneticProgrammingNeuralNetClassifier
   :members:
   :inherited-members:
   :show-inheritance: