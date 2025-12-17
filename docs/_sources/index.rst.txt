.. Thefittest documentation master file, created by
   sphinx-quickstart on Sun Dec 24 00:35:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Thefittest's documentation!
======================================
|

.. image:: logos/logo1.png
   :align: center

|

``thefittest`` is an open-source library designed for the efficient application of classical evolutionary algorithms and their effective modifications in optimization and machine learning. Our project aims to provide performance, accessibility, and ease of use, opening up the world of advanced evolutionary methods to you.

Optimizers
----------

The library provides a comprehensive set of evolutionary optimization algorithms for solving continuous and discrete optimization problems. Each optimizer is designed with flexibility in mind, supporting various selection methods, crossover operators, and mutation strategies.

**Differential Evolution**

Differential Evolution is a population-based stochastic optimization method that operates on real-valued vectors. It uses vector differences for generating new candidate solutions and has proven effective for continuous optimization problems.

*Reference:* Storn, R., & Price, K. (1995). Differential Evolution: A Simple and Efficient Adaptive Scheme for Global Optimization Over Continuous Spaces. Journal of Global Optimization, 23.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Algorithm
     - Description
   * - :class:`~thefittest.optimizers.DifferentialEvolution`
     - Classical Differential Evolution algorithm with multiple mutation strategies
   * - :class:`~thefittest.optimizers.jDE`
     - Self-adaptive Differential Evolution with dynamic parameter control (`Brest et al., 2007 <http://dx.doi.org/10.1109/TEVC.2009.2014613>`_)
   * - :class:`~thefittest.optimizers.SHADE`
     - Success-History based Adaptive Differential Evolution with parameter adaptation (`Tanabe & Fukunaga, 2013 <https://doi.org/10.1109/CEC.2013.6557555>`_)

**Genetic Algorithms**

Genetic Algorithms are search heuristics inspired by natural selection. They work with binary string representations and use selection, crossover, and mutation operators to evolve solutions over generations.

*Reference:* Holland, J. H. (1992). Genetic Algorithms. Scientific American, 267(1), 66-72.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Algorithm
     - Description
   * - :class:`~thefittest.optimizers.GeneticAlgorithm`
     - Classical Genetic Algorithm with binary string representation
   * - :class:`~thefittest.optimizers.SelfCGA`
     - Self-configuring Genetic Algorithm with automatic parameter tuning (`Semenkin & Semenkina, 2012 <https://doi.org/10.1007/978-3-642-30976-2_50>`_)
   * - :class:`~thefittest.optimizers.PDPGA`
     - Population-level Dynamic Probabilities Genetic Algorithm with operator probability adaptation (`Niehaus & Banzhaf, 2001 <https://doi.org/10.1007/3-540-45355-5_26>`_)
   * - :class:`~thefittest.optimizers.SHAGA`
     - Success-History based Adaptive Genetic Algorithm (`Stanovov et al., 2019 <http://dx.doi.org/10.5220/0008071201800187>`_)

**Genetic Programming**

Genetic Programming evolves computer programs to solve problems. It uses tree-based representations and can perform symbolic regression, program synthesis, and other tasks requiring automatic program generation.

*Reference:* Koza, J. R. (1993). Genetic Programming - On the Programming of Computers by Means of Natural Selection. Complex Adaptive Systems.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Algorithm
     - Description
   * - :class:`~thefittest.optimizers.GeneticProgramming`
     - Genetic Programming for symbolic regression and program synthesis
   * - :class:`~thefittest.optimizers.SelfCGP`
     - Self-configuring Genetic Programming (`Semenkin & Semenkina, 2012 <http://dx.doi.org/10.1109/CEC.2012.6256587>`_)
   * - :class:`~thefittest.optimizers.PDPGP`
     - Population-level Dynamic Probabilities Genetic Programming (`Niehaus & Banzhaf, 2001 <https://doi.org/10.1007/3-540-45355-5_26>`_)

Classifiers
-----------

The library provides several classifier implementations based on evolutionary algorithms. These classifiers can learn complex decision boundaries, evolve neural network architectures, and optimize network weights using evolutionary strategies.

**Genetic Programming Classifiers**

Genetic Programming classifiers evolve symbolic expressions or tree structures to perform classification. They can discover interpretable decision rules and handle non-linear separations.

*Reference:* Koza, J. R. (1993). Genetic Programming - On the Programming of Computers by Means of Natural Selection. Complex Adaptive Systems.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Classifier
     - Description
   * - :class:`~thefittest.classifiers.GeneticProgrammingClassifier`
     - GP-based classifier evolving symbolic expressions for decision boundaries

**Neural Network Classifiers**

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

Regressors
----------

The library provides several regressor implementations based on evolutionary algorithms. These regressors can perform symbolic regression, optimize neural network weights, and evolve network architectures for continuous value prediction.

**Genetic Programming Regressors**

Genetic Programming regressors evolve symbolic expressions or tree structures to perform regression. They can discover interpretable mathematical models and handle complex non-linear relationships.

*Reference:* Koza, J. R. (1993). Genetic Programming - On the Programming of Computers by Means of Natural Selection. Complex Adaptive Systems.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Regressor
     - Description
   * - :class:`~thefittest.regressors.GeneticProgrammingRegressor`
     - GP-based regressor evolving symbolic expressions for explicit functional relationships

**Neural Network Regressors**

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

Contents:
---------

.. toctree::
   :maxdepth: 2

   modules/index
   references
   