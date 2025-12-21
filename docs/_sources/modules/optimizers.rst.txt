optimizers
==========

The library provides a comprehensive set of evolutionary optimization algorithms for solving continuous and discrete optimization problems. Each optimizer is designed with flexibility in mind, supporting various selection methods, crossover operators, and mutation strategies.

Contents
--------

.. contents::
   :local:
   :depth: 2

Differential Evolution Optimizers
----------------------------------

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

DifferentialEvolution
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.optimizers.DifferentialEvolution
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

jDE
~~~

.. autoclass:: thefittest.optimizers.jDE
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

SHADE
~~~~~

.. autoclass:: thefittest.optimizers.SHADE
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Genetic Algorithms
------------------

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

GeneticAlgorithm
~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.optimizers.GeneticAlgorithm
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

SHAGA
~~~~~

.. autoclass:: thefittest.optimizers.SHAGA
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

SelfCGA
~~~~~~~

.. autoclass:: thefittest.optimizers.SelfCGA
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

PDPGA
~~~~~

.. autoclass:: thefittest.optimizers.PDPGA
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Genetic Programming
-------------------

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

GeneticProgramming
~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.optimizers.GeneticProgramming
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

PDPGP
~~~~~

.. autoclass:: thefittest.optimizers.PDPGP
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

SelfCGP
~~~~~~~

.. autoclass:: thefittest.optimizers.SelfCGP
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__