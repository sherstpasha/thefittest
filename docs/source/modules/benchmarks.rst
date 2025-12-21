Benchmarks
==========

The benchmarks module provides a comprehensive collection of test problems for evaluating optimization algorithms and machine learning models. It includes classification datasets, optimization test functions, and benchmark suites.

Contents
--------

.. contents::
   :local:
   :depth: 2

ML Datasets
-----------

Machine learning datasets from the UCI Machine Learning Repository and other sources. These datasets are commonly used for testing classification algorithms.

*Reference:* Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Dataset
     - Description
   * - :class:`~thefittest.benchmarks.IrisDataset`
     - Famous dataset with iris flower measurements (150 samples, 4 features, 3 classes)
   * - :class:`~thefittest.benchmarks.WineDataset`
     - Wine recognition dataset from chemical analysis (178 samples, 13 features, 3 classes)
   * - :class:`~thefittest.benchmarks.BreastCancerDataset`
     - Breast cancer diagnostic dataset (569 samples, 30 features, 2 classes)
   * - :class:`~thefittest.benchmarks.DigitsDataset`
     - Handwritten digits recognition (5620 samples, 64 features, 10 classes)
   * - :class:`~thefittest.benchmarks.CreditRiskDataset`
     - Credit risk prediction dataset (3 features, 2 classes)
   * - :class:`~thefittest.benchmarks.UserKnowladgeDataset`
     - Student knowledge modeling (403 samples, 5 features, 4 classes)
   * - :class:`~thefittest.benchmarks.BanknoteDataset`
     - Banknote authentication dataset (1372 samples, 4 features, 2 classes)

IrisDataset
~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.IrisDataset
   :members:
   :undoc-members:
   :show-inheritance:

WineDataset
~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.WineDataset
   :members:
   :undoc-members:
   :show-inheritance:

BreastCancerDataset
~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.BreastCancerDataset
   :members:
   :undoc-members:
   :show-inheritance:

DigitsDataset
~~~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.DigitsDataset
   :members:
   :undoc-members:
   :show-inheritance:

CreditRiskDataset
~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.CreditRiskDataset
   :members:
   :undoc-members:
   :show-inheritance:

UserKnowladgeDataset
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.UserKnowladgeDataset
   :members:
   :undoc-members:
   :show-inheritance:

BanknoteDataset
~~~~~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.BanknoteDataset
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Functions
----------------------

Classic benchmark functions for testing continuous optimization algorithms. Each function has different characteristics (unimodal/multimodal, separable/non-separable) that challenge different aspects of optimization algorithms.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Function
     - Description
   * - :class:`~thefittest.benchmarks.Sphere`
     - Simple quadratic function, unimodal and convex
   * - :class:`~thefittest.benchmarks.Rosenbrock`
     - Valley-shaped function, unimodal with narrow parabolic valley
   * - :class:`~thefittest.benchmarks.Rastrigin`
     - Highly multimodal with many local minima
   * - :class:`~thefittest.benchmarks.Ackley`
     - Multimodal with nearly flat outer region and large hole at center
   * - :class:`~thefittest.benchmarks.Griewank`
     - Multimodal with many widespread local minima
   * - :class:`~thefittest.benchmarks.Weierstrass`
     - Continuous nowhere differentiable function with fractal structure
   * - :class:`~thefittest.benchmarks.Schwefe1_2`
     - Unimodal with non-separable variables
   * - :class:`~thefittest.benchmarks.HighConditionedElliptic`
     - Unimodal with high condition number
   * - :class:`~thefittest.benchmarks.OneMax`
     - Simple sum of all variables

Sphere
~~~~~~

.. autoclass:: thefittest.benchmarks.Sphere
   :members:
   :undoc-members:
   :show-inheritance:

Rosenbrock
~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.Rosenbrock
   :members:
   :undoc-members:
   :show-inheritance:

Rastrigin
~~~~~~~~~

.. autoclass:: thefittest.benchmarks.Rastrigin
   :members:
   :undoc-members:
   :show-inheritance:

Ackley
~~~~~~

.. autoclass:: thefittest.benchmarks.Ackley
   :members:
   :undoc-members:
   :show-inheritance:

Griewank
~~~~~~~~

.. autoclass:: thefittest.benchmarks.Griewank
   :members:
   :undoc-members:
   :show-inheritance:

Weierstrass
~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.Weierstrass
   :members:
   :undoc-members:
   :show-inheritance:

Schwefe1_2
~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.Schwefe1_2
   :members:
   :undoc-members:
   :show-inheritance:

HighConditionedElliptic
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: thefittest.benchmarks.HighConditionedElliptic
   :members:
   :undoc-members:
   :show-inheritance:

OneMax
~~~~~~

.. autoclass:: thefittest.benchmarks.OneMax
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark Suites
----------------

Comprehensive benchmark suites with multiple test functions for systematic algorithm evaluation.

CEC2005
~~~~~~~

The CEC 2005 Special Session on Real-Parameter Optimization provides 25 test functions organized into categories: unimodal (F1-F5), basic multimodal (F6-F12), expanded (F13-F14), and hybrid composition functions (F15-F25).

*Reference:* Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005). Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization.

**Module:** :mod:`thefittest.benchmarks.CEC2005`

.. automodule:: thefittest.benchmarks.CEC2005
   :members: problems_dict
   :undoc-members:

**Usage Example:**

.. code-block:: python

    from thefittest.benchmarks import CEC2005
    
    # Access problem dictionary
    problems = CEC2005.problems_dict
    
    # Get F1 (Shifted Sphere)
    f1_config = problems["F1"]
    function = f1_config["function"]()
    bounds = f1_config["bounds"]
    optimum = f1_config["optimum"]

Symbolic Regression
~~~~~~~~~~~~~~~~~~~

A collection of 17 test functions for symbolic regression and genetic programming benchmarks. Functions range from 1D to 2D with varying complexity.

**Module:** :mod:`thefittest.benchmarks.symbolicregression17`

.. automodule:: thefittest.benchmarks.symbolicregression17
   :members: F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, problems_dict
   :undoc-members:

**Usage Example:**

.. code-block:: python

    from thefittest.benchmarks import symbolicregression17
    import numpy as np
    
    # Get F5 (Rosenbrock)
    f5_config = symbolicregression17.problems_dict["F5"]
    function = f5_config["function"]
    
    # Generate data
    X = np.random.uniform(*f5_config["bounds"], size=(100, 2))
    y = function(X)