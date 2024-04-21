.. image:: https://img.shields.io/pypi/v/thefittest?label=PyPI%20-%20Package%20version
    :target: https://pypi.org/project/thefittest/
    :alt: PyPI - Package version

.. image:: https://static.pepy.tech/badge/thefittest
    :target: https://pepy.tech/project/thefittest
    :alt: Downloads

.. image:: https://komarev.com/ghpvc/?username=thefittest
    :alt: Profile views

.. image:: https://codecov.io/github/sherstpasha/thefittest/coverage.svg?branch=master
    :alt: codecov.io

.. image:: https://app.codacy.com/project/badge/Grade/4c47b6de61c4422180529bbc360262c4
    :target: https://app.codacy.com/gh/sherstpasha/thefittest/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: Codacy Badge

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

.. image:: https://readthedocs.com/projects/sherstpasha-pavel/badge/?version=latest&token=71adf5d63b55f0def96b09e1ce4c60f8d57cbdaed7db777117f34e4718d5a1ea
    :target: https://sherstpasha-pavel.readthedocs-hosted.com/ru/latest/?badge=latest
    :alt: Documentation Status

|
.. image:: docs/logos/logo1.png
   :align: center

|

``thefittest`` is an open-source library designed for the efficient application of classical evolutionary algorithms and their effective modifications in optimization and machine learning. Our project aims to provide performance, accessibility, and ease of use, opening up the world of advanced evolutionary methods to you.

Features of ``thefittest``
--------------------------

**Performance**
  Our library is developed using advanced coding practices and delivers high performance through integration with `NumPy <https://numpy.org/>`_, `Scipy <https://scipy.org/>`_, `Numba <https://numba.pydata.org/>`_, and `scikit-learn <https://scikit-learn.org/>`_.

**Versatility**
  ``thefittest`` offers a wide range of classical evolutionary algorithms and effective modifications, making it the ideal choice for a variety of optimization and machine learning tasks.

**Integration with scikit-learn**
  Easily integrate machine learning methods from ``thefittest`` with `scikit-learn <https://scikit-learn.org/>`_ tools, creating comprehensive and versatile solutions for evolutionary optimization and model training tasks.


Installation
------------

.. code-block:: bash

    pip install thefittest

Dependencies
------------

``thefittest`` requires:

- `Python (>=3.7,<3.12) <https://www.python.org/>`_,
- `NumPy <https://numpy.org/>`_, 
- `Numba <https://numba.pydata.org/>`_;
- `Scipy <https://scipy.org/>`_.

``thefittest`` contains methods
-------------------------------

- **Genetic algorithm** (Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-72):

  - **Self-configuring genetic algorithm** (`Semenkin, E.S., Semenkina, M.E. Self-configuring Genetic Algorithm with Modified Uniform Crossover Operator. LNCS, 7331, 2012, pp. 414-421. <https://doi.org/10.1007/978-3-642-30976-2_50>`_);
  - **SHAGA** (`Stanovov, Vladimir & Akhmedova, Shakhnaz & Semenkin, Eugene. (2019). Genetic Algorithm with Success History based Parameter Adaptation. 180-187. <http://dx.doi.org/10.5220/0008071201800187>`_);
  - **PDPGA** (`Niehaus, J., Banzhaf, W. (2001); Adaption of Operator Probabilities in Genetic Programming. In: Miller, J., Tomassini, M., Lanzi, P.L., Ryan, C., Tettamanzi, A.G.B., Langdon, W.B. (eds) Genetic Programming. EuroGP 2001. Lecture Notes in Computer Science, vol 2038. Springer, Berlin, Heidelberg. <https://doi.org/10.1007/3-540-45355-5_26>`_).

- **Differential evolution** (Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient Adaptive Scheme for Global Optimization Over Continuous Spaces. Journal of Global Optimization. 23)

  - **jDE** (`Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan & Zumer, Viljem. (2007). Self-Adapting Control Parameters in Differential Evolution: A Comparative 13. 945 - 958. <http://dx.doi.org/10.1109/TEVC.2009.2014613>`_);
  - **SHADE** (`Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based parameter adaptation for Differential Evolution. 2013 IEEE Congress on Evolutionary Computation, CEC 2013. 71-78. <https://doi.org/10.1109/CEC.2013.6557555>`_).

- **Genetic programming** (Koza, John R.. “Genetic programming - on the programming of computers by means of natural selection.” Complex Adaptive Systems (1993)):

  - **Self-configuring genetic programming** (`Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm with modified uniform crossover. 1-6. <http://dx.doi.org/10.1109/CEC.2012.6256587>`_).
  - **PDPGP** (`Niehaus, J., Banzhaf, W. (2001); Adaption of Operator Probabilities in Genetic Programming. In: Miller, J., Tomassini, M., Lanzi, P.L., Ryan, C., Tettamanzi, A.G.B., Langdon, W.B. (eds) Genetic Programming. EuroGP 2001. Lecture Notes in Computer Science, vol 2038. Springer, Berlin, Heidelberg. <https://doi.org/10.1007/3-540-45355-5_26>`_).

- **Genetic programming of neural networks (GPNN)** (`Lipinsky L., Semenkin E., Bulletin of the Siberian State Aerospace University., 3(10), 22-26 (2006). In Russian`_);
- **Multilayer perceptron trained by evolutionary algorithms** (`Cotta, Carlos & Alba, Enrique & Sagarna, R. & Larranaga, Pedro. (2002). Adjusting Weights in Artificial Neural Networks using Evolutionary Algorithms. <http://dx.doi.org/10.1007/978-1-4615-1539-5_18>`_);

Benchmarks
----------

- **CEC2005** (`Suganthan, Ponnuthurai & Hansen, Nikolaus & Liang, Jing & Deb, Kalyan & Chen, Ying-ping & Auger, Anne & Tiwari, Santosh. (2005). Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization. Natural Computing. 341-357`_);
- **Symbolicregression17. 17 test regression problem from the paper** (`Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm with modified uniform crossover. 1-6. <http://dx.doi.org/10.1109/CEC.2012.6256587>`_).
- **Iris** (`Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. <https://doi.org/10.24432/C56C76>`_);
- **Wine** (`Aeberhard,Stefan and Forina,M.. (1991). Wine. UCI Machine Learning Repository. <https://doi.org/10.24432/C5PC7J>`_);
- **Breast Cancer Wisconsin (Diagnostic)** (`Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. <https://doi.org/10.24432/C5DW2B>`_);
- **Optical Recognition of Handwritten Digits** (`Alpaydin,E. and Kaynak,C.. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. <https://doi.org/10.24432/C50P49>`_);

Examples
--------

Notebooks on how to use ``thefittest``:

- `Solving Binary and Real-Valued Optimization Problems with Genetic Algorithms;; <https://github.com/sherstpasha/thefittest-notebooks/blob/main/genetic_algorithm_binary_rastrigin_custom_problems.ipynb>`_
- `Solving Real-Valued Optimization Problems with Differential Evolution; <https://github.com/sherstpasha/thefittest-notebooks/blob/main/differential_evolution_griewank_custom_problems.ipynb>`_
- `Solving Symbolic Regression Problems Using Genetic Programming Algorithms; <https://github.com/sherstpasha/thefittest-notebooks/blob/main/genetic_programming_symbolic_regression_problem.ipynb>`_
- `Training Neural Networks Using Evolutionary Algorithms for Regression and Classification Problems; <https://github.com/sherstpasha/thefittest-notebooks/blob/main/mlpea_regression_classification_problem.ipynb>`_
- `Optimizing Neural Network Structure Using Genetic Programming; <https://github.com/sherstpasha/thefittest-notebooks/blob/main/gpnn_regression_classification_problems.ipynb>`_

If some notebooks are too big to display, you can use `NBviewer <https://nbviewer.org/>`_.