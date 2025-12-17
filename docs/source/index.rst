.. Thefittest documentation master file

.. raw:: html

   <div class="header-section">
      <h1>thefittest</h1>
      <p class="tagline">Evolutionary algorithms for optimization & machine learning</p>
      <div class="badges">
         <a href="https://pypi.org/project/thefittest/"><img src="https://img.shields.io/pypi/v/thefittest?label=PyPI" alt="PyPI version"></a>
         <a href="https://pepy.tech/project/thefittest"><img src="https://static.pepy.tech/badge/thefittest" alt="Downloads"></a>
         <img src="https://komarev.com/ghpvc/?username=thefittest" alt="Profile views">
         <a href="https://codecov.io/github/sherstpasha/thefittest"><img src="https://codecov.io/github/sherstpasha/thefittest/coverage.svg?branch=master" alt="codecov.io"></a>
         <a href="https://app.codacy.com/gh/sherstpasha/thefittest/dashboard"><img src="https://app.codacy.com/project/badge/Grade/4c47b6de61c4422180529bbc360262c4" alt="Codacy Badge"></a>
         <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
      </div>
   </div>

.. image:: logos/logo1.png
   :align: center
   :class: logo-image

|

``thefittest`` is an open-source library designed for the efficient application of classical evolutionary algorithms and their effective modifications in optimization and machine learning. Our project aims to provide performance, accessibility, and ease of use, opening up the world of advanced evolutionary methods to you.

.. raw:: html

   <h2 class="section-header">Modules</h2>
   <div class="module-grid">
      <div class="module-card">
         <h3>Optimizers</h3>
         <p class="module-items">
            <a href="modules/optimizers.html#thefittest.optimizers.DifferentialEvolution">DifferentialEvolution</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.jDE">jDE</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.SHADE">SHADE</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.GeneticAlgorithm">GeneticAlgorithm</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.SelfCGA">SelfCGA</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.PDPGA">PDPGA</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.GeneticProgramming">GeneticProgramming</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.SelfCGP">SelfCGP</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.PDPGP">PDPGP</a>, 
            <a href="modules/optimizers.html#thefittest.optimizers.SHAGA">SHAGA</a>
         </p>
         <a href="modules/optimizers.html" class="module-card-arrow">View documentation →</a>
      </div>
      
      <div class="module-card">
         <h3>Classifiers</h3>
         <p class="module-items">
            <a href="modules/classifiers.html#thefittest.classifiers.GPClassifier">GPClassifier</a>, 
            <a href="modules/classifiers.html#thefittest.classifiers.MLPEAClassifier">MLPEAClassifier</a>, 
            <a href="modules/classifiers.html#thefittest.classifiers.GPNNClassifier">GPNNClassifier</a>
         </p>
         <a href="modules/classifiers.html" class="module-card-arrow">View documentation →</a>
      </div>
      
      <div class="module-card">
         <h3>Regressors</h3>
         <p class="module-items">
            <a href="modules/regressors.html#thefittest.regressors.GPRegressor">GPRegressor</a>, 
            <a href="modules/regressors.html#thefittest.regressors.MLPEARegressor">MLPEARegressor</a>, 
            <a href="modules/regressors.html#thefittest.regressors.GPNNRegressor">GPNNRegressor</a>
         </p>
         <a href="modules/regressors.html" class="module-card-arrow">View documentation →</a>
      </div>
      
      <div class="module-card">
         <h3>Benchmarks</h3>
         <p>CEC2005 functions, symbolic regression datasets, ML classification datasets</p>
         <a href="modules/benchmarks.html" class="module-card-arrow">View documentation →</a>
      </div>

      <div class="module-card">
         <h3>Utils</h3>
         <p class="module-items">
            <a href="modules/utils/selections.html">selections</a>, 
            <a href="modules/utils/crossovers.html">crossovers</a>, 
            <a href="modules/utils/mutations.html">mutations</a>, 
            <a href="modules/utils/transformations.html">transformations</a>, 
            <a href="modules/utils/random.html">random</a>
         </p>
         <a href="modules/utils/index.html" class="module-card-arrow">View documentation →</a>
      </div>
   </div>

Installation
============

**Basic installation** (for evolutionary algorithms and symbolic regression):

.. code-block:: bash

    pip install thefittest

**Full installation with neural networks** (requires GPU with CUDA):

First, install PyTorch with CUDA support: https://pytorch.org/get-started/locally/

.. code-block:: bash

    pip3 install torch --index-url https://download.pytorch.org/whl/cu124
    pip install thefittest

Quick Start
===========

**Optimization Example**

.. code-block:: python

    from thefittest.optimizers import SHADE

    # Define the objective function to minimize
    def custom_problem(x):
        return (5 - x[:, 0])**2 + (12 - x[:, 1])**2

    # Initialize the SHADE optimizer
    optimizer = SHADE(
        fitness_function=custom_problem,
        iters=25,
        pop_size=10,
        left_border=-100,
        right_border=100,
        num_variables=2,
        show_progress_each=10,
        minimization=True,
    )

    optimizer.fit()
    fittest = optimizer.get_fittest()
    print('Best solution:', fittest['phenotype'])
    print('Fitness:', fittest['fitness'])

**Machine Learning Example**

.. code-block:: python

    from thefittest.optimizers import SHAGA
    from thefittest.benchmarks import IrisDataset
    from thefittest.classifiers import MLPEAClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import minmax_scale
    from sklearn.metrics import f1_score

    # Load and prepare data
    data = IrisDataset()
    X = minmax_scale(data.get_X())
    y = data.get_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Train model
    model = MLPEAClassifier(
        n_iter=500,
        pop_size=500,
        hidden_layers=[5, 5],
        weights_optimizer=SHAGA,
        weights_optimizer_args={"show_progress_each": 10}
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    predict = model.predict(X_test)
    print("F1 score:", f1_score(y_test, predict, average="macro"))

Dependencies
============

Required packages (installed automatically):

- Python (>=3.7, <=3.13)
- numpy (>=1.26)
- numba (>=0.60)
- scipy
- scikit-learn (>=1.4)
- joblib (>=1.3.0)

Optional packages: **networkx** for visualization, **torch (>=2.0)** with CUDA for neural networks.

.. raw:: html

   <div class="examples-section">
      <h2>Learning Materials</h2>
      <div class="example-grid">
         <div class="example-card">
            <span class="card-category">Notebook</span>
            <h3>Solving Binary and Real-Valued Optimization Problems with Genetic Algorithms</h3>
            <a href="https://github.com/sherstpasha/thefittest-notebooks/blob/main/genetic_algorithm_binary_rastrigin_custom_problems.ipynb" target="_blank">Open Notebook →</a>
         </div>
         <div class="example-card">
            <span class="card-category">Notebook</span>
            <h3>Solving Real-Valued Optimization Problems with Differential Evolution</h3>
            <a href="https://github.com/sherstpasha/thefittest-notebooks/blob/main/differential_evolution_griewank_custom_problems.ipynb" target="_blank">Open Notebook →</a>
         </div>
         <div class="example-card">
            <span class="card-category">Notebook</span>
            <h3>Solving Symbolic Regression Problems Using Genetic Programming Algorithms</h3>
            <a href="https://github.com/sherstpasha/thefittest-notebooks/blob/main/genetic_programming_symbolic_regression_problem.ipynb" target="_blank">Open Notebook →</a>
         </div>
         <div class="example-card">
            <span class="card-category">Notebook</span>
            <h3>Training Neural Networks Using Evolutionary Algorithms for Regression and Classification Problems</h3>
            <a href="https://github.com/sherstpasha/thefittest-notebooks/blob/main/mlpea_regression_classification_problem.ipynb" target="_blank">Open Notebook →</a>
         </div>
         <div class="example-card">
            <span class="card-category">Notebook</span>
            <h3>Optimizing Neural Network Structure Using Genetic Programming</h3>
            <a href="https://github.com/sherstpasha/thefittest-notebooks/blob/main/gpnn_regression_classification_problems.ipynb" target="_blank">Open Notebook →</a>
         </div>
         <div class="example-card">
            <span class="card-category">Kaggle</span>
            <h3>Can Evolution Guide Us to Better Machine Learning?</h3>
            <a href="https://www.kaggle.com/code/pashasherst/can-evolution-guide-us-to-better-machine-learning" target="_blank">Open Kaggle →</a>
         </div>
         <div class="example-card">
            <span class="card-category">Article</span>
            <h3>Thefittest: evolutionary machine learning in Python</h3>
            <a href="https://doi.org/10.1051/itmconf/20245902020" target="_blank">Read Article →</a>
         </div>
         <div class="example-card">
            <span class="card-category">Article</span>
            <h3>Thefittest: зачем я пишу свою open-source библиотеку эволюционных алгоритмов</h3>
            <a href="https://habr.com/ru/articles/961924/" target="_blank">Read on Habr →</a>
         </div>
      </div>
   </div>

.. toctree::
   :maxdepth: 2
   :hidden:

   modules/index
   references
