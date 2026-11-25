.. Thefittest documentation master file

Thefittest Documentation
========================

.. raw:: html

   <div class="header-section">
      <div class="badges">
         <a href="https://pypi.org/project/thefittest/"><img src="https://img.shields.io/pypi/v/thefittest?label=PyPI" alt="PyPI version"></a>
         <a href="https://pepy.tech/project/thefittest"><img src="https://static.pepy.tech/badge/thefittest" alt="Downloads"></a>
         <img src="https://komarev.com/ghpvc/?username=thefittest" alt="Profile views">
         <a href="https://app.codacy.com/gh/sherstpasha/thefittest/dashboard"><img src="https://app.codacy.com/project/badge/Grade/4c47b6de61c4422180529bbc360262c4" alt="Codacy Badge"></a>
         <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
      </div>
   </div>

.. image:: ../logos/logo1.png
   :align: center
   :class: logo-image

|

``thefittest`` is an open-source library designed for the efficient application of classical evolutionary algorithms and their effective modifications in optimization and machine learning. Our project aims to provide performance, accessibility, and ease of use, opening up the world of advanced evolutionary methods to you.

.. raw:: html

   <h2 class="centered-header">Modules</h2>
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
            <a href="modules/optimizers.html#thefittest.optimizers.SHAGA">SHAGA</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.SelfCSHAGA">SelfCSHAGA</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.PDPSHAGA">PDPSHAGA</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.NSGAII">NSGAII</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.SelfCNSGAII">SelfCNSGAII</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.PDPNSGAII">PDPNSGAII</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.NSGPII">NSGPII</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.SelfCNSGPII">SelfCNSGPII</a>,
            <a href="modules/optimizers.html#thefittest.optimizers.PDPNSGPII">PDPNSGPII</a>
         </p>
         <a href="modules/optimizers.html" class="module-card-arrow">View documentation →</a>
      </div>
      
      <div class="module-card">
         <h3>Classifiers</h3>
         <p class="module-items">
            <a href="modules/classifiers.html#thefittest.classifiers.GeneticProgrammingClassifier">GeneticProgrammingClassifier</a>,
            <a href="modules/classifiers.html#thefittest.classifiers.MLPEAClassifier">MLPEAClassifier</a>, 
            <a href="modules/classifiers.html#thefittest.classifiers.GeneticProgrammingNeuralNetClassifier">GeneticProgrammingNeuralNetClassifier</a>
         </p>
         <a href="modules/classifiers.html" class="module-card-arrow">View documentation →</a>
      </div>
      
      <div class="module-card">
         <h3>Regressors</h3>
         <p class="module-items">
            <a href="modules/regressors.html#thefittest.regressors.GeneticProgrammingRegressor">GeneticProgrammingRegressor</a>,
            <a href="modules/regressors.html#thefittest.regressors.MLPEARegressor">MLPEARegressor</a>, 
            <a href="modules/regressors.html#thefittest.regressors.GeneticProgrammingNeuralNetRegressor">GeneticProgrammingNeuralNetRegressor</a>
         </p>
         <a href="modules/regressors.html" class="module-card-arrow">View documentation →</a>
      </div>
      
      <div class="module-card">
         <h3>Benchmarks</h3>
         <p class="module-items">
            <a href="modules/benchmarks.html#ml-datasets">ML Datasets</a>, 
            <a href="modules/benchmarks.html#optimization-functions">Optimization Functions</a>, 
            <a href="modules/benchmarks.html#benchmark-suites">Benchmark Suites</a>,
            <a href="modules/benchmarks.html#multiobjective-problems">Multiobjective Problems</a>
         </p>
         <a href="modules/benchmarks.html" class="module-card-arrow">View documentation →</a>
      </div>

      <div class="module-card">
         <h3>Utils</h3>
         <p class="module-items">
            <a href="modules/utils/selections.html">selections</a>, 
            <a href="modules/utils/crossovers.html">crossovers</a>, 
            <a href="modules/utils/mutations.html">mutations</a>, 
            <a href="modules/utils/transformations.html">transformations</a>, 
            <a href="modules/utils/random.html">random</a>,
            <a href="modules/utils/multiobjective.html">multiobjective</a>
         </p>
         <a href="modules/utils/index.html" class="module-card-arrow">View documentation →</a>
      </div>
   </div>

.. raw:: html

   <h2 class="centered-header">Installation</h2>

**Basic installation** (for evolutionary algorithms and symbolic regression):

.. code-block:: bash

    pip install thefittest

**Neural networks on CPU or GPU**:

.. code-block:: bash

    python -m pip install "thefittest[torch]"

PyTorch is optional for optimization and symbolic regression. Neural estimators
use ``device="cpu"`` by default; CUDA is not required. For a CUDA installation,
follow the `official PyTorch instructions <https://pytorch.org/get-started/locally/>`_
and select ``device="cuda"`` on a compatible machine.

For tree/network plots, also install Matplotlib and NetworkX:

.. code-block:: bash

    python -m pip install matplotlib networkx

See :doc:`modules/optimizers` for the available algorithms and
:doc:`release_notes` for changes in 0.3.0.

.. raw:: html

   <h2 class="centered-header">Quick Start</h2>

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
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import f1_score

    # Load and prepare data
    data = IrisDataset()
    X = data.get_X()
    y = data.get_y()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = MinMaxScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # Train model
    model = MLPEAClassifier(
        n_iter=5,
        pop_size=20,
        hidden_layers=[5, 5],
        weights_optimizer=SHAGA,
        weights_optimizer_args={"show_progress_each": 1},
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    predict = model.predict(X_test)
    print("F1 score:", f1_score(y_test, predict, average="macro"))

.. raw:: html

   <h2 class="centered-header">Dependencies</h2>

.. raw:: html

   <div class="dependencies-grid">
      <div class="dependency-card">
         <h3>Required</h3>
         <ul>
            <li>
               <strong>Python</strong> 3.9–3.13
               <br/>
               <span class="dependency-note">
                  Core runtime environment supported across a wide range of Python versions
               </span>
            </li>
            <li>
               <strong>NumPy</strong> ≥ 1.26
               <br/>
               <span class="dependency-note">
                  Vectorized numerical computations and population-level data representation
               </span>
            </li>
            <li>
               <strong>Numba</strong> ≥ 0.60
               <br/>
               <span class="dependency-note">
                  JIT compilation of performance-critical evolutionary operators and neural network routines
               </span>
            </li>
            <li>
               <strong>SciPy</strong>
               <br/>
               <span class="dependency-note">
                  Linear algebra and numerical utilities used in optimization procedures
               </span>
            </li>
            <li>
               <strong>scikit-learn</strong> ≥ 1.6
               <br/>
               <span class="dependency-note">
                  Dataset utilities, preprocessing, metrics, and sklearn-compatible API design
               </span>
            </li>
            <li>
               <strong>joblib</strong> ≥ 1.3.0
               <br/>
               <span class="dependency-note">
                  Lightweight parallelization and efficient execution of independent tasks
               </span>
            </li>
         </ul>
      </div>

      <div class="dependency-card">
         <h3>Optional</h3>
         <ul>
            <li>
               <strong>networkx</strong>
               <br/>
               <span class="dependency-note">
                  Visualization of genetic programming trees and neural network structures
               </span>
            </li>
            <li>
               <strong>PyTorch</strong> ≥ 2.0 (CPU or CUDA)
               <br/>
               <span class="dependency-note">
                  Evolutionary training of neural networks with optional GPU acceleration
               </span>
            </li>
         </ul>
      </div>
   </div>

.. toctree::
   :maxdepth: 2
   :hidden:

   modules/index
   learning_materials
   release_notes
   references
