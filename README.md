# Thefittest 

[![PyPI](https://img.shields.io/pypi/v/thefittest?label=PyPI%20-%20Package%20version)](https://pypi.org/project/thefittest/)
[![Downloads](https://static.pepy.tech/badge/thefittest)](https://pepy.tech/project/thefittest)
![codecov.io](https://codecov.io/github/sherstpasha/thefittest/coverage.svg?branch=master)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4c47b6de61c4422180529bbc360262c4)](https://app.codacy.com/gh/sherstpasha/thefittest/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Installation
```bash
pip install thefittest
```

## Dependencies
thefittest requires:
*   python (>=3.7,<3.11);
*   numpy (>=1.21.6,<=1.23);
*   numba (>=0.56.4);
*   scipy (>0.16).

## The package contains methods
*   **Genetic algorithm** (Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-72):
    *   **Self-configuring genetic algorithm** (Semenkin, E.S., Semenkina, M.E. Self-configuring Genetic Algorithm with Modified Uniform Crossover Operator. LNCS, 7331, 2012, pp. 414-421. https://doi.org/10.1007/978-3-642-30976-2_50);
    *   **SHAGA** (Stanovov, Vladimir & Akhmedova, Shakhnaz & Semenkin, Eugene. (2019). Genetic Algorithm with Success History based Parameter Adaptation. 180-187. http://dx.doi.org/10.5220/0008071201800187).
*   **Differential evolution** (Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient Adaptive Scheme for Global Optimization Over Continuous Spaces. Journal of Global Optimization. 23):
    *   **SaDE** (Qin, Kai & Suganthan, Ponnuthurai. (2005). Self-adaptive differential evolution algorithm for numerical optimization. 2005 IEEE Congress on Evolutionary Computation, IEEE CEC 2005. Proceedings. 2. 1785-1791. http://dx.doi.org/10.1109/CEC.2005.1554904);
    *   **jDE** (Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan & Zumer, Viljem. (2007). Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical Benchmark Problems. Evolutionary Computation, IEEE Transactions on. 10. 646 - 657. http://dx.doi.org/10.1109/TEVC.2006.872133);
    *   **JADE** (Zhang, Jingqiao & Sanderson, A.C.. (2009). JADE: Adaptive Differential Evolution With Optional External Archive. Evolutionary Computation, IEEE Transactions on. 13. 945 - 958. http://dx.doi.org/10.1109/TEVC.2009.2014613);
    *   **SHADE** (Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based parameter adaptation for Differential Evolution. 2013 IEEE Congress on Evolutionary Computation, CEC 2013. 71-78. https://doi.org/10.1109/CEC.2013.6557555).
*   **Genetic programming** (Koza, John R.. “Genetic programming - on the programming of computers by means
    of natural selection.” Complex Adaptive Systems (1993)):
    *   **Self-configuring genetic programming** (Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm with modified uniform crossover. 1-6. http://dx.doi.org/10.1109/CEC.2012.6256587).
*   **Genetic programming of neural networks (GPNN)** (Lipinsky L., Semenkin E., Bulletin of the Siberian State Aerospace University., 3(10), 22-26 (2006). In Russian);
*   **Multilayer perceptron trained by evolutionary algorithms** (Cotta, Carlos & Alba, Enrique & Sagarna, R. & Larranaga, Pedro. (2002). Adjusting Weights in Artificial Neural Networks using Evolutionary Algorithms. http://dx.doi.org/10.1007/978-1-4615-1539-5_18);

## Benchmarks
*   **CEC2005** (Suganthan, Ponnuthurai & Hansen, Nikolaus & Liang, Jing & Deb, Kalyan & Chen, Ying-ping & Auger, Anne & Tiwari, Santosh. (2005). Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization. Natural Computing. 341-357);
*   **Symbolicregression17. 17 test regression problem from the paper** (Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm with modified uniform crossover. 1-6. http://dx.doi.org/10.1109/CEC.2012.6256587).
*   **Iris** (Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.);
*   **Wine** (Aeberhard,Stefan and Forina,M.. (1991). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.);
*   **Breast Cancer Wisconsin (Diagnostic)** (Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.);
*   **Optical Recognition of Handwritten Digits** (Alpaydin,E. and Kaynak,C.. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.);

## Examples
Notebooks on how to use thefittest:
*   [**Solving binary and real-valued optimization problems with a genetic algorithm;**](https://github.com/sherstpasha/thefittest-notebooks/blob/main/genetic_algorithm_binary_rastrigin_custom_problems.ipynb) 
*   [**Solving real-valued optimization problems with a differential evolution;**](https://github.com/sherstpasha/thefittest-notebooks/blob/main/differential_evolution_griewank_custom_problems.ipynb) 
*   [**Symbolic regression problems solving using genetic programming algorithm;**](https://github.com/sherstpasha/thefittest-notebooks/blob/main/genetic_programming_symbolic_regression_problem.ipynb) 
*   [**SelfCGA self-configuring visualisation;**](https://github.com/sherstpasha/thefittest-notebooks/blob/main/selfcga_animation_proba.ipynb) 
*   [**SelfCGP self-configuring visualisation;**](https://github.com/sherstpasha/thefittest-notebooks/blob/main/selfcgp_proba.ipynb) 
*   [**The process of adapting SaDE parameters;**](https://github.com/sherstpasha/thefittest-notebooks/blob/main/sade2005_rastrigin.ipynb) 


 **If some notebooks are <u>too big to display</u>, you can use [<u>NBviewer</u>](https://nbviewer.org/)**.
