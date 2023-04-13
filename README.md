# thefittest

## Installation
```bash
pip install thefittest
```
##  Dependencies
thefittest requires:
* python (>=3.7,<3.11);
* numpy (>=1.21.6,<=1.23);
* numba (>=0.56.4).


## The package contains methods
* **Genetic algorithm** (Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-72):
    * **Self-configuring genetic algorithm** (Semenkin, E.S., Semenkina, M.E. Self-configuring Genetic Algorithm with Modified Uniform Crossover Operator. LNCS, 7331, 2012, pp. 414-421);
    * **SHAGA** (Stanovov, Vladimir & Akhmedova, Shakhnaz & Semenkin, Eugene. (2019). Genetic Algorithm with Success History based Parameter Adaptation. 180-187. 10.5220/0008071201800187).
* **Differential evolution** (Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient Adaptive Scheme for Global Optimization Over Continuous Spaces. Journal of Global Optimization. 23):
    * **SaDE** (Qin, Kai & Suganthan, Ponnuthurai. (2005). Self-adaptive differential evolution algorithm for numerical optimization. 2005 IEEE Congress on Evolutionary Computation, IEEE CEC 2005. Proceedings. 2. 1785-1791. 10.1109/CEC.2005.1554904);
    * **jDE** (Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan & Zumer, Viljem. (2007). Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical Benchmark Problems. Evolutionary Computation, IEEE Transactions on. 10. 646 - 657. 10.1109/TEVC.2006.872133);
    * **JADE** (Zhang, Jingqiao & Sanderson, A.C.. (2009). JADE: Adaptive Differential Evolution With Optional External Archive. Evolutionary Computation, IEEE Transactions on. 13. 945 - 958. 10.1109/TEVC.2009.2014613);
    * **SHADE** (Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based parameter adaptation for Differential Evolution. 2013 IEEE Congress on Evolutionary Computation, CEC 2013. 71-78. 10.1109/CEC.2013.6557555).
* **Genetic programming** (Koza, John R.. “Genetic programming - on the programming of computers by means
    of natural selection.” Complex Adaptive Systems (1993)):
    * **Self-configuring genetic programming** (Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm with modified uniform crossover. 1-6. 10.1109/CEC.2012.6256587).
## Benchmarks
* **CEC2005** (Suganthan, Ponnuthurai & Hansen, Nikolaus & Liang, Jing & Deb, Kalyan & Chen, Ying-ping & Auger, Anne & Tiwari, Santosh. (2005). Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization. Natural Computing. 341-357);
* **Symbolicregression17. 17 test regression problem from the paper** (Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm with modified uniform crossover. 1-6. 10.1109/CEC.2012.6256587).

You can also look at [**notebooks**](https://github.com/sherstpasha/thefittest-notebooks) with examples of how to use thefittest.
