from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
import numpy as np
# from ..base._net import MultilayerPerceptron
from ..tools import donothing
# from ..tools.operators import LogisticSigmoid
# from ..tools.operators import SoftMax
# from ..tools.operators import ReLU
from ..tools.metrics import categorical_crossentropy
from ..optimizers import SHADE
from ..optimizers import DifferentialEvolution
from ..optimizers import JADE
from ..optimizers import jDE
from ..optimizers import SaDE2005
from ..optimizers import OptimizerStringType


# class MLPClassifierEA(MultilayerPerceptron):
#     def __init__(
#             self,
#             iters: int,
#             pop_size: int,
#             hidden_layers: Tuple,
#             activation: Union[LogisticSigmoid, ReLU] = LogisticSigmoid,
#             activation_output: SoftMax = SoftMax,
#             no_increase_num: Optional[int] = None,
#             show_progress_each: Optional[int] = None,
#             keep_history: bool = False,
#             optimizer: OptimizerStringType = SHADE):

#         MultilayerPerceptron.__init__(self,
#                                       hidden_layers=hidden_layers,
#                                       activation=activation,
#                                       activation_output=activation_output)
#         self._optimizer = optimizer(fitness_function=donothing,
#                                     genotype_to_phenotype=donothing,
#                                     iters=iters,
#                                     pop_size=pop_size,
#                                     left=-2,
#                                     right=2,
#                                     optimal_value=0.,
#                                     minimization=True,
#                                     no_increase_num=no_increase_num,
#                                     show_progress_each=show_progress_each,
#                                     keep_history=keep_history)
#         self._weight_shapes: Tuple
#         self._sizes: Tuple

#     def _fitness_function(self,
#                           population: np.ndarray,
#                           X,
#                           targets: np.ndarray) -> np.ndarray:
#         fitness = [self._evaluate_net(net, X, targets) for net in population]
#         return np.array(fitness, dtype=np.float64)

#     def _evaluate_net(self,
#                       weights: np.ndarray,
#                       X: np.ndarray,
#                       targets: np.ndarray) -> float:
#         shaped_weight = self._weight_from_flat(weights)
#         output = self.forward(X, weights=shaped_weight)
#         error = categorical_crossentropy(targets, output)
#         return error

#     def _weight_from_flat(self,
#                           flat_array: np.ndarray) -> List:
#         cut_points = np.add.accumulate(self._sizes)[:-1]

#         flat_weights = np.split(flat_array, cut_points)
#         map_ = map(lambda arr, shape: np.reshape(arr, shape),
#                    flat_weights, self._weight_shapes)
#         shaped_weights = list(map_)
#         return shaped_weights

#     def _get_weights_shape(self,
#                            X: np.ndarray) -> Tuple:
#         shape = X.shape
#         return shape, np.multiply.reduce(shape)

#     def _fit(self,
#              X: np.ndarray,
#              y: np.ndarray):
#         n_inputs = X.shape[1]
#         n_outputs = len(set(y))
#         eye = np.eye(n_outputs)
#         target_probas = eye[y]

#         self._define_structure(n_inputs, n_outputs)
#         self._define_weights()

#         map_ = map(self._get_weights_shape, self._weights)
#         self._weight_shapes, self._sizes = list(zip(*list(map_)))
#         n_variables = sum(self._sizes)

#         self.optimizer._fitness_function = lambda population: self._fitness_function(
#             population, X, target_probas)
#         self.optimizer._left = np.full(
#             shape=n_variables, fill_value=-2, dtype=np.float64)
#         self.optimizer._right = np.full(
#             shape=n_variables, fill_value=2, dtype=np.float64)

#         self.optimizer.fit()
#         solution = self.optimizer.get_fittest()
#         genotype, phenotype, fitness = solution.get()
#         self._weights = self._weight_from_flat(phenotype)
#         return self

#     def _predict(self,
#                  X: np.ndarray) -> np.ndarray:
#         output = self.forward(X)
#         y_pred = np.argmax(output, axis=1)
#         return y_pred
