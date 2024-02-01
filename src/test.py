from thefittest.classifiers._mlpeaclassifier import MLPClassifierEA2
from thefittest.optimizers import SHADE
from sklearn.utils.estimator_checks import check_estimator

model = MLPClassifierEA2(iters=500, pop_size=250,
                        hidden_layers=(5,),
                        activation="relu",
                        weights_optimizer=SHADE,
                        weights_optimizer_args={"show_progress_each": 50})

print(check_estimator(model))