from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import consts


class MnistSiameseOptimizerExperiment(ExperimentLauncher):
    @property
    def name(self):
        return "mnist_siamese_optimizer_exp"


class OptimizerAwareMnistSiameseModel(MnistSiameseModel):

    @property
    def summary(self) -> str:
        return self.name + "optimizer_" + str(self.optimizer) + "_lr_" + str(self.lr)

    def __init__(self, predict_margin, lr) -> None:
        super().__init__()
        self.optimizer = predict_margin
        self.lr = lr

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {
            consts.OPTIMIZER: self.optimizer,
            consts.LEARNING_RATE: self.lr
        })


launcher = MnistSiameseOptimizerExperiment([
    OptimizerAwareMnistSiameseModel(consts.GRADIENT_DESCEND_OPTIMIZER, 0.01),
    OptimizerAwareMnistSiameseModel(consts.MOMENTUM_OPTIMIZER, 0.01),
    OptimizerAwareMnistSiameseModel(consts.NESTEROV_OPTIMIZER, 0.01),
    OptimizerAwareMnistSiameseModel(consts.ADAM_OPTIMIZER, 0.001),  # best one
    OptimizerAwareMnistSiameseModel(consts.ADAM_OPTIMIZER, 0.01),
    OptimizerAwareMnistSiameseModel(consts.ADAM_OPTIMIZER, 0.1),
])
