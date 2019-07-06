from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.utils import consts


class MnistSoftmaxOptimizerExperiment(ExperimentLauncher):
    @property
    def name(self):
        return "mnist_softmax_optimizer_exp"


class OptimizerAwareMnistSoftmaxModel(MnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return self.name + "optimizer_" + str(self.optimizer) + "_lr_" + str(self.lr) + "_batch_size_" + str(
            self.batch_size)

    def __init__(self, predict_margin, lr, batch_size) -> None:
        super().__init__()
        self.optimizer = predict_margin
        self.lr = lr
        self.batch_size = batch_size

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {
            consts.OPTIMIZER: self.optimizer,
            consts.LEARNING_RATE: self.lr,
            consts.BATCH_SIZE: self.batch_size,
        })


launcher = MnistSoftmaxOptimizerExperiment([
    OptimizerAwareMnistSoftmaxModel(consts.GRADIENT_DESCEND_OPTIMIZER, 0.05, 300),
    OptimizerAwareMnistSoftmaxModel(consts.GRADIENT_DESCEND_OPTIMIZER, 0.1, 300),
    OptimizerAwareMnistSoftmaxModel(consts.GRADIENT_DESCEND_OPTIMIZER, 0.15, 300),
    OptimizerAwareMnistSoftmaxModel(consts.GRADIENT_DESCEND_OPTIMIZER, 0.25, 300),
    OptimizerAwareMnistSoftmaxModel(consts.MOMENTUM_OPTIMIZER, 0.001, 300),
    OptimizerAwareMnistSoftmaxModel(consts.MOMENTUM_OPTIMIZER, 0.02, 300),
    OptimizerAwareMnistSoftmaxModel(consts.MOMENTUM_OPTIMIZER, 0.05, 300),
    OptimizerAwareMnistSoftmaxModel(consts.MOMENTUM_OPTIMIZER, 0.01, 300),
    OptimizerAwareMnistSoftmaxModel(consts.MOMENTUM_OPTIMIZER, 0.02, 300),
    OptimizerAwareMnistSoftmaxModel(consts.ADAM_OPTIMIZER, 0.0001, 300),
    OptimizerAwareMnistSoftmaxModel(consts.ADAM_OPTIMIZER, 0.0005, 300),
    OptimizerAwareMnistSoftmaxModel(consts.ADAM_OPTIMIZER, 0.001, 300),
    OptimizerAwareMnistSoftmaxModel(consts.ADAM_OPTIMIZER, 0.002, 300),
    OptimizerAwareMnistSoftmaxModel(consts.ADAM_OPTIMIZER, 0.005, 300),
    OptimizerAwareMnistSoftmaxModel(consts.ADAM_OPTIMIZER, 0.01, 300),
])
