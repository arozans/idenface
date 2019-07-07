from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.tba_model import FmnistTBAModel
from src.utils import consts


class FmnistTBAPredictMarginExperimentExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "fmnist_tba_predict_margin_exp"

    @property
    def params(self):
        return {
            consts.TRAIN_STEPS: 30 * 1000,

            consts.NUM_CHANNELS: 32,
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.PREDICT_SIMILARITY_MARGIN: 5.0,
            consts.DENSE_UNITS: [64],
            consts.BATCH_SIZE: 64,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
        }


class PredictMarginAwareMnistContrastiveModel(FmnistTBAModel):

    @property
    def summary(self) -> str:
        return self._summary_from_dict(
            {
                "pm": self.pm,
            })

    def __init__(self, pm) -> None:
        super().__init__()
        self.pm = pm

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.PREDICT_SIMILARITY_MARGIN: self.pm
                               })


launcher = FmnistTBAPredictMarginExperimentExperimentLauncher([
    PredictMarginAwareMnistContrastiveModel(1.0),
    PredictMarginAwareMnistContrastiveModel(3.0),
    PredictMarginAwareMnistContrastiveModel(5.0),
    PredictMarginAwareMnistContrastiveModel(7.0),
    PredictMarginAwareMnistContrastiveModel(9.0),
    PredictMarginAwareMnistContrastiveModel(10.0),
    PredictMarginAwareMnistContrastiveModel(12.0),
    PredictMarginAwareMnistContrastiveModel(14.0),
    PredictMarginAwareMnistContrastiveModel(16.0),
])
