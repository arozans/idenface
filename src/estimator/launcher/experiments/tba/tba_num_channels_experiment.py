from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.tba_model import FmnistTBAModel
from src.utils import consts


class FmnistTBANumChannelsExperimentExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "fmnist_tba_num_channels_exp"

    @property
    def params(self):
        return {
            consts.TRAIN_STEPS: 5 * 1000,
            consts.PREDICT_SIMILARITY_MARGIN: 3.0,

            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.DENSE_UNITS: [64],
            consts.BATCH_SIZE: 32,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
        }


class NumChannelsAwareFmnistTBAModel(FmnistTBAModel):

    @property
    def summary(self) -> str:
        return self._summary_from_dict(
            {
                "num_channles": self.nc,
            })

    def __init__(self, nc) -> None:
        super().__init__()
        self.nc = nc

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.NUM_CHANNELS: self.nc
                               })


launcher = FmnistTBANumChannelsExperimentExperimentLauncher([
    NumChannelsAwareFmnistTBAModel(16),
    NumChannelsAwareFmnistTBAModel(32),
    NumChannelsAwareFmnistTBAModel(64),
    NumChannelsAwareFmnistTBAModel(128),
    NumChannelsAwareFmnistTBAModel(256),
])
