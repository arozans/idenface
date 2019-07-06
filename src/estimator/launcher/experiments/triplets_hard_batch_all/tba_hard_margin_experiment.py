from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.triplet_batch_all_model import FmnistTripletBatchAllModel
from src.utils import consts


class FmnistTBAHardMarginExperimentExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "fmnist_tba_hard_margin_exp"

    @property
    def params(self):
        return {
            consts.TRAIN_STEPS: 5 * 1000,
            consts.EXCLUDED_KEYS: [1, 2, 3],
            consts.PREDICT_SIMILARITY_MARGIN: 3.0,
            consts.NUM_CHANNELS: 64,
            consts.DENSE_UNITS: [64],
            consts.BATCH_SIZE: 32,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
            consts.GLOBAL_SUFFIX: "v2",
        }


class HardMarginAwareFmnistTBAModel(FmnistTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
            {
                "hard_triplet_margin": self.htm,
            })

    def __init__(self, nc) -> None:
        super().__init__()
        self.htm = nc

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.HARD_TRIPLET_MARGIN: self.htm
                               })


launcher = FmnistTBAHardMarginExperimentExperimentLauncher([
    HardMarginAwareFmnistTBAModel(0.1),
    HardMarginAwareFmnistTBAModel(0.2),
    HardMarginAwareFmnistTBAModel(0.35),
    HardMarginAwareFmnistTBAModel(0.5),
    HardMarginAwareFmnistTBAModel(0.6),
    HardMarginAwareFmnistTBAModel(0.7),
    HardMarginAwareFmnistTBAModel(0.8),
    HardMarginAwareFmnistTBAModel(1.0),
    HardMarginAwareFmnistTBAModel(2.0),
    HardMarginAwareFmnistTBAModel(3.0),
])
