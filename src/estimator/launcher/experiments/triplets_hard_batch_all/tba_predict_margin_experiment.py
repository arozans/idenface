from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.triplet_batch_all_model import FmnistTripletBatchAllModel
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
            consts.EMBEDDING_SIZE: 64,
            consts.BATCH_SIZE: 64,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
        }


class PredictMarginAwareMnistSiameseModel(FmnistTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
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
    PredictMarginAwareMnistSiameseModel(1.0),
    PredictMarginAwareMnistSiameseModel(3.0),
    PredictMarginAwareMnistSiameseModel(5.0),
    PredictMarginAwareMnistSiameseModel(7.0),
    PredictMarginAwareMnistSiameseModel(9.0),
    PredictMarginAwareMnistSiameseModel(10.0),
    PredictMarginAwareMnistSiameseModel(12.0),
    PredictMarginAwareMnistSiameseModel(14.0),
    PredictMarginAwareMnistSiameseModel(16.0),
])
