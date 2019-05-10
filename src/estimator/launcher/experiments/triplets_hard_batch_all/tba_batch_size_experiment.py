from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.triplet_batch_all_model import FmnistTripletBatchAllModel
from src.utils import consts


class FmnistTBABatchSizeExperimentExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "fmnist_tba_batch_size_exp"

    @property
    def params(self):
        return {
            consts.TRAIN_STEPS: 15 * 1000,
            consts.EXCLUDED_KEYS: [1, 2, 3],
            consts.PREDICT_SIMILARITY_MARGIN: 3.0,
            consts.NUM_CHANNELS: 32,
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.EMBEDDING_SIZE: 64,
            consts.BATCH_SIZE: 64,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
        }


class BatchSizeAwareFmnistTBAModel(FmnistTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
            {
                "bs": self.bs,
            })

    def __init__(self, bs) -> None:
        super().__init__()
        self.bs = bs

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.BATCH_SIZE: self.bs
                               })


launcher = FmnistTBABatchSizeExperimentExperimentLauncher([
    BatchSizeAwareFmnistTBAModel(32),
    BatchSizeAwareFmnistTBAModel(64),
    BatchSizeAwareFmnistTBAModel(128),
    BatchSizeAwareFmnistTBAModel(256),
    # BatchSizeAwareFmnistTBAModel(512),  #ResourceExhaustedError
    # BatchSizeAwareFmnistTBAModel(1024),
])
