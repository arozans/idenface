from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.triplet_batch_all_model import FmnistTripletBatchAllModel
from src.utils import consts


class FmnistTBAEmbeddingSizesExperimentExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "fmnist_tba_embedding_size_exp"

    @property
    def params(self):
        return {
            consts.TRAIN_STEPS: 5 * 1000,
            consts.EXCLUDED_KEYS: [1, 2, 3],
            consts.PREDICT_SIMILARITY_MARGIN: 3.0,
            consts.NUM_CHANNELS: 64,
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.BATCH_SIZE: 32,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
        }


class EmbeddingSizesAwareFmnistTBAModel(FmnistTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
            {
                "embed_size": self.es,
            })

    def __init__(self, nc) -> None:
        super().__init__()
        self.es = nc

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.EMBEDDING_SIZE: self.es
                               })


launcher = FmnistTBAEmbeddingSizesExperimentExperimentLauncher([
    EmbeddingSizesAwareFmnistTBAModel(32),
    EmbeddingSizesAwareFmnistTBAModel(64),
    EmbeddingSizesAwareFmnistTBAModel(128),
    EmbeddingSizesAwareFmnistTBAModel(256),
])
