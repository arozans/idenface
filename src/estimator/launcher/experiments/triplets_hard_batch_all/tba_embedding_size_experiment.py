from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.triplet_batch_all_model import ExtruderTripletBatchAllModel
from src.utils import consts


class ExtruderTBAEmbeddingSizesExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "tba_extruder_embedding_size_exp"

    @property
    def params(self):
        return {
            consts.PREDICT_SIMILARITY_MARGIN: 7.0,  # before: 6.3, similar results
            consts.GLOBAL_SUFFIX: "v2",
            consts.TRAIN_STEPS: 800,
        }


class EmbeddingSizesAwareExtruderTBAModel(ExtruderTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
            {
                "dense_units": self.du,
            })

    def __init__(self, es) -> None:
        super().__init__()
        self.du = es

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.DENSE_UNITS: [self.du]
                               })


launcher = ExtruderTBAEmbeddingSizesExperimentLauncher([
    EmbeddingSizesAwareExtruderTBAModel(2),
    EmbeddingSizesAwareExtruderTBAModel(10),
    EmbeddingSizesAwareExtruderTBAModel(20),  # best result, 82%
    EmbeddingSizesAwareExtruderTBAModel(30),
    EmbeddingSizesAwareExtruderTBAModel(40),
    EmbeddingSizesAwareExtruderTBAModel(50),
    EmbeddingSizesAwareExtruderTBAModel(60),
    EmbeddingSizesAwareExtruderTBAModel(70),
    EmbeddingSizesAwareExtruderTBAModel(80),
    EmbeddingSizesAwareExtruderTBAModel(90),
])
