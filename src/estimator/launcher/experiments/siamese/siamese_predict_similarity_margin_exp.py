from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import consts


class SiamesePredictSimilarityMarginExperiment(ExperimentLauncher):
    @property
    def name(self):
        return "siamese_predict_similarity_margin_exp"


class PredictSimilarityMarginMnistSiameseModel(MnistSiameseModel):

    @property
    def summary(self) -> str:
        return "pm_" + str(self.predict_margin)

    def __init__(self, predict_margin) -> None:
        super().__init__()
        self.predict_margin = predict_margin

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.PREDICT_MARGIN: self.predict_margin})


launcher = SiamesePredictSimilarityMarginExperiment([
    PredictSimilarityMarginMnistSiameseModel(0.15),
    PredictSimilarityMarginMnistSiameseModel(0.20),
    PredictSimilarityMarginMnistSiameseModel(0.25),
    PredictSimilarityMarginMnistSiameseModel(0.3),
    PredictSimilarityMarginMnistSiameseModel(0.35),
])
