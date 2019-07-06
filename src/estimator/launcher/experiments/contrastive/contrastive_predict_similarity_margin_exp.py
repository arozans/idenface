from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import MnistContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.utils import consts


class PredictSimilarityMarginMnistContrastiveExperiment(ExperimentLauncher):
    @property
    def name(self):
        return "contrastive_predict_similarity_margin_exp"


class PredictSimilarityMarginMnistContrastiveModel(MnistContrastiveModel):

    @property
    def summary(self) -> str:
        return "pm_" + str(self.predict_margin)

    def __init__(self, predict_margin) -> None:
        super().__init__()
        self.predict_margin = predict_margin

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.PREDICT_SIMILARITY_MARGIN: self.predict_margin})


launcher = PredictSimilarityMarginMnistContrastiveExperiment([
    PredictSimilarityMarginMnistContrastiveModel(0.15),
    PredictSimilarityMarginMnistContrastiveModel(0.20),
    PredictSimilarityMarginMnistContrastiveModel(0.25),
    PredictSimilarityMarginMnistContrastiveModel(0.3),
    PredictSimilarityMarginMnistContrastiveModel(0.35),
])
