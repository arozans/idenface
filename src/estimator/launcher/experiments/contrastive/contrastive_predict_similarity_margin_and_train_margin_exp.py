from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import MnistContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.utils import consts


class ContrastivePredictAndTrainSimilarityMarginExperiment(ExperimentLauncher):
    @property
    def name(self):
        return "contrastive_predict_and_train_similarity_margin_exp"


class PredictSimilarityMarginAndTrainMnistContrastiveModel(MnistContrastiveModel):

    @property
    def summary(self) -> str:
        return "pm_" + str(self.predict_margin) + '_' + "tm_" + str(self.train_margin)

    def __init__(self, predict_margin, train_margin) -> None:
        super().__init__()
        self.predict_margin = predict_margin
        self.train_margin = train_margin

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {
            consts.PREDICT_SIMILARITY_MARGIN: self.predict_margin,
            consts.TRAIN_SIMILARITY_MARGIN: self.train_margin
        })


launcher = ContrastivePredictAndTrainSimilarityMarginExperiment([
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.3, 0.2),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.3, 0.3),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.3, 0.4),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.3, 0.5),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.3, 0.6),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.3, 0.7),

    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.35, 0.2),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.35, 0.3),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.35, 0.4),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.35, 0.5),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.35, 0.6),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.35, 0.7),

    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.4, 0.2),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.4, 0.3),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.4, 0.4),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.4, 0.5),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.4, 0.6),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.4, 0.7),

    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.45, 0.2),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.45, 0.3),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.45, 0.4),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.45, 0.5),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.45, 0.6),
    # PredictSimilarityMarginAndTrainMnistContrastiveModel(0.45, 0.7),

    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.5, 0.2),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.5, 0.3),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.5, 0.4),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.5, 0.5),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.5, 0.6),
    PredictSimilarityMarginAndTrainMnistContrastiveModel(0.5, 0.7),
])
