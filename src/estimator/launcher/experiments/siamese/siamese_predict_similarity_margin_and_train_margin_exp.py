from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import consts


class SiamesePredictAndTrainSimilarityMarginExperiment(ExperimentLauncher):
    @property
    def launcher_name(self):
        return "siamese_predict_and_train_similarity_margin_exp"


class PredictSimilarityMarginAndTrainMnistSiameseModel(MnistSiameseModel):

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
            consts.PREDICT_MARGIN: self.predict_margin,
            consts.TRAIN_MARGIN: self.train_margin
        })


launcher = SiamesePredictAndTrainSimilarityMarginExperiment([
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.3, 0.3),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.3, 0.4),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.3, 0.5),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.3, 0.6),

    PredictSimilarityMarginAndTrainMnistSiameseModel(0.35, 0.3),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.35, 0.4),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.35, 0.5),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.35, 0.6),

    PredictSimilarityMarginAndTrainMnistSiameseModel(0.4, 0.3),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.4, 0.4),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.4, 0.5),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.4, 0.6),

    PredictSimilarityMarginAndTrainMnistSiameseModel(0.45, 0.3),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.45, 0.4),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.45, 0.5),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.45, 0.6),

    PredictSimilarityMarginAndTrainMnistSiameseModel(0.5, 0.3),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.5, 0.4),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.5, 0.5),
    PredictSimilarityMarginAndTrainMnistSiameseModel(0.5, 0.6),
])
