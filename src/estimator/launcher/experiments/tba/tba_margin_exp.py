from abc import ABC
from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.tba_model import TBAModel, ExtruderTBAModel
from src.utils import consts


class TBAMarginExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "TBA_margin_exp"

    def run_spec(self):
        return (
            # DELETE_AND_RESTART delete current run true/false
            # FINISH_MISSING Econtinue with next (omit interrupted one_ model
            # FINISH_INRERRUPTED Econtinue with next (omit interrupted one_ model
            # RETRY``` continue with next run (omit interrupted and next ones)
        )


class MarginParamsAwareTBAModel(TBAModel, ABC):

    @property
    def summary(self) -> str:
        return self._summary_from_dict(
            {
                'htm': self.hard_triplet_margin,
                'psm': self.predict_similarity_margin,
            }, stem=self.name[:1] + '_' + self.raw_data_provider.description.variant.name.lower()[:1])

    def __init__(self, predict_similarity_margin, hard_triplet_margin) -> None:
        super().__init__()
        self.predict_similarity_margin = predict_similarity_margin
        self.hard_triplet_margin = hard_triplet_margin

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.PREDICT_SIMILARITY_MARGIN: self.predict_similarity_margin,
                                   consts.HARD_TRIPLET_MARGIN: self.hard_triplet_margin
                               })


class ExtruderMarginParamsAwareTBAModel(MarginParamsAwareTBAModel, ExtruderTBAModel):
    pass


launcher = TBAMarginExperimentLauncher([
    # MnistMarginParamsAwareTBAModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[7, 5, 3, 3, 3], dense=[]),
    # # MnistMarginParamsAwareTBAModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    # #                                  dropout_rates=[0.5, None]),
    #
    # ######
    #
    # FmnistMarginParamsAwareTBAModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[7, 5, 3, 3, 3], dense=[]),
    # # FmnistMarginParamsAwareTBAModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    # #                                   dropout_rates=[0.5, None]),

    #####
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.0, hard_triplet_margin=0.6),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.0, hard_triplet_margin=0.5),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.0, hard_triplet_margin=0.4),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.0, hard_triplet_margin=0.3),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=15.0, hard_triplet_margin=0.3),

    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.8),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=1),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=10, hard_triplet_margin=1),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=2),

    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.6),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.5),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.4),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.3),

    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.6, hard_triplet_margin=0.6),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.6, hard_triplet_margin=0.5),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.6, hard_triplet_margin=0.4),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.6, hard_triplet_margin=0.3),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=4.0, hard_triplet_margin=0.5),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=3.0, hard_triplet_margin=0.5),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=2.0, hard_triplet_margin=0.5),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=1.0, hard_triplet_margin=0.5),
    ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=7.0, hard_triplet_margin=0.5),
    #
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=1.0, hard_triplet_margin=0.1),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=1.0, hard_triplet_margin=0.2),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=1.0, hard_triplet_margin=0.3),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=1.0, hard_triplet_margin=0.05),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=1.0, hard_triplet_margin=1.0),
    #
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.5),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.4),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.3),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.2),
    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=0.1),

    # ExtruderMarginParamsAwareTBAModel(predict_similarity_margin=6.3, hard_triplet_margin=3),

])

# ExtruderMarginParamsAwareTBAModel(filters=[32, 8], kernel_side_lengths=[5, `5], dense=[], concat_dense=[20, 2],
#                                     dropout_rates=[0.5, None])
