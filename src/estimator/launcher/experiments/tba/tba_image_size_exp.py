from typing import Dict, Any

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import ExtruderRawDataProvider
from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.tba_model import ExtruderTBAModel
from src.utils import consts


class ExtruderTBAImageSizeExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "tba_extruder_image_size"

    @property
    def params(self):
        return {
            consts.NUM_CHANNELS: 32,
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.PREDICT_SIMILARITY_MARGIN: 4.0,
            consts.DENSE_UNITS: [80],
            consts.BATCH_SIZE: 400,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
            consts.SHUFFLE_BUFFER_SIZE: 10000,
            consts.EVAL_STEPS_INTERVAL: 50,
        }


class ExtruderTBAImageSizeAwareTBAModel(ExtruderTBAModel):

    @property
    def summary(self) -> str:
        return self._summary_from_dict(
            {
                "size": self.im_size,
                "no": self.no
            })

    def __init__(self, im_size, no) -> None:
        super().__init__()
        self.no = no
        self.im_size = im_size

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return ExtruderRawDataProvider(self.im_size)

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.TRAIN_STEPS: 800 if self.im_size > 99 else 1000,
                               })


launcher = ExtruderTBAImageSizeExperimentLauncher([
    ExtruderTBAImageSizeAwareTBAModel(150, 1),  # accuraccy 83
    ExtruderTBAImageSizeAwareTBAModel(150, 2),
    ExtruderTBAImageSizeAwareTBAModel(150, 3),
    ExtruderTBAImageSizeAwareTBAModel(200, 1),
    ExtruderTBAImageSizeAwareTBAModel(200, 2),
    ExtruderTBAImageSizeAwareTBAModel(200, 3),
])
