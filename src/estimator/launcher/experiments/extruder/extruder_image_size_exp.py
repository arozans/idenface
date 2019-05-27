from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import ExtruderRawDataProvider
from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.triplet_batch_all_model import ExtruderTripletBatchAllModel
from src.utils import consts


class ExtruderImageSizeExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "extruder_image_size"

    @property
    def params(self):
        return {
            consts.NUM_CHANNELS: 32,
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.PREDICT_SIMILARITY_MARGIN: 4.0,
            consts.EMBEDDING_SIZE: 80,
            consts.BATCH_SIZE: 400,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
            consts.TRAIN_STEPS: 500,
            consts.SHUFFLE_BUFFER_SIZE: 10000,
            consts.EVAL_STEPS_INTERVAL: 100,
        }


class ExtruderImageSizeAwareTBAModel(ExtruderTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
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


launcher = ExtruderImageSizeExperimentLauncher([
    ExtruderImageSizeAwareTBAModel(100, 1),
    ExtruderImageSizeAwareTBAModel(100, 2),
    ExtruderImageSizeAwareTBAModel(100, 3),
    ExtruderImageSizeAwareTBAModel(150, 1),
    ExtruderImageSizeAwareTBAModel(150, 2),
    ExtruderImageSizeAwareTBAModel(150, 3),
    ExtruderImageSizeAwareTBAModel(200, 1),
    ExtruderImageSizeAwareTBAModel(200, 2),
    # ExtruderImageSizeAwareTBAModel(200, 3),
])
