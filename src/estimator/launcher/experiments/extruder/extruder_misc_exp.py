from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.triplet_batch_all_model import ExtruderTripletBatchAllModel
from src.utils import consts


class ExtruderMiscExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "extruder_misc"

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


class ExtruderMiscTripletBatchAllModel(ExtruderTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
            {
                "run_no": self.run_no,
            })

    def __init__(self, run_no) -> None:
        super().__init__()
        self.run_no = run_no


launcher = ExtruderMiscExperimentLauncher([
    ExtruderMiscTripletBatchAllModel(1),
    ExtruderMiscTripletBatchAllModel(2),
    ExtruderMiscTripletBatchAllModel(3),
    ExtruderMiscTripletBatchAllModel(4),
    ExtruderMiscTripletBatchAllModel(5),
    ExtruderMiscTripletBatchAllModel(6),
    ExtruderMiscTripletBatchAllModel(7),
    ExtruderMiscTripletBatchAllModel(8),

])
