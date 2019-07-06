from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.tba_model import ExtruderTBAModel
from src.utils import consts


class ExtruderTBAMiscExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "tba_extruder_misc"

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
            consts.TRAIN_STEPS: 500,
            consts.SHUFFLE_BUFFER_SIZE: 10000,
            consts.EVAL_STEPS_INTERVAL: 100,
        }


class ExtruderTBAMiscModel(ExtruderTBAModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
            {
                "run_no": self.run_no,
            })

    def __init__(self, run_no) -> None:
        super().__init__()
        self.run_no = run_no


launcher = ExtruderTBAMiscExperimentLauncher([
    ExtruderTBAMiscModel(1),
    ExtruderTBAMiscModel(2),
    ExtruderTBAMiscModel(3),
    ExtruderTBAMiscModel(4),
    ExtruderTBAMiscModel(5),
    ExtruderTBAMiscModel(6),
    ExtruderTBAMiscModel(7),
    ExtruderTBAMiscModel(8),

])
