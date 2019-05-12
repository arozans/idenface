from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.triplet_batch_all_model import FmnistTripletBatchAllModel, \
    FmnistTripletBatchAllUnpairedTrainModel
from src.utils import consts


class FmnistTBATrainPairedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "fmnist_tba_train_paired_exp"

    @property
    def params(self):
        return {
            consts.TRAIN_STEPS: 5 * 1000,
            consts.EXCLUDED_KEYS: [1, 2, 3],
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.PREDICT_SIMILARITY_MARGIN: 3.0,
            consts.NUM_CHANNELS: 64,
            consts.EMBEDDING_SIZE: 64,
            consts.BATCH_SIZE: 64,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
        }


launcher = FmnistTBATrainPairedExperimentLauncher([
    FmnistTripletBatchAllModel(),
    FmnistTripletBatchAllUnpairedTrainModel()
])