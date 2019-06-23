from src.estimator.launcher.experiments.triplets_hard_batch_all import tba_embedding_size_experiment
from src.estimator.launcher.launchers import Launcher, RunData
from src.utils import utils


def provide_launcher() -> Launcher:
    # return DefaultLauncher([
    #     ExtruderTripletBatchAllModel()
    # ])
    return tba_embedding_size_experiment.launcher


def provide_single_run_data() -> RunData:
    launcher = provide_launcher()
    if launcher.is_experiment:
        return utils.user_run_selection(launcher)
    else:
        return launcher.runs_data[0]
