from src.estimator.launcher.launchers import Launcher, RunData, DefaultLauncher
from src.estimator.model.triplet_batch_all_model import ExtruderTripletBatchAllModel
from src.utils import utils


def provide_launcher() -> Launcher:
    return DefaultLauncher([
        ExtruderTripletBatchAllModel()
    ])


def provide_single_run_data() -> RunData:
    launcher = provide_launcher()
    if launcher.is_experiment:
        return utils.user_run_selection(launcher)
    else:
        return launcher.runs_data[0]
