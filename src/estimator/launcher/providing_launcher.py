from src.estimator.launcher.launchers import Launcher, RunData, DefaultLauncher
from src.estimator.model.siamese_conv_model import FmnistSiameseModel
from src.utils import utils


def provide_launcher() -> Launcher:
    return DefaultLauncher([
        FmnistSiameseModel()
    ])
    # return siamese_extruder_exp.launcher


def provide_single_run_data() -> RunData:
    launcher = provide_launcher()
    if launcher.is_experiment:
        return utils.user_run_selection(launcher)
    else:
        return launcher.runs_data[0]
