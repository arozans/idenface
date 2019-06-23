from src.estimator.launcher.experiments.siamese import siamese_extruder_exp
from src.estimator.launcher.launchers import Launcher, RunData
from src.utils import utils


def provide_launcher() -> Launcher:
    # return DefaultLauncher([
    #     ExtruderSiameseModel()
    # ])
    return siamese_extruder_exp.launcher


def provide_single_run_data() -> RunData:
    launcher = provide_launcher()
    if launcher.is_experiment:
        return utils.user_run_selection(launcher)
    else:
        return launcher.runs_data[0]
