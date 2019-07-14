from src.estimator.launcher.experiments.tba import tba_margin_exp
from src.estimator.launcher.launchers import Launcher, RunData
from src.utils import utils


def provide_launcher() -> Launcher:
    # return DefaultLauncher([
    #     MnistSoftmaxModel()
    # ])
    return tba_margin_exp.launcher


def provide_single_run_data() -> RunData:
    launcher = provide_launcher()
    if launcher.is_experiment:
        return utils.user_run_selection(launcher)
    else:
        return launcher.runs_data[0]
