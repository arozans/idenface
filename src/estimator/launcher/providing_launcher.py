from src.estimator.launcher.launchers import Launcher, DefaultLauncher, RunData
from src.estimator.model.siamese_conv_model import MnistSiameseModel


def provide_launcher() -> Launcher:
    # return DefaultLauncher([
    #     MnistCNNModel()
    # ])
    # return standard_cnn_single_excluded_exp.launcher
    #
    return DefaultLauncher([
        MnistSiameseModel()
    ])

    # return standard_cnn_batch_size.launcher


def get_run_data() -> RunData:
    launcher = provide_launcher()
    return launcher.runs_data[0]
