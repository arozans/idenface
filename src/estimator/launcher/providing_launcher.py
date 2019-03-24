from src.estimator.launcher.experiments import standard_cnn_dataset_providers
from src.estimator.launcher.launchers import Launcher, RunData


def provide_launcher() -> Launcher:
    # return DefaultLauncher([
    #     MnistCNNModel()
    # ])
    # return standard_cnn_single_excluded_exp.launcher
    #
    # return DefaultLauncher([
    #     MnistSiameseModel()
    # ])

    # return standard_cnn_batch_size.launcher
    return standard_cnn_dataset_providers.launcher


def get_run_data() -> RunData:
    launcher = provide_launcher()
    return launcher.runs_data[0]
