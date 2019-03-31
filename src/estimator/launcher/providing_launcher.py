from src.estimator.launcher.launchers import Launcher, RunData, DefaultLauncher
from src.estimator.model.siamese_conv_model import MnistSiameseModel


def provide_launcher() -> Launcher:
    return DefaultLauncher([
        # MnistCNNModel()
        MnistSiameseModel()

    ])
    # return standard_cnn_single_excluded_exp.launcher
    #
    # return DefaultLauncher([
    # ])

    # return standard_cnn_batch_size.launcher
    # return standard_cnn_dataset_providers.launcher
    # return standard_cnn_images_encoding.launcher


def get_run_data() -> RunData:
    launcher = provide_launcher()
    return launcher.runs_data[0]
