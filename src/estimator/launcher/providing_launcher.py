from src.estimator.launcher.experiments.siamese import siamese_predict_similarity_margin_and_train_margin_exp
from src.estimator.launcher.launchers import Launcher, RunData
from src.utils import utils


def provide_launcher() -> Launcher:
    # return Launcher([
    #     MnistCNNModel()
    # MnistSiameseModel()
    #
    # ])
    # return standard_cnn_single_excluded_exp.launcher
    #
    # return DefaultLauncher([
    # ])
    return siamese_predict_similarity_margin_and_train_margin_exp.launcher
    # return standard_cnn_batch_size.launcher
    # return standard_cnn_dataset_providers.launcher
    # return standard_cnn_images_encoding.launcher


def provide_single_run_data() -> RunData:
    launcher = provide_launcher()
    if launcher.is_experiment:
        return utils.user_run_selection(launcher)
    else:
        return launcher.runs_data[0]
