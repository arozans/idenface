from src.estimator.launcher.experiments.siamese import siamese_single_excluded_exp
from src.estimator.launcher.launchers import Launcher, RunData


def provide_launcher() -> Launcher:
    # return DefaultLauncher([
    ##    MnistCNNModel()
    # MnistSiameseModel()
    # ])
    return siamese_single_excluded_exp.launcher
    # return siamese_optimizer_exp.launcher
    # return standard_cnn_single_excluded_exp.launcher
    #
    # return DefaultLauncher([
    # ])
    # return siamese_predict_similarity_margin_and_train_margin_exp.launcher
    # return standard_cnn_batch_size.launcher
    # return standard_cnn_dataset_providers.launcher
    # return standard_cnn_images_encoding.launcher


def get_run_data() -> RunData:
    launcher = provide_launcher()
    return launcher.runs_data[0]
