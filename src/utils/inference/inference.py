from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from src.estimator.launcher import providing_launcher
from src.estimator.launcher.launchers import RunData
from src.estimator.training import training
from src.utils import image_summaries, filenames, consts, before_run, utils
from src.utils.consts import INFER_PLOT_CLUSTERS_NAME
from src.utils.image_summaries import map_pair_of_points_to_plot_data


def find_latest_ing_dir(run_text_logs_dir: Path) -> Path:
    date_to_file = {file.stat().st_ctime: file for file in run_text_logs_dir.iterdir()}
    max_date = max(date_to_file.keys())
    return date_to_file[max_date]


def copy_text_log(run_data):
    inference_dir = filenames.get_infer_dir(run_data)
    run_text_logs_dir = filenames.get_run_text_logs_dir(run_data)
    if not run_text_logs_dir.exists():
        utils.log("{} not exists - not copying text log".format(run_text_logs_dir))
        return
    latest_log = find_latest_ing_dir(run_text_logs_dir)
    import shutil
    shutil.copy(str(latest_log.absolute()), str((inference_dir / latest_log.name).absolute()))


def single_run_inference(run_data: RunData, show=False):
    model = run_data.model
    before_run.prepare_infer_env(run_data)

    features_dict, labels_dict = _get_infer_data(run_data, consts.INFER_IMAGE_COUNT)

    predictions: Dict[str, np.ndarray] = _predict(run_data, features_dict, labels_dict, consts.INFER_IMAGE_COUNT)

    _log_predictions(predictions)
    plot_predicted_data(run_data, features_dict, labels_dict, predictions, show)
    copy_text_log(run_data)
    _log_metrics(labels_dict, model.get_predicted_labels(predictions))


def _log_predictions(predictions):
    _log_dict_shapes('predictions', predictions)
    utils.log(predictions)


def plot_predicted_data(run_data: RunData, features_dict: Dict[str, np.ndarray], labels_dict: Dict[str, np.ndarray],
                        predictions: Dict[str, np.ndarray], show):
    model = run_data.model
    inference_dir = filenames.get_infer_dir(run_data)

    utils.log("Plotting pairs...")
    image_summaries.create_pairs_board(
        features_dict,
        labels_dict,
        predicted_labels=model.get_predicted_labels(predictions),
        predicted_scores=model.get_predicted_scores(predictions),
        path=inference_dir / filenames.summary_to_name(model, suffix=consts.PNG, with_date_fragment=False,
                                                       name=consts.INFER_PLOT_BOARD_NAME),
        show=show
    )
    if model.produces_2d_embedding:
        utils.log("Plotting distances...")
        x, y = map_pair_of_points_to_plot_data(
            predictions[consts.INFERENCE_LEFT_EMBEDDINGS],
            predictions[consts.INFERENCE_RIGHT_EMBEDDINGS]
        )
        image_summaries.create_distances_plot(
            left_coors=x,
            right_coors=y,
            labels_dict=labels_dict,
            infer_result=predictions,
            path=inference_dir / filenames.summary_to_name(model, suffix=consts.PNG, with_date_fragment=False,
                                                           name=consts.INFER_PLOT_DISTANCES_NAME),
            show=show
        )

        utils.log("Plotting clusters...")
        image_summaries.create_clusters_plot(
            feat=np.concatenate(
                (predictions[consts.INFERENCE_LEFT_EMBEDDINGS], predictions[consts.INFERENCE_RIGHT_EMBEDDINGS])),
            labels=np.concatenate((labels_dict[consts.TFRECORD_LEFT_LABEL], labels_dict[consts.TFRECORD_RIGHT_LABEL])),
            path=inference_dir / filenames.summary_to_name(model, suffix=consts.PNG, with_date_fragment=False,
                                                           name=INFER_PLOT_CLUSTERS_NAME),
            show=show
        )


def _predict(run_data: RunData, features_dict_to_infer: Dict[str, np.ndarray], _: Dict[str, np.ndarray],
             images_count: int):
    estimator = training.create_estimator(run_data)
    result = estimator.predict(
        input_fn=lambda: tf.data.Dataset.from_tensor_slices((features_dict_to_infer, _)).batch(
            images_count).make_one_shot_iterator().get_next(),
        # todo maybe dict would be enough, no need for dataset
        yield_single_examples=False
    )
    return list(result)[0]


def _get_infer_data(run_data: RunData, batch_size: int):
    dataset = run_data.model.get_dataset_provider().infer(batch_size)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()

    with tf.Session() as sess:
        res = sess.run(first_batch)
    _log_dict_shapes('inference features', res[0])
    _log_dict_shapes('inference labels', res[1])
    return res[0], res[1]


def _log_metrics(labels_dict, predicted_labels):
    utils.log("Predicted accuracy: {}".format(
        ((np.squeeze(predicted_labels) == labels_dict[
            consts.PAIR_LABEL]).mean()) if predicted_labels is not None else "Unknown"
    ))


def _log_dict_shapes(name: str, dict_with_ndarrays: Dict[str, np.ndarray]):
    utils.log("Obtained {} dict with keys {} and shapes: {}".format(name, dict_with_ndarrays.keys(),
                                                                    [x.shape for x in
                                                                     list(dict_with_ndarrays.values())]))


def infer(show: bool = False):
    launcher_run_data: RunData = providing_launcher.provide_single_run_data()
    single_run_inference(launcher_run_data, show=show)


if __name__ == '__main__':
    infer()
