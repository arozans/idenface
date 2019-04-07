from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from src.estimator.launcher import providing_launcher
from src.estimator.launcher.launchers import RunData
from src.estimator.training import training
from src.utils import image_summaries, filenames, consts
from src.utils.configuration import config
from src.utils.consts import INFER_PLOT_CLUSTERS_NAME
from src.utils.image_summaries import map_pair_of_points_to_plot_data


def run_inference(run_data: RunData, show=False) -> Path:
    model = run_data.model
    print("Running inference with model name: {}, and summary: {}, and params: {}".format(model.name, model.summary,
                                                                                          model.params))
    inference_dir = filenames.get_infer_dir(run_data)
    print("Inference data will be saved into: {}".format(inference_dir))

    _check_model_checkpoint_existence(run_data)

    images_count = consts.INFER_IMAGE_COUNT
    features_dict, labels_dict = _get_infer_data(run_data, images_count)
    _print_dict_shapes('inference features', features_dict)
    _print_dict_shapes('inference labels', labels_dict)

    predictions: Dict[str, np.ndarray] = _predict(run_data, features_dict, labels_dict, images_count)

    _print_dict_shapes('predictions', predictions)

    predicted_labels = model.get_predicted_labels(predictions)
    _print_metrics(labels_dict, predicted_labels)
    print(predictions)

    print("Plotting pairs...")
    image_summaries.create_pairs_board(
        features_dict,
        labels_dict,
        predicted_labels=predicted_labels,
        predicted_scores=model.get_predicted_scores(predictions),
        path=inference_dir / filenames.summary_to_name(model, suffix=consts.PNG, with_date_fragment=False,
                                                       name=consts.INFER_PLOT_BOARD_NAME),
        show=show
    )
    if model.produces_2d_embedding:
        print("Plotting distances...")
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

        print("Plotting clusters...")
        image_summaries.create_clusters_plot(
            feat=np.concatenate(
                (predictions[consts.INFERENCE_LEFT_EMBEDDINGS], predictions[consts.INFERENCE_RIGHT_EMBEDDINGS])),
            labels=np.concatenate((labels_dict[consts.TFRECORD_LEFT_LABEL], labels_dict[consts.TFRECORD_RIGHT_LABEL])),
            path=inference_dir / filenames.summary_to_name(model, suffix=consts.PNG, with_date_fragment=False,
                                                           name=INFER_PLOT_CLUSTERS_NAME),
            show=show
        )
    return inference_dir


def _predict(run_data, infer_features, infer_labels, images_count):
    estimator = training.create_estimator(run_data)
    result = estimator.predict(
        input_fn=lambda: tf.data.Dataset.from_tensor_slices((infer_features, infer_labels)).batch(
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
    return res[0], res[1]


def _check_model_checkpoint_existence(run_data: RunData):
    strict: bool = config.is_infer_checkpoint_obligatory
    if not strict:
        print("Not checking checkpoint existence")
        return
    model_dir = filenames.get_run_logs_data_dir(run_data)
    assert model_dir.exists(), "{} does not exists - no model to load!".format(model_dir)

    checkpoints = model_dir.glob('*.ckpt-*')
    checkpoints_with_number = {x for y in checkpoints for x in str(y).split('.') if x.startswith("ckpt")}
    step_numbers = {int(x.split('-')[-1]) for x in checkpoints_with_number}

    assert bool(step_numbers), "No checkpoints exists!"
    assert len(step_numbers) > 1 or 0 not in step_numbers, "Only one checkpoint  - for 0th step exists!"


def _print_metrics(labels_dict, predicted_labels):
    print("Predicted accuracy: {}".format(
        ((predicted_labels == labels_dict[consts.PAIR_LABEL]).mean()) if predicted_labels is not None else "Unknown"))


def _print_dict_shapes(name: str, dict_with_ndarrays: Dict[str, np.ndarray]):
    print("Obtained {} dict with keys {} and shapes: {}".format(name, dict_with_ndarrays.keys(),
                                                                [x.shape for x in list(dict_with_ndarrays.values())]))


if __name__ == '__main__':
    launcher_run_data = providing_launcher.get_run_data()
    infer_dir = run_inference(launcher_run_data, show=False)
