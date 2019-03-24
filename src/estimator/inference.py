from pathlib import Path

import tensorflow as tf

from src.estimator.launcher import providing_launcher
from src.estimator.launcher.launchers import RunData
from src.estimator.training import training
from src.utils import image_summaries, filenames, consts


def run_inference(run_data: RunData, predicted_images_path: Path, show=False):
    images_count = consts.INFER_IMAGE_COUNT
    features, labels = get_infer_data(run_data, images_count)
    print("Obtained features dict to predict with entries shape {} and labels with shape {}".format(
        list(features.values())[0].shape,
        labels.shape))
    result = predict(run_data, features, labels, images_count)
    print("Obtained result dict with keys {} and shapes: {}".format(result.keys(),
                                                                    [x.shape for x in list(result.values())]))

    predicted_labels = run_data.model.get_predicted_labels(result)
    print("Predicted accuraccy: {}".format(
        ((predicted_labels == labels).mean()) if predicted_labels is not None else "Unknown"))

    print("Image saved into: {}".format(predicted_images_path))

    image_summaries.create_pair_board(
        features,
        labels,
        predicted_labels=predicted_labels,
        predicted_scores=run_data.model.get_predicted_scores(result),
        path=predicted_images_path,
        show=show
    )


def predict(run_data, infer_features, infer_labels, images_count):
    estimator = training.create_estimator(run_data)
    result = estimator.predict(
        input_fn=lambda: tf.data.Dataset.from_tensor_slices((infer_features, infer_labels)).batch(
            images_count).make_one_shot_iterator().get_next(),  # todo maybe dict would be enough, no need for dataset
        yield_single_examples=False
    )
    return list(result)[0]


def get_infer_data(run_data: RunData, batch_size: int):
    dataset = run_data.model.get_dataset_provider().infer(batch_size)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()

    with tf.Session() as sess:
        res = sess.run(first_batch)
    return res[0], res[1]


if __name__ == '__main__':
    run_data = providing_launcher.get_run_data()
    predicted_images_path = filenames.get_infer_dir() / filenames.create_infer_images_name(run_data.model)
    run_inference(run_data, predicted_images_path=None, show=True)
