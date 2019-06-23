from pathlib import Path
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tfmpl
from matplotlib.figure import Figure

from src.data.common_types import DatasetSpec, DatasetType, DataDescription, DictsDataset, LabelsDict
from src.estimator.launcher.launchers import RunData
from src.utils import utils, consts, filenames
from src.utils.configuration import config


def _create_tfmpl_figure(l, r, pair_label, left_label, right_label) -> Figure:
    fig: Figure = tfmpl.create_figure()

    a = fig.add_subplot(1, 2, 1)
    a.imshow(l)
    a.axis('off')

    b = fig.add_subplot(1, 2, 2)
    b.imshow(r)
    b.axis('off')

    fig.tight_layout()

    if str(pair_label) == '1':
        figure_title = "same ({}-{})".format(left_label, right_label)
    else:
        figure_title = "different ({}-{})".format(left_label, right_label)
    fig.suptitle(figure_title, fontsize=32)

    return fig


def create_pair_summaries(run_data: RunData):
    dataset_provider_cls = run_data.model.raw_data_provider
    tf.reset_default_graph()
    batch_size = 10
    utils.log('Creating {} sample features summaries'.format(batch_size))
    dataset: tf.data.Dataset = run_data.model.dataset_provider.supply_dataset(
        dataset_spec=DatasetSpec(dataset_provider_cls,
                                 DatasetType.TEST,
                                 with_excludes=False,
                                 encoding=run_data.model.dataset_provider.is_encoded()),
        shuffle_buffer_size=10000, batch_size=batch_size,
        prefetch=False)
    iterator = dataset.make_one_shot_iterator()
    iterator = iterator.get_next()
    with tf.Session() as sess:
        left = iterator[0][consts.LEFT_FEATURE_IMAGE]
        right = iterator[0][consts.RIGHT_FEATURE_IMAGE]
        pair_labels = iterator[1][consts.PAIR_LABEL]
        left_labels = iterator[1][consts.LEFT_FEATURE_LABEL]
        right_labels = iterator[1][consts.RIGHT_FEATURE_LABEL]
        pairs_imgs_summary = create_pair_summary(
            left,
            right,
            pair_labels,
            left_labels,
            right_labels,
            dataset_provider_cls.description
        )

        image_summary = tf.summary.image('paired_images', pairs_imgs_summary, max_outputs=batch_size)
        all_summaries = tf.summary.merge_all()

        dir = filenames.get_run_logs_data_dir(run_data) / 'features'
        dir.mkdir(exist_ok=True, parents=True)
        writer = tf.summary.FileWriter(str(dir), sess.graph)

        sess.run(tf.global_variables_initializer())

        summary = sess.run(all_summaries)
        writer.add_summary(summary)
        writer.flush()


@tfmpl.figure_tensor
def create_pair_summary(left_image: np.ndarray,
                        right_image: np.ndarray,
                        pair_labels: np.ndarray,
                        left_labels: np.ndarray,
                        right_labels: np.ndarray,
                        description: DataDescription):
    left_image, right_image = [x.reshape([-1, description.image_dimensions.width, description.image_dimensions.height,
                                          description.image_dimensions.channels]).squeeze() + 0.5 for x in
                               (left_image, right_image)]
    images = []
    _maybe_enable_dark_background()
    for left, right, pair_label, left_label, right_label in zip(left_image, right_image, pair_labels, left_labels,
                                                                right_labels):
        fig = _create_tfmpl_figure(left, right, pair_label, left_label, right_label)
        images.append(fig)

    return images


def _maybe_save_and_show(fig, path, show, with_title: bool):
    if path:
        if with_title:
            fig.suptitle(path.stem, fontsize=16)
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)
    if show:
        plt.show()


def create_pair_image(index, left_images, right_images):
    pixel_division_line = np.zeros(shape=[left_images.shape[1], 1, left_images.shape[-1]])
    return np.concatenate((
        left_images[index, :] + 0.5,
        pixel_division_line,
        right_images[index, :] + 0.5
    ), axis=1)


def translate_label(index, labels):
    if labels is None:
        return "unknown"
    return "same" if labels[index] == 1 else "diff"


def format_score(index, scores):
    return (" (" + "{:0.3f}".format(np.squeeze(scores)[index]) + ")") if (
            scores is not None or np.array(scores) != None) else ""


def add_tick_or_cross(index, predicted_labels, labels_dict):
    predicted = predicted_labels[index] if predicted_labels is not None else 1
    return " " \
           + (u"\u2714" if predicted == labels_dict.pair[index] else u"\u2718") \
           + (" ({}‐{})".format(labels_dict.left[index], labels_dict.right[index]))


@tfmpl.figure_tensor
def draw_tf_clusters_plot(feat, labels):
    fig: Figure = tfmpl.create_figure()
    return _draw_scatter(fig, feat, labels)


def _draw_scatter(figure, feat, labels):
    _maybe_enable_dark_background()
    ax = figure.add_subplot(1, 1, 1)
    for j in range(10):
        ax.plot(feat[labels == j, 0].flatten(), feat[labels == j, 1].flatten(), '.', c=get_infer_plot_colors()[j],
                markersize=15)

    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    return figure


def get_infer_plot_colors():
    colors = consts.INFER_PLOT_COLORS
    if not config[consts.IS_INFER_PLOT_DARK_BG_ENABLED]:
        colors = ["black" if x == "white" else x for x in colors]
    return colors


def map_pair_of_points_to_plot_data(left_points, right_points):
    left_x = left_points[:, 0]
    right_x = right_points[:, 0]

    left_y = left_points[:, 1]
    right_y = right_points[:, 1]

    return np.array([left_x, right_x]), np.array([left_y, right_y])


def _maybe_enable_dark_background():
    if config[consts.IS_INFER_PLOT_DARK_BG_ENABLED]:
        plt.style.use('dark_background')


def create_pairs_board(dataset: DictsDataset,
                       predicted_labels: np.ndarray,
                       predicted_scores: np.ndarray = None,
                       cols: int = 5,
                       max_rows: int = 5,
                       path: Optional[Path] = None,
                       show: bool = True):
    left_images = dataset.features.left
    right_images = dataset.features.right

    assert cols <= 10
    assert left_images.shape == right_images.shape
    if len(left_images.shape) == 2:
        left_images = left_images[None, :]
        right_images = right_images[None, :]
    images_num = left_images.shape[0]
    import math
    rows = math.ceil(images_num / cols)
    rows = rows if rows < max_rows else max_rows

    _maybe_enable_dark_background()

    fig = plt.figure(figsize=consts.INFER_FIG_SIZE)

    try:
        for row in np.arange(rows):
            for col in np.arange(cols):
                index = row * cols + col
                pair_image = create_pair_image(index, left_images, right_images)
                a = fig.add_subplot(rows, cols, index + 1)
                plt.imshow(np.squeeze(pair_image))
                a.set_title(_create_pair_desc(index, dataset.labels, predicted_labels, predicted_scores), fontsize=16)
                a.set_xticks([])
                a.set_yticks([])
    except IndexError:
        pass

    _maybe_save_and_show(fig, path, show, with_title=config[consts.IS_INFER_PLOT_BOARD_WITH_TITLE])


def _create_pair_desc(index, labels_dict, predicted_labels, predicted_scores) -> str:
    title = translate_label(index, predicted_labels) + \
            format_score(index, predicted_scores) + \
            add_tick_or_cross(index, predicted_labels, labels_dict)
    return r"$\bf{" + title + "}$"


def create_distances_plot(left_coors: np.ndarray,
                          right_coors: np.ndarray,
                          labels_dict: LabelsDict,
                          infer_result: Dict[str, np.ndarray],
                          path: Optional[Path] = None,
                          show: bool = True):
    fig = plt.figure(figsize=consts.INFER_FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)

    cycler = mpl.cycler(
        color=get_infer_plot_colors(),
        linestyle=['-', '--', ':', '-.', '-'] * 5
    )
    ax.set_prop_cycle(cycler)
    plot = ax.plot(left_coors, right_coors, marker='o')
    plt.setp(plot, linewidth=3, markersize=7)
    fig.legend([_create_pair_desc(idx, labels_dict, infer_result[consts.INFERENCE_CLASSES],
                                  infer_result[consts.INFERENCE_DISTANCES]) for idx in
                range(len(infer_result[consts.INFERENCE_CLASSES]))])
    plt.subplots_adjust(right=0.9)
    _maybe_save_and_show(fig, path, show, with_title=config[consts.IS_INFER_PLOT_BOARD_WITH_TITLE])


def create_clusters_plot(feat: np.ndarray, labels: np.ndarray, path: Optional[Path] = None,
                         show: bool = True):
    fig = plt.figure(figsize=consts.INFER_FIG_SIZE)
    _draw_scatter(figure=fig, feat=feat, labels=labels)
    plt.subplots_adjust(right=0.9)
    _maybe_save_and_show(fig, path, show, with_title=config[consts.IS_INFER_PLOT_BOARD_WITH_TITLE])
