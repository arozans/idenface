from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tfmpl
from matplotlib.figure import Figure
from tensorflow.python.data import TFRecordDataset

from src.data.common_types import DatasetSpec, DatasetType
from src.estimator.launcher.launchers import RunData
from src.estimator.training import supplying_datasets
from src.utils import utils, consts, filenames


def _create_tfmpl_figure(l, r, title) -> Figure:
    fig: Figure = tfmpl.create_figure()

    a = fig.add_subplot(1, 2, 1)
    a.imshow(l)
    a.axis('off')

    b = fig.add_subplot(1, 2, 2)
    b.imshow(r)
    b.axis('off')

    fig.tight_layout()

    if str(title) == '1':
        figure_title = 'same'
    else:
        figure_title = 'different'
    fig.suptitle(figure_title, fontsize=32)

    return fig


def create_pair_summaries(run_data: RunData):
    dataset_provider_cls = run_data.model.dataset_provider_cls
    tf.reset_default_graph()
    batch_size = 10
    utils.log('Creating {} sample images summaries'.format(batch_size))
    dataset: TFRecordDataset = supplying_datasets.supply_dataset(dataset_spec=DatasetSpec(dataset_provider_cls,
                                                                                          DatasetType.TRAIN,
                                                                                          with_excludes=False),
                                                                 shuffle_buffer_size=10000, batch_size=batch_size,
                                                                 prefetch=False)
    iterator = dataset.make_one_shot_iterator()
    iterator = iterator.get_next()
    with tf.Session() as sess:
        left = iterator[0][consts.LEFT_FEATURE_IMAGE]
        right = iterator[0][consts.RIGHT_FEATURE_IMAGE]
        label = iterator[1]

        pairs_imgs_summary = create_pair_summary(left, right, label,
                                                 dataset_provider_cls.description().image_side_length)

        image_summary = tf.summary.image('paired_images', pairs_imgs_summary, max_outputs=batch_size)
        all_summaries = tf.summary.merge_all()

        dir = filenames.get_run_logs_data_dir(run_data) / 'images'
        dir.mkdir(exist_ok=True, parents=True)
        writer = tf.summary.FileWriter(str(dir), sess.graph)

        sess.run(tf.global_variables_initializer())

        summary = sess.run(all_summaries)
        writer.add_summary(summary)
        writer.flush()


@tfmpl.figure_tensor
def create_pair_summary(left: np.ndarray, right: np.ndarray, labels: np.ndarray, side_len: int):
    left = left.reshape([-1, side_len, side_len])
    right = right.reshape([-1, side_len, side_len])
    images = []
    for l, r, tit in zip(left, right, labels):
        fig = _create_tfmpl_figure(l, r, tit)
        images.append(fig)

    return images


def create_pair_board(features_dict: dict, true_labels: np.ndarray, predicted_labels: np.ndarray,
                      predicted_scores: np.ndarray = None,
                      cols: int = 5, max_rows: int = 5, path: Path = None, show: bool = True):
    left_images = np.squeeze(list(features_dict.values())[0])
    right_images = np.squeeze(list(features_dict.values())[1])
    predicted_scores = np.squeeze(predicted_scores)

    assert cols <= 10
    assert left_images.shape == right_images.shape
    if len(left_images.shape) == 2:
        left_images = left_images[None, :]
        right_images = right_images[None, :]
    images_num = left_images.shape[0]
    import math
    rows = math.ceil(images_num / cols)
    rows = rows if rows < max_rows else max_rows

    plt.style.use('dark_background')
    figw = 10
    figh = 8.3
    fig = plt.figure(figsize=(figw, figh))

    try:
        for row in np.arange(rows):
            for col in np.arange(cols):
                index = row * cols + col
                pair_image = create_pair_image(index, left_images, right_images)
                a = fig.add_subplot(rows, cols, index + 1)
                plt.imshow(pair_image)
                a.set_title(translate_label(index, predicted_labels)
                            + format_score(index, predicted_scores)
                            + add_tick_or_cross(index, predicted_labels, true_labels))
                a.set_xticks([])
                a.set_yticks([])
                fig.tight_layout()
                # plt.subplots_adjust(top=0.6)
    except IndexError:
        pass
    # plt.subplots_adjust(left=1/figw, right=1-1/figw, bottom=1/figh, top=1-1/figh)

    if path:
        fig.suptitle(path.parts[-1], fontsize=16)
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)

    if show:
        plt.show()


def create_pair_image(index, left_images, right_images):
    return np.concatenate(
        (left_images[index, :], np.zeros(shape=[left_images.shape[1], 1]), right_images[index, :]), axis=1)


def translate_label(index, labels):
    if labels is None:
        return "unknown"
    return "same" if labels[index] == 1 else "different"


def format_score(index, scores):
    return (" (" + "{:0.3f}".format(scores[index]) + ")") if (scores is not None or np.array(scores) != None) else ""


def add_tick_or_cross(index, labels, true_labels):
    predicted = labels[index] if labels is not None else 1
    return " " + (u"\u2714" if predicted == true_labels[index] else u"\u2718")


@tfmpl.figure_tensor
def draw_scatter(points):
    '''Draw scatter plots. One for each color.'''
    fig: Figure = tfmpl.create_figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.scatter(points[:, 0], points[:, 1], c='r')
    fig.tight_layout()
    return fig


@tfmpl.figure_tensor
def draw_scatters(feat, labels):
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900',
              '#009999']

    '''Draw scatter plots. One for each color.'''
    fig: Figure = tfmpl.create_figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    print("lol, ", feat.shape)
    print("lol2, ", feat[0].shape)
    print("lol3, ", feat[0, 0].shape)
    print("rolf, ", labels.shape)
    # print("rolf2, " ,labels[0].shape)
    # print("rolf3, " ,labels[0,0].shape)
    for j in range(10):
        searched_label_positions = tf.math.equal(labels, j)
        searched_label_coor = tf.where(searched_label_positions)

        zeros = tf.constant(0, dtype=tf.int64)[None, None]
        zeros = tf.tile(zeros, [tf.shape(searched_label_coor)[0], 1])  # Repeat rows. Shape=(tf.shape(a)[0], 1)
        left_coor = tf.concat([searched_label_coor, zeros], axis=1)

        ones = tf.constant(1, dtype=tf.int64)[None, None]
        ones = tf.tile(ones, [tf.shape(searched_label_coor)[0], 1])  # Repeat rows. Shape=(tf.shape(a)[0], 1)
        right_coor = tf.concat([searched_label_coor, ones], axis=1)
        left_indices = tf.gather_nd(feat, left_coor)
        right_indices = tf.gather_nd(feat, right_coor)
        ax.plot(left_indices.shape.as_list(), right_indices.shape.as_list(), '.', c=colors[j], alpha=0.8)

    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    fig.tight_layout()
    return fig
