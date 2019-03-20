from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

from helpers.tf_helpers import run_eagerly
from src.data.saving import reading_tfrecords
from src.data.saving.tf_saving import save_to_tfrecord
from src.utils import utils, consts


@pytest.fixture()
def thor_image_path(patched_home_dir):
    import urllib.request
    thor_path = patched_home_dir / "thor_is_here.png"
    image_address = 'https://i.stack.imgur.com/Cr57x.png'
    urllib.request.urlretrieve(image_address, thor_path)
    assert utils.check_filepath(thor_path, exists=True, is_directory=False, is_empty=False)
    yield str(thor_path)
    thor_path.unlink()


@pytest.fixture()
def patched_home_dir_path(patched_home_dir):
    yield str(patched_home_dir)


@pytest.fixture()
def tensor_5x4x3():
    return np.array(
        [[[10, 20, 30], [40, 60, 70], [80, 90, 50], [40, 30, 20]],
         [[11, 21, 31], [41, 61, 71], [81, 91, 51], [41, 31, 21]],
         [[12, 22, 32], [42, 62, 72], [82, 92, 52], [42, 32, 22]],
         [[13, 23, 33], [43, 63, 73], [83, 93, 53], [43, 33, 23]],
         [[14, 24, 34], [44, 64, 74], [84, 94, 54], [44, 34, 24]]]
    )


@run_eagerly
@pytest.mark.parametrize('batch_size', [1, 5, 120])
def test_should_save_and_read_correctly(tensor_5x4x3, patched_home_dir_path, batch_size):
    tensor_5x4x3 = tf.cast(tensor_5x4x3, tf.float32).numpy()
    tmp_foo_bar_dir = Path(patched_home_dir_path.numpy().decode("utf-8"))
    batch_size = batch_size.numpy()

    left_images = [tensor_5x4x3[:] + 1] * batch_size
    right_image = [tensor_5x4x3[:] - 1] * batch_size
    mock_images_data = {
        consts.LEFT_FEATURE_IMAGE: left_images,
        consts.RIGHT_FEATURE_IMAGE: right_image,
    }
    mock_image_labels = np.array([1] * batch_size)

    tmp_foo_bar_tf_file = (tmp_foo_bar_dir / 'foobar.tfrecord')

    save_to_tfrecord(mock_images_data, mock_image_labels, tmp_foo_bar_tf_file)
    assert utils.check_filepath(tmp_foo_bar_tf_file, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tmp_foo_bar_dir)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    first_batch = iterator.get_next()
    _check_result(batch_size, first_batch, tensor_5x4x3)


def _check_result(batch_size, first_batch, tensor_5x4x3):
    features = first_batch[0]
    labels = first_batch[1].numpy()
    left_batch = (features[consts.LEFT_FEATURE_IMAGE]).numpy()
    right_batch = (features[consts.RIGHT_FEATURE_IMAGE]).numpy()
    assert len(left_batch) == len(right_batch) == len(labels) == batch_size
    for left_image, right_image in zip(left_batch, right_batch):
        assert (left_image == tensor_5x4x3[:] + 1 - 0.5).all()  # normalizing!
        assert (right_batch == tensor_5x4x3[:] - 1 - 0.5).all()


@run_eagerly
def test_should_save_image_correctly_read_and_show(patched_home_dir_path, thor_image_path):
    patched_home_dir_path = patched_home_dir_path.numpy().decode("utf-8")
    thor_image_path = thor_image_path.numpy().decode("utf-8")
    tmp_foo_bar_tf_file = Path(patched_home_dir_path) / 'thor.tfrecord'

    # png image encoded as floats, need to be converted to float before displaying
    image_arr = np.array([mpimg.imread(thor_image_path)])

    thor = mpimg.imread(thor_image_path)
    plt.imshow(thor)
    plt.title('before')

    # show for manual testing
    # plt.show()

    two_images = {consts.LEFT_FEATURE_IMAGE: image_arr,
                  consts.RIGHT_FEATURE_IMAGE: image_arr}
    label = np.array([1])
    save_to_tfrecord(two_images, label, tmp_foo_bar_tf_file)

    dataset = reading_tfrecords.assemble_dataset(tmp_foo_bar_tf_file.parent)
    iterator = dataset.make_one_shot_iterator()

    first_batch = iterator.get_next()
    thor = (first_batch[0][consts.LEFT_FEATURE_IMAGE]).numpy()

    assert np.squeeze(thor).shape == np.squeeze(image_arr).shape

    plt.imshow(thor)
    plt.title('after')

    # show for manual testing
    # plt.show()
