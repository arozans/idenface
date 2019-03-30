import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from helpers import tf_helpers, gen
from src.data import preparing_data
from src.data.saving import reading_tfrecords
from src.utils import utils, consts


@pytest.fixture()
def thor_image_path(patched_home_dir):
    import urllib.request
    thor_path = patched_home_dir / "downloaded/thor_is_here.png"
    thor_path.parent.mkdir()
    image_address = 'https://i.stack.imgur.com/Cr57x.png'
    # image_address = 'https://liquipedia.net/commons/images/0/0d/ThorCE.jpg'
    urllib.request.urlretrieve(image_address, thor_path)
    assert utils.check_filepath(thor_path, exists=True, is_directory=False, is_empty=False)
    yield str(thor_path)
    thor_path.unlink()


@pytest.fixture()
def patched_home_dir_path(patched_home_dir):
    yield str(patched_home_dir)


@pytest.fixture()
def tensor_5x4x3():
    tensor = np.array(
        [[[10, 20, 30], [40, 60, 70], [80, 90, 50], [40, 30, 20]],
         [[11, 21, 31], [41, 61, 71], [81, 91, 51], [41, 31, 21]],
         [[12, 22, 32], [42, 62, 72], [82, 92, 52], [42, 32, 22]],
         [[13, 23, 33], [43, 63, 73], [83, 93, 53], [43, 33, 23]],
         [[14, 24, 34], [44, 64, 74], [84, 94, 54], [44, 34, 24]]]
    ) / 100
    return tensor.astype(np.float32)


diff = 0.05


@pytest.mark.parametrize('batch_size', [1, 5, 120])
@pytest.mark.parametrize('encoding', [False, True], ids=lambda x: "with encoding" if x else "no encoding", )
def test_should_save_and_read_correctly(tensor_5x4x3, patched_home_dir_path, batch_size, encoding):
    left_images = np.array([tensor_5x4x3[:] - diff] * batch_size)
    right_image = np.array([tensor_5x4x3[:] + diff] * batch_size)
    mock_images_data = {
        consts.LEFT_FEATURE_IMAGE: left_images,
        consts.RIGHT_FEATURE_IMAGE: right_image,
    }
    mock_image_labels = gen.labels_dict(batch_size=batch_size)

    tfrecord_full_path = preparing_data.save_to_tfrecord(mock_images_data, mock_image_labels,
                                                         str(patched_home_dir_path + '/data'), encoding)

    assert utils.check_filepath(tfrecord_full_path, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, encoding)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()
    _check_result(batch_size, first_batch, tensor_5x4x3, mock_image_labels)


def _check_result(batch_size, first_batch, tensor_5x4x3, labels):
    left_images, right_images, pair_labels, left_labels, right_labels = tf_helpers.unpack_batch(first_batch)
    assert len(left_images) == len(right_images) == len(pair_labels) == len(left_labels) == len(
        right_labels) == batch_size
    for left_image, right_image in zip(left_images, right_images):
        assert np.allclose(left_image + 0.5, tensor_5x4x3 - diff, rtol=1.e-4, atol=1.e-4)
        assert np.allclose(right_image + 0.5, tensor_5x4x3 + diff, rtol=1.e-4, atol=1.e-4)
    assert (pair_labels == labels[consts.PAIR_LABEL]).all()
    assert (left_labels == labels[consts.LEFT_FEATURE_LABEL]).all()
    assert (right_labels == labels[consts.RIGHT_FEATURE_LABEL]).all()


@pytest.mark.parametrize('encoding', [False, True])
def test_should_save_image_correctly_read_and_show(patched_home_dir_path, thor_image_path, encoding):
    show = False

    thor = mpimg.imread(tf_helpers.get_string(thor_image_path))

    image_arr = thor[None, :]

    if show:
        plt.imshow(image_arr.squeeze())
        plt.title('before')
        plt.show()

    two_images = {
        consts.LEFT_FEATURE_IMAGE: image_arr,
        consts.RIGHT_FEATURE_IMAGE: image_arr
    }
    label_dict = gen.labels_dict()

    tfrecord_full_path = preparing_data.save_to_tfrecord(two_images, label_dict, str(patched_home_dir_path + '/thor'),
                                                         encoding)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, encoding)

    left_images, _, _, _, _ = tf_helpers.unpack_first_batch(dataset)

    decoded_thor = left_images + 0.5

    if show:
        plt.imshow(decoded_thor)
        plt.title('after')
        plt.show()

    assert np.squeeze(decoded_thor).shape == np.squeeze(image_arr).shape
    assert np.allclose(decoded_thor, image_arr)
