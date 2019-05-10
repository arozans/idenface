import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
import skimage

from data.tfrecord.conftest import random_images, _check_paired_result, _check_result
from src.data import preparing_data
from src.data.tfrecord.reading import reading_tfrecords
from src.utils import utils, consts
from testing_utils import tf_helpers, gen, testing_helpers


@pytest.mark.parametrize(consts.BATCH_SIZE, [1, 5, 120])
def test_should_save_and_read_pairs_correctly(patched_home_dir_path, batch_size):
    left_images = random_images(batch_size)
    right_images = random_images(batch_size)
    mock_images_data = {
        consts.LEFT_FEATURE_IMAGE: left_images,
        consts.RIGHT_FEATURE_IMAGE: right_images,
    }
    mock_images_labels = gen.paired_labels_dict(batch_size=batch_size)
    mock_images_data = testing_helpers.save_save_dataset_dict_on_disc(mock_images_data, mock_images_labels)
    dataset_spec = gen.dataset_spec(on_disc=True)
    tfrecord_full_path = preparing_data.save_to_tfrecord(mock_images_data, mock_images_labels,
                                                         str(patched_home_dir_path + '/data'),
                                                         dataset_spec)

    assert utils.check_filepath(tfrecord_full_path, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, dataset_spec)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()
    _check_paired_result(first_batch, (left_images, right_images), mock_images_labels)


@pytest.mark.parametrize(consts.BATCH_SIZE, [1, 5, 120])
def test_should_save_and_read_unpaired_correctly(patched_home_dir_path, batch_size):
    images = random_images(batch_size)
    mock_images_data = {
        consts.FEATURES: images,
    }
    mock_images_labels = gen.unpaired_labels_dict(batch_size=batch_size)
    mock_images_data = testing_helpers.save_save_dataset_dict_on_disc(mock_images_data, mock_images_labels)
    dataset_spec = gen.dataset_spec(on_disc=True, paired=False)
    tfrecord_full_path = preparing_data.save_to_tfrecord(mock_images_data, mock_images_labels,
                                                         str(patched_home_dir_path + '/data'),
                                                         dataset_spec)

    assert utils.check_filepath(tfrecord_full_path, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, dataset_spec)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()
    _check_result(first_batch, images, mock_images_labels)


def test_should_save_image_correctly_read_and_show(patched_home_dir_path, thor_image_path):
    show = False

    thor = mpimg.imread(tf_helpers.get_string(thor_image_path))
    thor = skimage.img_as_float(thor)
    image_arr = thor[None, :]

    if show:
        plt.imshow(image_arr.squeeze())
        plt.title('before')
        plt.show()

    features_as_paths = {
        consts.FEATURES: np.array([thor_image_path]),
    }

    labels = gen.unpaired_labels_dict()

    dataset_spec = gen.dataset_spec(on_disc=True, paired=False, encoding=False)
    tfrecord_full_path = preparing_data.save_to_tfrecord(features_as_paths, labels,
                                                         str(patched_home_dir_path + '/thor'),
                                                         dataset_spec)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, dataset_spec)

    left_images, _ = tf_helpers.unpack_first_batch(dataset)

    decoded_thor = left_images + 0.5

    if show:
        plt.imshow(decoded_thor)
        plt.title('after')
        plt.show()

    assert np.squeeze(decoded_thor).shape == np.squeeze(image_arr).shape
    assert np.allclose(decoded_thor, image_arr, rtol=1.e-1, atol=1.e-1)
