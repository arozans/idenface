import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from data.tfrecord.conftest import _check_result, _check_paired_result
from src.data import preparing_data
from src.data.common_types import DictsDataset
from src.data.tfrecord.reading import reading_tfrecords
from src.utils import utils, consts
from testing_utils import tf_helpers, gen


@pytest.mark.parametrize(consts.BATCH_SIZE, [1, 5, 120])
@pytest.mark.parametrize('encoding', [False, True], ids=lambda x: "with encoding" if x else "no encoding", )
def test_should_save_and_read_pairs_correctly(batch_size, encoding):
    images_dataset: DictsDataset = gen.dicts_dataset(batch_size=batch_size, paired=True)

    tfrecord_full_path = preparing_data.save_to_tfrecord(images_dataset.features,
                                                         images_dataset.labels,
                                                         'data',
                                                         gen.dataset_spec(encoding=encoding))

    assert utils.check_filepath(tfrecord_full_path, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, gen.dataset_spec(encoding=encoding))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()
    _check_paired_result(first_batch, (images_dataset.features.left, images_dataset.features.right),
                         images_dataset.labels)


@pytest.mark.parametrize(consts.BATCH_SIZE, [1, 5, 120])
def test_should_save_and_read_unpaired_correctly(batch_size):
    images_dataset: DictsDataset = gen.dicts_dataset(batch_size=batch_size, paired=False)

    tfrecord_full_path = preparing_data.save_to_tfrecord(images_dataset.features, images_dataset.labels, 'data',
                                                         gen.dataset_spec(paired=False))

    assert utils.check_filepath(tfrecord_full_path, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, gen.dataset_spec(paired=False))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()
    _check_result(first_batch, images_dataset.features.all, images_dataset.labels)


@pytest.mark.parametrize('encoding', [False, True])
def test_should_save_image_correctly(thor_image_path, encoding):
    show = False

    if thor_image_path.endswith(".jpg"):
        from PIL import Image
        thor = Image.open(tf_helpers.get_string(thor_image_path))
    else:
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
    label_dict = gen.paired_labels_dict()

    tfrecord_full_path = preparing_data.save_to_tfrecord(two_images, label_dict, 'thor',
                                                         gen.dataset_spec(encoding=encoding))

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, gen.dataset_spec(encoding=encoding))

    left_images, _, _, _, _ = tf_helpers.unpack_first_batch(dataset)

    decoded_thor = left_images + 0.5

    if show:
        plt.imshow(decoded_thor)
        plt.title('after')
        plt.show()

    assert np.squeeze(decoded_thor).shape == np.squeeze(image_arr).shape
    assert np.allclose(decoded_thor, image_arr)
