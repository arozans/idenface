import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
import skimage
from hamcrest import contains, not_
from hamcrest.core import assert_that

from data.tfrecord.conftest import _check_paired_result, _check_result
from src.data import preparing_data
from src.data.common_types import DatasetStorageMethod, ImageDimensions, DictsDataset, RawDatasetFragment
from src.data.tfrecord.reading import reading_tfrecords
from src.utils import utils, consts
from testing_utils import tf_helpers, gen, testing_helpers, testing_consts


@pytest.mark.parametrize(consts.BATCH_SIZE, [1, 3, 120])
def test_should_save_and_read_pairs_correctly(batch_size):
    images_dataset: DictsDataset
    paths_dataset: DictsDataset
    images_dataset, paths_dataset = gen.images(batch_size=batch_size, paired=True, save_on_disc=True)
    raw_dataset_fragment = testing_helpers.dicts_dataset_to_raw_dataset_fragment(images_dataset)

    dataset_desc = gen.dataset_desc(
        storage_method=DatasetStorageMethod.ON_DISC,
        image_dimensions=ImageDimensions.from_tuple(testing_consts.TEST_IMAGE_SIZE)
    )
    dataset_spec = gen.dataset_spec(
        description=dataset_desc,
        raw_dataset_fragment=raw_dataset_fragment
    )

    tfrecord_full_path = preparing_data.save_to_tfrecord(paths_dataset.features, paths_dataset.labels,
                                                         'data',
                                                         dataset_spec)

    assert utils.check_filepath(tfrecord_full_path, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, dataset_spec)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    first_batch = dataset.make_one_shot_iterator().get_next()
    _check_paired_result(first_batch, (images_dataset.features.left, images_dataset.features.right),
                         images_dataset.labels)


@pytest.mark.parametrize(consts.BATCH_SIZE, [1, 5, 120])
def test_should_save_and_read_unpaired_correctly(batch_size):
    images_dataset: DictsDataset
    paths_dataset: DictsDataset
    images_dataset, paths_dataset = gen.images(batch_size=batch_size, save_on_disc=True)

    dataset_desc = gen.dataset_desc(storage_method=DatasetStorageMethod.ON_DISC,
                                    image_dimensions=ImageDimensions.from_tuple(testing_consts.TEST_IMAGE_SIZE)
                                    )
    dataset_spec = gen.dataset_spec(description=dataset_desc, paired=False)
    tfrecord_full_path = preparing_data.save_to_tfrecord(paths_dataset.features, paths_dataset.labels,
                                                         'data', dataset_spec)

    assert utils.check_filepath(tfrecord_full_path, is_directory=False, is_empty=False)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, dataset_spec)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    first_batch = dataset.make_one_shot_iterator().get_next()
    _check_result(first_batch, images_dataset.features.all, images_dataset.labels)


@pytest.mark.parametrize(
    'expected_size, should_image_size_be_reduced',
    [(testing_consts.TEST_IMAGE_SIZE, False), ((2, 2, 3), True)],
    ids=lambda x: "reduced_size" if x[1] else "not_reduced_size"
)
def test_should_include_reduced_size_in_path(expected_size, should_image_size_be_reduced):
    images_dataset: DictsDataset
    paths_dataset: DictsDataset
    images_dataset, paths_dataset = gen.images(save_on_disc=True)

    dataset_desc = gen.dataset_desc(storage_method=DatasetStorageMethod.ON_DISC,
                                    image_dimensions=ImageDimensions.from_tuple(expected_size)
                                    )
    raw_dataset_fragment = testing_helpers.dicts_dataset_to_raw_dataset_fragment(images_dataset)
    dataset_spec = gen.dataset_spec(description=dataset_desc,
                                    raw_dataset_fragment=raw_dataset_fragment,
                                    paired=False)
    tfrecord_full_path = preparing_data.save_to_tfrecord(paths_dataset.features, paths_dataset.labels,
                                                         'data', dataset_spec)

    parts = tfrecord_full_path.parts
    if should_image_size_be_reduced:
        assert ("size_" + str(expected_size[0])) in parts
    else:
        assert_that(parts, not_(contains("size_" + str(expected_size[0]))))


@pytest.mark.parametrize('resizing', [False, True])
def test_should_read_and_save_image_correctly(thor_image_path, resizing):
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

    if resizing:
        shape = (100, 100, 3)
    else:
        shape = thor.shape
    dataset_desc = gen.dataset_desc(storage_method=DatasetStorageMethod.ON_DISC,
                                    image_dimensions=ImageDimensions.from_tuple(shape))
    raw_dataset_fragment = RawDatasetFragment(features=image_arr, labels=np.array(list(labels.values())))
    dataset_spec = gen.dataset_spec(description=dataset_desc, raw_dataset_fragment=raw_dataset_fragment, paired=False)

    tfrecord_full_path = preparing_data.save_to_tfrecord(features_as_paths, labels, 'thor', dataset_spec)

    dataset = reading_tfrecords.assemble_dataset(tfrecord_full_path.parent, dataset_spec)

    left_images, _ = tf_helpers.unpack_first_batch(dataset)

    decoded_thor = left_images + 0.5

    if show:
        plt.imshow(decoded_thor)
        plt.title('after')
        plt.show()

    assert np.squeeze(decoded_thor).shape == shape
    if not resizing:
        assert np.allclose(decoded_thor, image_arr, rtol=1.e-1, atol=1.e-1)
