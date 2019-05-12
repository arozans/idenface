import numpy as np
import pytest

from src.utils import utils, consts
from testing_utils import tf_helpers

tol = 1.e-2


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


def _check_paired_result(first_batch, expected_images_values, labels):
    left_images, right_images, pair_labels, left_labels, right_labels = tf_helpers.unpack_batch(first_batch)
    assert len(left_images) == len(right_images) == len(pair_labels) == len(left_labels) == len(
        right_labels) == len(expected_images_values[0])
    for left_image, left_expected in zip(left_images, expected_images_values[0]):
        assert np.allclose(left_image + 0.5, left_expected, rtol=tol, atol=tol)
    for right_image, right_expected in zip(right_images, expected_images_values[1]):
        assert np.allclose(right_image + 0.5, right_expected, rtol=tol, atol=tol)
    assert (pair_labels == labels[consts.PAIR_LABEL]).all()
    assert (left_labels == labels[consts.LEFT_FEATURE_LABEL]).all()
    assert (right_labels == labels[consts.RIGHT_FEATURE_LABEL]).all()


def _check_result(first_batch, expected_images_values, labels):
    images, unpack_labels = tf_helpers.unpack_batch(first_batch)
    assert len(images) == len(unpack_labels) == len(expected_images_values)
    for image, expected in zip(images, expected_images_values):
        assert np.allclose(image + 0.5, expected, rtol=tol, atol=tol)
    assert (unpack_labels == labels[consts.LABELS]).all()
