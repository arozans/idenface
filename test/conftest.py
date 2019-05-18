import copy
import functools
import shutil
from pathlib import Path
from typing import Any

import pytest
import tensorflow as tf
from dataclasses import dataclass, replace
from mock import PropertyMock

from src.data.common_types import DataDescription, AbstractRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel
from src.utils import consts, configuration
from src.utils.configuration import config
from testing_utils import testing_consts, testing_helpers
from testing_utils.testing_classes import FakeRawDataProvider


def pytest_runtest_setup(item):
    if 'integration' in item.keywords and not item.config.getoption("--integration"):
        pytest.skip("need --integration option to run this test")


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true",
                     help="run the tests only in case of that command line (marked with marker @no_cmsd)")


@pytest.fixture
def create_test_tmp_dir():
    directory = Path('/tmp/foo/bar')
    if not testing_consts.DEBUG_MODE and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=testing_consts.DEBUG_MODE)

    yield directory

    if not testing_consts.DEBUG_MODE:
        shutil.rmtree(directory)


@pytest.fixture(autouse=True)
def patched_home_dir(mocker, create_test_tmp_dir):
    mocker.patch('src.utils.filenames._get_home_directory', return_value=create_test_tmp_dir)
    return create_test_tmp_dir


@pytest.fixture()
def patched_params(request, patched_configuration):
    config_dict = request.param
    yield from _patch(config_dict, expected_type=dict)


def _patch(element_to_patch, key=None, expected_type=None):
    if expected_type is not None:
        assert isinstance(element_to_patch, expected_type)
    previous = copy.deepcopy(config.tf_flags)

    if key is not None:
        config.tf_flags.update({key: element_to_patch})
    else:
        config.tf_flags.update(element_to_patch)

    config._rebuild_full_config()
    yield element_to_patch

    config.tf_flags = previous
    config._rebuild_full_config()


@pytest.fixture()
def patched_excluded(request, patched_configuration):
    excluded = request.param
    yield from _patch(excluded, expected_type=list, key=consts.EXCLUDED_KEYS)


@pytest.fixture()
def patched_global_suffix(request, patched_configuration):
    excluded = request.param
    yield from _patch(excluded, key=consts.GLOBAL_SUFFIX)


@pytest.fixture(autouse=True, scope="session")
def define_cli_args():
    configuration.define_cli_args()


@pytest.fixture(autouse=True)
def patched_configuration():
    test_conf = {
        consts.BATCH_SIZE: testing_consts.TEST_BATCH_SIZE,
        consts.TRAIN_STEPS: testing_consts.TEST_TRAIN_STEPS,
        consts.EVAL_STEPS_INTERVAL: testing_consts.TEST_EVAL_STEPS_INTERVAL,
        # consts.EXCLUDED_KEYS: testing_consts.TEST_EXCLUDED_KEYS,
        consts.IS_INFER_CHECKPOINT_OBLIGATORY: True
    }
    import sys
    sys.argv = sys.argv[0:3]
    for key, value in test_conf.items():
        sys.argv.append('--' + str(key) + '=' + str(value))

    config.tf_flags.update(test_conf)
    config._rebuild_full_config()


def fake_raw_images(image_side_length: int):
    return testing_helpers.generate_fake_images(
        (testing_consts.FAKE_IMAGES_IN_DATASET_COUNT, image_side_length,
         image_side_length))


def fake_labels_dict(class_count: int):
    pair_labels = testing_helpers.generate_fake_labels(size=testing_consts.FAKE_IMAGES_IN_DATASET_COUNT, classes=2)
    left_labels = testing_helpers.generate_fake_labels(size=testing_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                                                       classes=class_count)
    right_labels = testing_helpers.generate_fake_labels(size=testing_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                                                        classes=class_count)
    return {
        consts.PAIR_LABEL: pair_labels,
        consts.LEFT_FEATURE_LABEL: left_labels,
        consts.RIGHT_FEATURE_LABEL: right_labels
    }


def fake_images_dict(image_side_length: int):
    def inner():
        return fake_raw_images(image_side_length).reshape(-1, image_side_length,
                                                          image_side_length, 1)

    return {
        consts.LEFT_FEATURE_IMAGE: inner(),
        consts.RIGHT_FEATURE_IMAGE: inner()
    }


def create_fake_dict_and_labels(image_side_length, classes_count):
    return fake_images_dict(image_side_length), fake_labels_dict(classes_count)


@pytest.fixture()
def fake_dict_and_labels(request):
    image_side_length, classes_count = extract_or_default(request)
    print("Creating fake dict and labels of features of side length: {} and classes num: {}".format(image_side_length,
                                                                                                    classes_count))
    return create_fake_dict_and_labels(image_side_length,
                                       classes_count)


def extract_or_default(request):
    try:
        param = request.param
    except AttributeError:
        image_side_length: int = testing_consts.FAKE_IMAGE_SIDE_PIXEL_COUNT
    else:
        if type(param) is tuple:
            param = param[0]
        if issubclass(param, EstimatorModel):
            description: DataDescription = param().raw_data_provider.description
        elif issubclass(param, AbstractRawDataProvider):
            description: DataDescription = param().description
        else:
            description: DataDescription = param
        image_side_length = description.image_dimensions.width
    classes_count = 10

    return image_side_length, classes_count


@dataclass
class FixtureResultWithParam:
    result: Any
    param: Any


def returns_param(a_fixture):
    @functools.wraps(a_fixture)
    def inner(*args, **kwargs):
        try:
            param = kwargs['request'].param
        except AttributeError:
            param = None
        return FixtureResultWithParam(a_fixture(*args, **kwargs), param)

    return inner


@returns_param
@pytest.fixture()
def patched_read_dataset(mocker, request):
    image_side_length, classes_count = extract_or_default(request)
    print("Preparing to mock reading data with features of side: {} and classes num: {}".format(image_side_length,
                                                                                                classes_count))

    def preparing_dataset(*args, **kwargs):
        return tf.data.Dataset.from_tensor_slices(create_fake_dict_and_labels(image_side_length, classes_count))

    # TODO: monkeypatch `build_dataset` method of the data_provider/model passed as an input
    return mocker.patch('src.estimator.training.supplying_datasets.TFRecordDatasetProvider.build_dataset',
                        side_effect=preparing_dataset)


@pytest.fixture()
def injected_raw_data_provider(mocker, request):
    model: EstimatorModel = request.param
    assert issubclass(model, EstimatorModel)
    print("Preparing to mock raw data provider of model ", model)
    desc = model().raw_data_provider.description
    reduced_image_dims = replace(desc.image_dimensions,
                                 width=min(desc.image_dimensions.width, consts.MNIST_IMAGE_SIDE_PIXEL_COUNT),
                                 height=min(desc.image_dimensions.height, consts.MNIST_IMAGE_SIDE_PIXEL_COUNT),
                                 )
    desc = replace(desc,
                   image_dimensions=reduced_image_dims,
                   classes_count=min(desc.classes_count, consts.MNIST_IMAGE_CLASSES_COUNT))

    mocker.patch.object(model, 'raw_data_provider', new_callable=PropertyMock,
                        return_value=FakeRawDataProvider(description=desc, curated=True))
    return model
