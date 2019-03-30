import functools
import shutil
from pathlib import Path
from typing import Any

import pytest
import tensorflow as tf
from absl.flags import UnrecognizedFlagError
from dataclasses import dataclass

from helpers import test_consts, test_helpers
from helpers.test_helpers import CuratedFakeRawDataProvider
from src.data.common_types import DataDescription, AbstractRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel
from src.utils import consts
from src.utils.configuration import Config


def pytest_runtest_setup(item):
    if 'integration' in item.keywords and not item.config.getoption("--integration"):
        pytest.skip("need --integration option to run this test")


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true",
                     help="run the tests only in case of that command line (marked with marker @no_cmsd)")


@pytest.fixture
def create_test_tmp_dir():
    directory = Path('/tmp/foo/bar')
    if not test_consts.DEBUG_MODE and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=test_consts.DEBUG_MODE)

    yield directory

    if not test_consts.DEBUG_MODE:
        shutil.rmtree(directory)


@pytest.fixture(autouse=True)
def patched_home_dir(mocker, create_test_tmp_dir):
    mocker.patch('src.utils.filenames._get_home_directory', return_value=create_test_tmp_dir)
    return create_test_tmp_dir


@pytest.fixture()
def patched_excluded(request, mocker):
    try:
        excluded = request.param
    except AttributeError:
        excluded = test_consts.TEST_EXCLUDED_KEYS
    mocker.patch('src.utils.configuration._excluded_keys', excluded)
    return excluded


@pytest.fixture(autouse=True)
def patched_configuration(mocker, patched_excluded):
    test_conf = {
        "batch_size": test_consts.TEST_BATCH_SIZE,
        "train_steps": test_consts.TEST_TRAIN_STEPS,
        "eval_steps_interval": test_consts.TEST_EVAL_STEPS_INTERVAL,
    }

    class TestConfig(Config):

        @staticmethod
        def try_search_commandline_flags(flag_name):
            try:
                return test_conf[flag_name]
            except KeyError:
                raise UnrecognizedFlagError(flag_name)

    mocker.patch('src.estimator.training.training.config', TestConfig())


def fake_raw_images(image_side_length: int):
    return test_helpers.generate_fake_images(
        (test_consts.FAKE_IMAGES_IN_DATASET_COUNT, image_side_length,
         image_side_length))


def fake_labels(class_count: int):
    labels = test_helpers.generate_fake_labels(size=test_consts.FAKE_IMAGES_IN_DATASET_COUNT, classes=class_count)
    return labels


def fake_dict(image_side_length: int):
    def inner():
        return fake_raw_images(image_side_length).reshape(-1, image_side_length,
                                                          image_side_length, 1)

    return {
        consts.LEFT_FEATURE_IMAGE: inner(),
        consts.RIGHT_FEATURE_IMAGE: inner()
    }


def create_fake_dict_and_labels(image_side_length, classes_count):
    return fake_dict(image_side_length), fake_labels(classes_count)


@pytest.fixture()
def fake_dict_and_labels(request):
    image_side_length, classes_count = extract_or_default(request)
    print("Creating fake dict and labels of images of side length: {} and classes num: {}".format(image_side_length,
                                                                                                  classes_count))
    return create_fake_dict_and_labels(image_side_length,
                                       classes_count)


def extract_or_default(request):
    try:
        param = request.param
    except AttributeError:
        image_side_length: int = test_consts.FAKE_IMAGE_SIDE_PIXEL_COUNT
    else:
        if issubclass(param, EstimatorModel):
            description: DataDescription = param().raw_data_provider_cls.description()
        elif issubclass(param, AbstractRawDataProvider):
            description: DataDescription = param.description()
        else:
            description: DataDescription = param
        image_side_length = description.image_side_length
    classes_count = 2

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
    print("Preparing to mock reading data with images of side: {} and classes num: {}".format(image_side_length,
                                                                                              classes_count))

    def preparing_dataset(*args, **kwargs):
        return tf.data.Dataset.from_tensor_slices(create_fake_dict_and_labels(image_side_length, classes_count))

    # TODO: monkeypatch `build_dataset` method of the data_provider/model passed as an input
    return mocker.patch('src.estimator.training.supplying_datasets.TFRecordDatasetProvider.build_dataset',
                        side_effect=preparing_dataset)


@pytest.fixture()
def injected_raw_data_provider(request):
    model: EstimatorModel = request.param
    assert issubclass(model, EstimatorModel)
    print("Preparing to mock raw data provider of model ", model)
    tmp_cls = model().raw_data_provider_cls
    desc = tmp_cls().description()

    class TempRawDataProvider(CuratedFakeRawDataProvider):

        @staticmethod
        def description() -> DataDescription:
            return desc

    # patched_raw_data_provider = CuratedFakeRawDataProvider(description=desc)

    model.get_raw_dataset_provider_cls = lambda x: TempRawDataProvider
    return model
