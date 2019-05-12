from pathlib import Path

import numpy as np
import pytest
from hamcrest import only_contains, is_in
from hamcrest.core import assert_that

from src.data.common_types import DatasetVariant, DatasetFragment
from src.data.raw_data.raw_data_providers import ExtruderRawDataProvider
from src.utils import filenames, consts


@pytest.fixture()
def prepare_data():
    directory: Path = filenames.get_raw_input_data_dir() / DatasetVariant.EXTRUDER.name.lower()
    labels = range(100)
    images_per_label = 5
    paths = []

    for label in labels:
        for image in range(images_per_label):
            path = Path(directory) / ("000" + str(label)) / str(image)
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.with_suffix(consts.LOG)
            path.write_text("a text")
            paths.append(path)
    return DatasetFragment(features=np.array(paths), labels=np.array(list(labels)))


provider = ExtruderRawDataProvider()


def test_should_get_train_features(prepare_data):
    images, labels = provider.get_raw_train()
    assert len(images) == len(labels) == 90 * 5
    assert_that(set(labels), only_contains(is_in(list(map(lambda x: x, range(0, 90))))))
    for i, l in zip(images, labels):
        assert str(l) in str(i)


def test_should_get_test_features(prepare_data):
    images, labels = provider.get_raw_test()
    assert len(images) == len(labels) == 10 * 5
    assert_that(set(labels), only_contains(is_in(list(map(lambda x: x, range(90, 100))))))
    for i, l in zip(images, labels):
        assert str(l) in str(i)


def test_should_get_sample_feature(prepare_data):
    sample_feature = provider.get_sample_feature()

    assert isinstance(sample_feature, Path)
    assert sample_feature in prepare_data.features
