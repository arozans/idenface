import numpy as np
import pytest
import tensorflow as tf
from hamcrest import only_contains, is_in
from hamcrest.core import assert_that, not_

from src.data.common_types import DatasetSpec, DatasetType, AbstractRawDataProvider
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordDatasetProvider, \
    FromGeneratorDatasetProvider, TFRecordTrainUnpairedDatasetProvider
from src.utils import consts
from src.utils.configuration import config
from testing_utils import tf_helpers
from testing_utils.testing_classes import CuratedFakeRawDataProvider, CuratedMnistFakeRawDataProvider
from testing_utils.tf_helpers import run_eagerly


class FakeDatasetProvider(AbstractDatasetProvider):
    def build_dataset(self, dataset_spec: DatasetSpec) -> tf.data.Dataset:
        pass


class FakeTrainUnpairedDatasetProvider(TFRecordTrainUnpairedDatasetProvider):
    def build_dataset(self, dataset_spec: DatasetSpec) -> tf.data.Dataset:
        pass


raw_data_provider_cls = AbstractRawDataProvider
default_provider = FakeDatasetProvider(raw_data_provider_cls)
train_unpaired_dataset_provider = FakeTrainUnpairedDatasetProvider(raw_data_provider_cls)


@pytest.fixture()
def patched_dataset_building(mocker):
    return mocker.patch.object(default_provider, 'build_dataset', autospec=True)


def test_train_input_fn_should_search_for_dataset_with_correct_spec(patched_dataset_building):
    default_provider.train_input_fn()
    patched_dataset_building.assert_called_once_with(DatasetSpec(raw_data_provider_cls,
                                                                 DatasetType.TRAIN,
                                                                 with_excludes=False))


def test_eval_input_fn_not_ignoring_excludes_should_search_for_dataset_with_correct_spec(patched_dataset_building):
    default_provider.eval_input_fn()
    patched_dataset_building.assert_called_once_with(DatasetSpec(raw_data_provider_cls,
                                                                 DatasetType.TEST,
                                                                 with_excludes=False))


def test_eval_with_excludes_input_fn_should_search_for_dataset_with_correct_spec(patched_dataset_building):
    default_provider.eval_with_excludes_input_fn()
    patched_dataset_building.assert_called_once_with(DatasetSpec(raw_data_provider_cls,
                                                                 DatasetType.TEST,
                                                                 with_excludes=True))


@pytest.fixture()
def patched_dataset_supplying(mocker):
    return mocker.patch.object(default_provider, 'supply_dataset', autospec=True)


@pytest.mark.parametrize('dataset_provider, paired',
                         [
                             (default_provider, True),
                             (train_unpaired_dataset_provider, False)
                         ])
def test_train_input_fn_should_correct_configure_dataset(mocker, dataset_provider, paired):
    patched_dataset_supplying = mocker.patch.object(dataset_provider, 'supply_dataset', autospec=True)

    dataset_provider.train_input_fn()

    patched_dataset_supplying.assert_called_once_with(
        DatasetSpec(
            raw_data_provider_cls,
            DatasetType.TRAIN,
            with_excludes=False,
            encoding=True,
            repeating_pairs=True,
            identical_pairs=False,
            paired=paired
        ),
        shuffle_buffer_size=config[consts.SHUFFLE_BUFFER_SIZE],
        batch_size=config[consts.BATCH_SIZE],
        repeat=True
    )


@pytest.mark.parametrize('dataset_provider',
                         [
                             default_provider,
                             train_unpaired_dataset_provider
                         ])
def test_test_eval_input_fn_should_correct_configure_dataset(mocker, dataset_provider):
    patched_dataset_supplying = mocker.patch.object(dataset_provider, 'supply_dataset', autospec=True)

    dataset_provider.eval_input_fn()
    patched_dataset_supplying.assert_called_once_with(
        DatasetSpec(
            raw_data_provider_cls,
            DatasetType.TEST,
            with_excludes=False,
            encoding=True,
            repeating_pairs=True,
            identical_pairs=False,
            paired=True
        ),
        batch_size=config[consts.BATCH_SIZE]
    )


@pytest.mark.parametrize('dataset_provider',
                         [
                             default_provider,
                             train_unpaired_dataset_provider
                         ])
def test_test_eval_with_excludes_input_fn_should_correct_configure_dataset(mocker, dataset_provider):
    patched_dataset_supplying = mocker.patch.object(dataset_provider, 'supply_dataset', autospec=True)

    dataset_provider.eval_with_excludes_input_fn()
    patched_dataset_supplying.assert_called_once_with(
        DatasetSpec(
            raw_data_provider_cls,
            DatasetType.TEST,
            with_excludes=True,
            encoding=True,
            repeating_pairs=True,
            identical_pairs=False,
            paired=True
        ),
        batch_size=config[consts.BATCH_SIZE]
    )


def get_class_name(clazz):
    return clazz.__name__


@pytest.mark.parametrize('raw_data_provider_cls',
                         [CuratedFakeRawDataProvider,
                          CuratedMnistFakeRawDataProvider])
@pytest.mark.parametrize('dataset_provider_cls_name',
                         [
                             TFRecordDatasetProvider,
                             FromGeneratorDatasetProvider,
                             TFRecordTrainUnpairedDatasetProvider
                         ])
def test_all_paired_dataset_providers_should_provide_raw_data_dimensions(raw_data_provider_cls,
                                                                         dataset_provider_cls_name):
    provider = dataset_provider_cls_name(raw_data_provider_cls)

    side_len = provider.raw_data_provider_cls.description().image_dimensions.width
    # dataset = provider.train_input_fn()
    batch_size = 12
    dataset_spec = DatasetSpec(raw_data_provider_cls, DatasetType.TEST, with_excludes=False, encoding=False)
    dataset = provider.supply_dataset(dataset_spec, batch_size=batch_size).take(100)
    left, right, same_labels, left_labels, right_labels = tf_helpers.unpack_first_batch(dataset)

    assert left.shape == (batch_size, side_len, side_len, 1)
    assert right.shape == (batch_size, side_len, side_len, 1)

    assert same_labels.shape == left_labels.shape == right_labels.shape == (batch_size,)


@pytest.mark.parametrize('raw_data_provider_cls',
                         [CuratedFakeRawDataProvider,
                          CuratedMnistFakeRawDataProvider])
@pytest.mark.parametrize('dataset_provider_cls_name',
                         [
                             TFRecordTrainUnpairedDatasetProvider
                         ])
def test_all_unpaired_dataset_providers_should_provide_raw_data_dimensions(raw_data_provider_cls,
                                                                           dataset_provider_cls_name):
    provider = dataset_provider_cls_name(raw_data_provider_cls)

    side_len = provider.raw_data_provider_cls.description().image_dimensions.width
    # dataset = provider.train_input_fn()
    batch_size = 12
    dataset_spec = DatasetSpec(raw_data_provider_cls, DatasetType.TRAIN, with_excludes=False, encoding=False,
                               paired=False)
    dataset = provider.supply_dataset(dataset_spec, batch_size=batch_size).take(100)
    images, labels = tf_helpers.unpack_first_batch(dataset)

    assert images.shape == (batch_size, side_len, side_len, 1)

    assert labels.shape == (batch_size,)


def from_class_name(name: str):
    if tf.executing_eagerly():
        return eval(name.numpy())
    else:
        return eval(name)


@pytest.mark.parametrize('dataset_provider_cls_name',
                         [
                             get_class_name(FromGeneratorDatasetProvider),
                             get_class_name(TFRecordDatasetProvider),
                             get_class_name(TFRecordTrainUnpairedDatasetProvider)
                         ])
@run_eagerly
def test_all_paired_dataset_providers_should_provide_correct_labels(dataset_provider_cls_name):
    provider_cls = from_class_name(dataset_provider_cls_name)
    raw_data_provider_cls = CuratedFakeRawDataProvider
    dataset_spec = DatasetSpec(raw_data_provider_cls, DatasetType.TEST, with_excludes=False, encoding=False)
    provider = provider_cls(raw_data_provider_cls)

    dataset = provider.supply_dataset(dataset_spec, batch_size=1).take(100)
    for batch in dataset:
        left_img = np.rint((batch[0][consts.LEFT_FEATURE_IMAGE].numpy().flatten()[0] + 0.5) * 10)
        right_img = np.rint((batch[0][consts.RIGHT_FEATURE_IMAGE].numpy().flatten()[0] + 0.5) * 10)
        pair_label = batch[1][consts.TFRECORD_PAIR_LABEL].numpy().flatten()[0]
        left_label = batch[1][consts.TFRECORD_LEFT_LABEL].numpy().flatten()[0]
        right_label = batch[1][consts.TFRECORD_RIGHT_LABEL].numpy().flatten()[0]

        assert pair_label == (1 if left_img == right_img else 0)
        assert pair_label == (1 if left_label == right_label else 0)


@pytest.mark.parametrize('dataset_provider_cls_name',
                         [
                             get_class_name(TFRecordTrainUnpairedDatasetProvider)
                         ])
@run_eagerly
def test_all_unpaired_dataset_providers_should_provide_correct_labels(dataset_provider_cls_name):
    provider_cls = from_class_name(dataset_provider_cls_name)
    raw_data_provider_cls = CuratedFakeRawDataProvider
    dataset_spec = DatasetSpec(raw_data_provider_cls, DatasetType.TRAIN, with_excludes=False, encoding=False,
                               paired=False)
    provider = provider_cls(raw_data_provider_cls)

    dataset = provider.supply_dataset(dataset_spec, batch_size=1).take(100)
    for batch in dataset:
        images = np.rint((batch[0][consts.FEATURES].numpy().flatten()[0] + 0.5) * 10)
        labels = batch[1][consts.LABELS].numpy().flatten()[0]

        assert images == labels


@pytest.mark.parametrize('dataset_provider_cls_name',
                         [
                             get_class_name(FromGeneratorDatasetProvider),
                             get_class_name(TFRecordDatasetProvider),
                             get_class_name(TFRecordTrainUnpairedDatasetProvider)
                         ])
@pytest.mark.parametrize('patched_excluded', [[2, 3]], indirect=True)
@run_eagerly
def test_all_paired_dataset_providers_should_honor_excludes(dataset_provider_cls_name, patched_excluded):
    provider_cls = from_class_name(dataset_provider_cls_name)
    raw_data_provider_cls = CuratedFakeRawDataProvider
    dataset_spec = DatasetSpec(raw_data_provider_cls, DatasetType.TEST, with_excludes=False, encoding=False)
    provider = provider_cls(raw_data_provider_cls)

    dataset = provider.supply_dataset(dataset_spec, batch_size=1).take(100)
    encountered_labels = set()
    for batch in dataset:
        left_label = batch[0][consts.LEFT_FEATURE_IMAGE].numpy().flatten()[0] + 0.5
        right_label = batch[0][consts.RIGHT_FEATURE_IMAGE].numpy().flatten()[0] + 0.5
        encountered_labels.update((left_label, right_label))
    assert_that((np.rint(list(encountered_labels)) * 10),
                only_contains(not_(is_in(list(patched_excluded.numpy())))))
    assert_that((np.rint(list(encountered_labels)) * 10), only_contains((is_in([0, 1, 4]))))


@pytest.mark.parametrize('dataset_provider_cls_name',
                         [
                             get_class_name(TFRecordTrainUnpairedDatasetProvider)
                         ])
@pytest.mark.parametrize('patched_excluded', [[2, 3]], indirect=True)
@run_eagerly
def test_all_unpaired_dataset_providers_should_honor_excludes(dataset_provider_cls_name, patched_excluded):
    provider_cls = from_class_name(dataset_provider_cls_name)
    raw_data_provider_cls = CuratedFakeRawDataProvider
    dataset_spec = DatasetSpec(raw_data_provider_cls, DatasetType.TRAIN, with_excludes=False, encoding=False,
                               paired=False)

    provider = provider_cls(raw_data_provider_cls)

    dataset = provider.supply_dataset(dataset_spec, batch_size=1).take(100)
    encountered_labels = set()
    for batch in dataset:
        image = np.rint((batch[0][consts.FEATURES].numpy().flatten()[0] + 0.5) * 10)
        label = batch[1][consts.LABELS].numpy().flatten()[0]

        encountered_labels.update((image,))
        encountered_labels.update((label,))
    assert_that((np.rint(list(encountered_labels))),
                only_contains(not_(is_in(list(patched_excluded.numpy())))))
    assert_that((np.rint(list(encountered_labels))), only_contains((is_in([0, 1, 4]))))
