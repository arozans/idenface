import pytest
import tensorflow as tf
from hamcrest import only_contains, is_in
from hamcrest.core import assert_that, not_

from helpers.test_helpers import CuratedMnistFakeRawDataProvider, CuratedFakeRawDataProvider
from helpers.tf_helpers import run_eagerly
from src.data.common_types import DatasetSpec, DatasetType, AbstractRawDataProvider
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordDatasetProvider, \
    FromGeneratorDatasetProvider
from src.utils import consts
from src.utils.configuration import config


class FakeDatasetProvider(AbstractDatasetProvider):
    def build_dataset(self, dataset_spec: DatasetSpec) -> tf.data.Dataset:
        pass


raw_data_provider_cls = AbstractRawDataProvider
provider = FakeDatasetProvider(raw_data_provider_cls)


@pytest.fixture()
def patched_dataset_building(mocker):
    return mocker.patch.object(provider, 'build_dataset', autospec=True)


def test_train_input_fn_should_search_for_dataset_with_correct_spec(patched_dataset_building):
    provider.train_input_fn()
    patched_dataset_building.assert_called_once_with(DatasetSpec(raw_data_provider_cls,
                                                                 DatasetType.TRAIN,
                                                                 with_excludes=False))


def test_eval_input_fn_not_ignoring_excludes_should_search_for_dataset_with_correct_spec(patched_dataset_building):
    provider.eval_input_fn()
    patched_dataset_building.assert_called_once_with(DatasetSpec(raw_data_provider_cls,
                                                                 DatasetType.TEST,
                                                                 with_excludes=False))


def test_eval_with_excludes_input_fn_should_search_for_dataset_with_correct_spec(patched_dataset_building):
    provider.eval_with_excludes_fn()
    patched_dataset_building.assert_called_once_with(DatasetSpec(raw_data_provider_cls,
                                                                 DatasetType.TEST,
                                                                 with_excludes=True))


def get_class_name(clazz):
    return clazz.__name__


def from_class_name(name: str):
    return eval(name.numpy())


@pytest.mark.parametrize('raw_data_provider_cls',
                         [get_class_name(CuratedFakeRawDataProvider),
                          get_class_name(CuratedMnistFakeRawDataProvider)])
@pytest.mark.parametrize('dataset_provider_cls_name',
                         [get_class_name(TFRecordDatasetProvider),
                          get_class_name(FromGeneratorDatasetProvider)])
@run_eagerly
def test_all_dataset_providers_should_provide_raw_data_dimensions(dataset_provider_cls_name, raw_data_provider_cls):
    provider_cls = from_class_name(dataset_provider_cls_name)
    raw_data_provider_cls = from_class_name(raw_data_provider_cls)
    provider = provider_cls(raw_data_provider_cls)

    side_len = provider.raw_data_provider_cls.description().image_side_length
    dataset = provider.train_input_fn()
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()

    left = first_batch[0][consts.LEFT_FEATURE_IMAGE].numpy()
    right = first_batch[0][consts.RIGHT_FEATURE_IMAGE].numpy()
    assert left.shape == (config.batch_size, side_len, side_len, 1)
    assert right.shape == (config.batch_size, side_len, side_len, 1)

    label = first_batch[1].numpy()
    assert label.shape == (config.batch_size,)


@pytest.mark.parametrize('dataset_provider_cls_name',
                         [get_class_name(FromGeneratorDatasetProvider),
                          get_class_name(TFRecordDatasetProvider)])
@pytest.mark.parametrize('patched_excluded', [([2, 3])], indirect=True)
@run_eagerly
def test_all_dataset_providers_should_honor_excludes(dataset_provider_cls_name, patched_excluded):
    provider_cls = from_class_name(dataset_provider_cls_name)
    raw_data_provider_cls = CuratedFakeRawDataProvider
    dataset_spec = DatasetSpec(raw_data_provider_cls, DatasetType.TRAIN, with_excludes=False)
    provider = provider_cls(raw_data_provider_cls)

    dataset = provider.supply_dataset(dataset_spec, batch_size=1).take(100)
    encountered_labels = set()
    for batch in dataset:
        print(batch)
        left_label = batch[0][consts.LEFT_FEATURE_IMAGE].numpy().flatten()[0] + (
            0.5 if provider_cls == TFRecordDatasetProvider else 0)
        right_label = batch[0][consts.RIGHT_FEATURE_IMAGE].numpy().flatten()[0] + (
            0.5 if provider_cls == TFRecordDatasetProvider else 0)
        encountered_labels.update((left_label, right_label))
    print("hmm", list(encountered_labels))
    assert_that(encountered_labels, only_contains(not_(is_in(list(patched_excluded.numpy())))))
    assert_that(encountered_labels, only_contains((is_in([0, 1, 4]))))
