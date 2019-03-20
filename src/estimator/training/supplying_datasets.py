from pathlib import Path
from typing import Optional, Type

import tensorflow as tf
from tensorflow.python.data import TFRecordDataset

from src.data import preparing_data
from src.data.common_types import DatasetSpec, DatasetType, AbstractRawDataProvider
from src.data.saving import reading_tfrecords
from src.utils import utils
from src.utils.configuration import config


def train_input_fn(raw_data_provider_cls: Type[AbstractRawDataProvider]) -> TFRecordDataset:
    utils.log('Calling train_input_fn')
    train_data_config = DatasetSpec(raw_data_provider_cls=raw_data_provider_cls, type=DatasetType.TRAIN,
                                    with_excludes=False)
    return supply_dataset(dataset_spec=train_data_config, shuffle_buffer_size=config.shuffle_buffer_size,
                          batch_size=config.batch_size, repeat=True)


def eval_input_fn(raw_data_provider_cls: Type[AbstractRawDataProvider]) -> TFRecordDataset:
    utils.log('Calling eval_input_fn')
    test_data_config = DatasetSpec(raw_data_provider_cls=raw_data_provider_cls, type=DatasetType.TEST,
                                   with_excludes=False)
    return supply_dataset(dataset_spec=test_data_config, batch_size=config.batch_size)


def eval_with_excludes_fn(raw_data_provider_cls: Type[AbstractRawDataProvider]) -> TFRecordDataset:
    utils.log('Calling eval_input_fn with excluded elements')
    test_ignoring_excludes = DatasetSpec(raw_data_provider_cls=raw_data_provider_cls, type=DatasetType.TEST,
                                         with_excludes=True)
    return supply_dataset(dataset_spec=test_ignoring_excludes, batch_size=config.batch_size)


def infer(raw_data_provider_cls: Type[AbstractRawDataProvider], take_num: int) -> TFRecordDataset:
    utils.log('Calling infer_fn')
    test_ignoring_excludes = DatasetSpec(raw_data_provider_cls=raw_data_provider_cls,
                                         type=DatasetType.TEST,
                                         with_excludes=True)
    return supply_dataset(dataset_spec=test_ignoring_excludes,
                          batch_size=take_num,
                          repeat=False,
                          shuffle_buffer_size=config.shuffle_buffer_size,
                          prefetch=False,
                          take_num=take_num)


def predict_input_fn(features, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(features)

    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset


def supply_dataset(dataset_spec: DatasetSpec, shuffle_buffer_size: Optional[int] = None,
                   batch_size: Optional[int] = None, repeat: bool = False, prefetch: bool = True,
                   take_num=None) -> TFRecordDataset:
    dataset = _read_dataset(dataset_spec)
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if take_num is not None:
        dataset = dataset.take(take_num)
    if repeat:
        dataset = dataset.repeat()
    if batch_size:
        dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(1)

    return dataset


def _read_dataset(dataset_spec: DatasetSpec):
    data_dir: Path = preparing_data.find_or_create_paired_data_dir(dataset_spec)
    dataset: TFRecordDataset = reading_tfrecords.assemble_dataset(data_dir)
    return dataset
