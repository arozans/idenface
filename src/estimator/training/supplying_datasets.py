from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.data import TFRecordDataset, Dataset
from tensorflow.python.framework import dtypes

from src.data import preparing_data
from src.data.common_types import DatasetSpec, DatasetType, AbstractRawDataProvider
from src.data.raw_data import raw_data
from src.data.tfrecord.reading import reading_tfrecords
from src.utils import utils, consts
from src.utils.configuration import config


class AbstractDatasetProvider(ABC):
    def __init__(self, raw_data_provider: AbstractRawDataProvider):
        self.raw_data_provider = raw_data_provider

    def train_input_fn(self) -> Dataset:
        utils.log('Creating train_input_fn')
        train_data_config = DatasetSpec(raw_data_provider=self.raw_data_provider,
                                        type=DatasetType.TRAIN,
                                        with_excludes=False,
                                        encoding=self.is_encoded(),
                                        paired=self.is_train_paired())
        return self.supply_dataset(dataset_spec=train_data_config,
                                   shuffle_buffer_size=config[consts.SHUFFLE_BUFFER_SIZE],
                                   batch_size=config[consts.BATCH_SIZE],
                                   repeat=True)

    def eval_input_fn(self) -> Dataset:
        utils.log('Creating eval_input_fn')
        test_data_config = DatasetSpec(raw_data_provider=self.raw_data_provider, type=DatasetType.TEST,
                                       with_excludes=False, encoding=self.is_encoded())
        return self.supply_dataset(dataset_spec=test_data_config, batch_size=config[consts.BATCH_SIZE])

    def eval_with_excludes_input_fn(self) -> Dataset:
        utils.log('Creating eval_input_fn with excluded elements')
        test_ignoring_excludes = DatasetSpec(raw_data_provider=self.raw_data_provider, type=DatasetType.TEST,
                                             with_excludes=True, encoding=self.is_encoded())
        return self.supply_dataset(dataset_spec=test_ignoring_excludes, batch_size=config[consts.BATCH_SIZE])

    def infer(self, take_num: int) -> Dataset:
        utils.log('Creating infer_fn')
        test_with_excludes = DatasetSpec(raw_data_provider=self.raw_data_provider,
                                         type=DatasetType.TEST,
                                         with_excludes=True, encoding=self.is_encoded())
        return self.supply_dataset(dataset_spec=test_with_excludes,
                                   batch_size=take_num,
                                   repeat=False,
                                   shuffle_buffer_size=config[consts.SHUFFLE_BUFFER_SIZE],
                                   prefetch=False,
                                   take_num=take_num)

    def predict_input_fn(self, features, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(features)

        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        return dataset

    def supply_dataset(self, dataset_spec: DatasetSpec, shuffle_buffer_size: Optional[int] = None,
                       batch_size: Optional[int] = None, repeat: bool = False, prefetch: bool = True,
                       take_num=None) -> Dataset:

        dataset = self.build_dataset(dataset_spec)

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

    @abstractmethod
    def build_dataset(self, dataset_spec: DatasetSpec) -> tf.data.Dataset:
        pass

    def is_encoded(self):
        return True

    def is_train_paired(self):
        return True


class TFRecordDatasetProvider(AbstractDatasetProvider):

    def build_dataset(self, dataset_spec: DatasetSpec):
        return build_from_tfrecord(dataset_spec)


def build_from_tfrecord(dataset_spec: DatasetSpec):
    data_dir: Path = preparing_data.find_or_create_dataset_dir(dataset_spec)

    dataset: TFRecordDataset = reading_tfrecords.assemble_dataset(data_dir, dataset_spec)
    return dataset


class FromGeneratorDatasetProvider(AbstractDatasetProvider):

    def build_dataset(self, dataset_spec: DatasetSpec):
        utils.log('Creating generator for: {}'.format(dataset_spec))
        self.excludes = config[consts.EXCLUDED_KEYS] if not dataset_spec.with_excludes else []
        self.raw_images, self.raw_labels = raw_data.get_raw_data(dataset_spec)
        self.unique_labels = np.unique(self.raw_labels)
        self.label_to_idxs_mapping = {label: np.flatnonzero(self.raw_labels == label) for label in
                                      self.unique_labels}

        if dataset_spec.type is DatasetType.TRAIN:
            return self.build_from_generator(dataset_spec)
        else:
            return build_from_tfrecord(dataset_spec)

    def build_from_generator(self, dataset_spec):
        image_side_length = dataset_spec.raw_data_provider.description.image_dimensions.width

        def generator():
            while True:
                left_idxs, right_idxs, pair_label, left_label, right_label = self._get_siamese_pair()
                yield (
                    {
                        consts.LEFT_FEATURE_IMAGE: self.raw_images[left_idxs] - 0.5,
                        consts.RIGHT_FEATURE_IMAGE: self.raw_images[right_idxs] - 0.5
                    },
                    {
                        consts.PAIR_LABEL: pair_label,
                        consts.LEFT_FEATURE_LABEL: left_label,
                        consts.RIGHT_FEATURE_LABEL: right_label
                    }
                )

        dataset: Dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(
                {
                    consts.LEFT_FEATURE_IMAGE: dtypes.float32,
                    consts.RIGHT_FEATURE_IMAGE: dtypes.float32
                },
                {
                    consts.PAIR_LABEL: dtypes.int64,
                    consts.LEFT_FEATURE_LABEL: dtypes.int64,
                    consts.RIGHT_FEATURE_LABEL: dtypes.int64
                }
            ),
            output_shapes=(
                {
                    consts.LEFT_FEATURE_IMAGE: (image_side_length, image_side_length, 1),
                    consts.RIGHT_FEATURE_IMAGE: (image_side_length, image_side_length, 1)
                },
                {
                    consts.PAIR_LABEL: (),
                    consts.LEFT_FEATURE_LABEL: (),
                    consts.RIGHT_FEATURE_LABEL: ()
                }
            )
        )
        return dataset

    def _get_siamese_similar_pair(self):
        label = np.random.choice(np.delete(self.unique_labels, self.excludes))
        try:
            l, r = np.random.choice(self.label_to_idxs_mapping[label], 2, replace=False)
        except:
            raise ValueError("Problem during similar pair retrieval: {}".format(self.label_to_idxs_mapping, label))
        return l, r, 1, label, label

    def _get_siamese_dissimilar_pair(self):
        label_l, label_r = np.random.choice(np.delete(self.unique_labels, self.excludes), 2, replace=False)
        l = np.random.choice(self.label_to_idxs_mapping[label_l])
        r = np.random.choice(self.label_to_idxs_mapping[label_r])
        return l, r, 0, label_l, label_r

    def _get_siamese_pair(self):
        if np.random.random() < 0.5:
            return self._get_siamese_similar_pair()
        else:
            return self._get_siamese_dissimilar_pair()


class TFRecordTrainUnpairedDatasetProvider(TFRecordDatasetProvider):

    def is_train_paired(self):
        return False
