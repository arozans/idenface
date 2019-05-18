import time
from enum import Enum, auto
from typing import Tuple, Type, Dict, Any

import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from src.data.common_types import AbstractRawDataProvider, DataDescription, RawDatasetFragment, ImageDimensions, \
    DatasetSpec, DatasetType
from src.data.raw_data.raw_data_providers import MnistRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, FromGeneratorDatasetProvider
from src.utils import utils, consts
from src.utils.configuration import config
from testing_utils import testing_consts
from testing_utils.testing_consts import FAKE_IMAGES_CLASSES_COUNT
from testing_utils.testing_helpers import generate_fake_images, generate_fake_labels, determine_optimizer


class TestDatasetVariant(Enum):
    NUMBERTRANSLATION = auto()
    FOO = auto()
    FAKEMNIST = auto()


# noinspection PyTypeChecker
FAKE_DATA_DESCRIPTION = DataDescription(variant=TestDatasetVariant.FOO,
                                        image_dimensions=ImageDimensions(testing_consts.FAKE_IMAGE_SIDE_PIXEL_COUNT),
                                        classes_count=FAKE_IMAGES_CLASSES_COUNT)
# noinspection PyTypeChecker
FAKE_MNIST_DESCRIPTION = DataDescription(variant=TestDatasetVariant.FAKEMNIST,
                                         image_dimensions=ImageDimensions(testing_consts.MNIST_IMAGE_SIDE_PIXEL_COUNT),
                                         classes_count=testing_consts.MNIST_IMAGES_CLASSES_COUNT)


class FakeRawDataProvider(AbstractRawDataProvider):

    def __init__(self, description: DataDescription = None, raw_fake_dataset: RawDatasetFragment = None, curated=False):
        self._description = description
        if raw_fake_dataset:
            raw_dataset_fragment = raw_fake_dataset
        else:
            raw_dataset_fragment = self._generate_fake_raw_dataset_fragment(self.description, curated)
        self.raw_fake_dataset = RawDataset(
            train=raw_dataset_fragment,
            test=raw_dataset_fragment
        )

    @property
    def description(self) -> DataDescription:
        return self._description if self._description is not None else FAKE_DATA_DESCRIPTION

    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.raw_fake_dataset.train.features, self.raw_fake_dataset.train.labels

    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.raw_fake_dataset.test.features, self.raw_fake_dataset.test.labels

    @staticmethod
    def _generate_fake_raw_dataset_fragment(desc, curated):
        labels = generate_fake_labels(
            size=testing_consts.FAKE_IMAGES_IN_DATASET_COUNT,
            classes=desc.classes_count,
            curated=curated
        )
        features = generate_fake_images(
            size=(testing_consts.FAKE_IMAGES_IN_DATASET_COUNT, *desc.image_dimensions.as_tuple()),
            mimic_values=labels if curated else None,
            storage_method=desc.storage_method
        )

        return RawDatasetFragment(features=features, labels=labels)


@dataclass
class RawDataset:
    train: RawDatasetFragment
    test: RawDatasetFragment


class FakeModel(EstimatorModel):
    def get_predicted_labels(self, result: np.ndarray):
        pass

    def get_predicted_scores(self, result: np.ndarray):
        pass

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return self._data_provider

    def __init__(self, data_provider=FakeRawDataProvider(curated=True)):
        self._data_provider: AbstractRawDataProvider = data_provider
        self.model_fn_calls = 0
        self.id = time.strftime('d%y%m%dt%H%M%S')

    @property
    def name(self) -> str:
        return "fakeCNN"

    @property
    def summary(self) -> str:
        return self.name + self.id

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {}

    def get_model_fn(self):
        return self.fake_cnn_model_fn

    def fake_cnn_model_fn(self, features, labels, mode, params):
        utils.log('Creating graph wih mode: {}'.format(mode))
        self.model_fn_calls += 1
        with tf.name_scope('left_cnn_stack'):
            flatten_left_stack = self.create_simple_cnn_layers(features[consts.LEFT_FEATURE_IMAGE])
        with tf.name_scope('right_cnn_stack'):
            flatten_right_stack = self.create_simple_cnn_layers(features[consts.RIGHT_FEATURE_IMAGE])

        flatted_concat = tf.concat(axis=1, values=[flatten_left_stack, flatten_right_stack])

        logits = tf.layers.dense(inputs=flatted_concat, units=2)

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        pair_labels = labels[consts.PAIR_LABEL]
        loss = tf.losses.sparse_softmax_cross_entropy(labels=pair_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = determine_optimizer(config[consts.OPTIMIZER])(config[consts.LEARNING_RATE])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def create_simple_cnn_layers(self, image):
        side_pixel_count = self.raw_data_provider.description.image_dimensions.width
        flat_image = tf.reshape(image, [-1, side_pixel_count, side_pixel_count, 1])

        conv1 = tf.layers.conv2d(
            inputs=flat_image,
            filters=2,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool1, [-1, (side_pixel_count // 2 * side_pixel_count // 2 * 2)])
        return pool2_flat


class MnistCNNModelWithGeneratedDataset(MnistCNNModel):
    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return FromGeneratorDatasetProvider

    @property
    def summary(self) -> str:
        return self.name + '_generated_dataset'


class MnistCNNModelWithTfRecordDataset(MnistCNNModel):
    @property
    def summary(self) -> str:
        return self.name + '_TFRecord_dataset'


MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES = DatasetSpec(raw_data_provider=MnistRawDataProvider(),
                                                         type=DatasetType.TRAIN,
                                                         with_excludes=True)
MNIST_TRAIN_DATASET_SPEC = DatasetSpec(raw_data_provider=MnistRawDataProvider(),
                                       type=DatasetType.TRAIN,
                                       with_excludes=False)
MNIST_TEST_DATASET_SPEC = DatasetSpec(raw_data_provider=MnistRawDataProvider(),
                                      type=DatasetType.TEST,
                                      with_excludes=False)
MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES = DatasetSpec(raw_data_provider=MnistRawDataProvider(),
                                                        type=DatasetType.TEST, with_excludes=True)
FAKE_TRAIN_DATASET_SPEC = DatasetSpec(raw_data_provider=FakeRawDataProvider(), type=DatasetType.TRAIN,
                                      with_excludes=False)
FAKE_TEST_DATASET_SPEC = DatasetSpec(raw_data_provider=FakeRawDataProvider(), type=DatasetType.TEST,
                                     with_excludes=False)
