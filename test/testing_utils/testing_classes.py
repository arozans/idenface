import time
from enum import Enum, auto
from typing import Tuple, Type, Dict, Any

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, replace

from src.data.common_types import AbstractRawDataProvider, DataDescription, DatasetSpec, DatasetType, \
    DatasetStorageMethod, DatasetFragment
from src.data.raw_data.raw_data_providers import MnistRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, FromGeneratorDatasetProvider
from src.utils import utils, consts
from src.utils.configuration import config
from testing_utils import testing_consts
from testing_utils.testing_helpers import generate_fake_images, generate_fake_labels, determine_optimizer


class FakeRawDataProvider(AbstractRawDataProvider):

    def __init__(self):
        random_data_fragment = DatasetFragment(
            features=generate_fake_images(
                size=(testing_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                      self.description().image_side_length,
                      self.description().image_side_length, 1)
            ),
            labels=generate_fake_labels(
                size=testing_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                classes=self.description().classes_count
            )
        )
        self.raw_fake_dataset = RawDataset(
            train=random_data_fragment,
            test=random_data_fragment
        )

    @staticmethod
    def description() -> DataDescription:
        return DataDescription(TestDatasetVariant.FOO, testing_consts.FAKE_IMAGE_SIDE_PIXEL_COUNT,
                               testing_consts.FAKE_IMAGES_CLASSES_COUNT)

    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.raw_fake_dataset.train.features, self.raw_fake_dataset.train.labels

    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.raw_fake_dataset.test.features, self.raw_fake_dataset.test.labels


class CuratedFakeRawDataProvider(FakeRawDataProvider):

    def __init__(self, description=None):
        desc = description if description is not None else self.description()
        super().__init__()
        curated_labels = generate_fake_labels(
            size=testing_consts.FAKE_IMAGES_IN_DATASET_COUNT,
            classes=desc.classes_count,
            curated=True
        )
        curated_data_fragment = DatasetFragment(
            features=generate_fake_images(
                size=(testing_consts.FAKE_IMAGES_IN_DATASET_COUNT, desc.image_side_length,
                      desc.image_side_length, desc.image_channels),
                mimic_values=curated_labels,
                storage_method=desc.storage_method
            ),
            labels=curated_labels
        )
        self.raw_fake_dataset = RawDataset(
            train=curated_data_fragment,
            test=curated_data_fragment
        )


class CuratedFakeRawOnDiscDataProvider(CuratedFakeRawDataProvider):

    @staticmethod
    def description() -> DataDescription:
        return replace(CuratedFakeRawDataProvider.description(), storage_method=DatasetStorageMethod.ON_DISC)


class CuratedMnistFakeRawDataProvider(CuratedFakeRawDataProvider):

    @staticmethod
    def description() -> DataDescription:
        return DataDescription(TestDatasetVariant.FAKEMNIST, testing_consts.MNIST_IMAGE_SIDE_PIXEL_COUNT,
                               testing_consts.MNIST_IMAGES_CLASSES_COUNT)


@dataclass
class RawDataset:
    train: DatasetFragment
    test: DatasetFragment


class TestDatasetVariant(Enum):
    NUMBERTRANSLATION = auto()
    FOO = auto()
    FAKEMNIST = auto()


class FakeModel(EstimatorModel):
    def get_predicted_labels(self, result: np.ndarray):
        pass

    def get_predicted_scores(self, result: np.ndarray):
        pass

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return self._data_provider

    def __init__(self, data_provider=CuratedFakeRawDataProvider):
        self._data_provider: Type[AbstractRawDataProvider] = data_provider
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
            optimizer = determine_optimizer(config[consts.OPTIMIZER])(
                config[consts.LEARNING_RATE])  # fix params to get from flags
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def create_simple_cnn_layers(self, image):
        side_pixel_count = self.raw_data_provider_cls.description().image_side_length
        flat_image = tf.reshape(image, [-1, side_pixel_count, side_pixel_count, 1])

        # Convolutional Layer #1
        # Computes 2 features using a 2x2 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, side_pixel_count, side_pixel_count, 1]
        # Output Tensor Shape: [batch_size, side_pixel_count, side_pixel_count, 2]
        conv1 = tf.layers.conv2d(
            inputs=flat_image,
            filters=2,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, side_pixel_count, side_pixel_count, 2]
        # Output Tensor Shape: [batch_size, side_pixel_count/2, side_pixel_count/2, 2]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, TEST_IMAGE_SIDE_PIXEL_COUNT/2, TEST_IMAGE_SIDE_PIXEL_COUNT/2, 2]
        # Output Tensor Shape: [batch_size, TEST_IMAGE_SIDE_PIXEL_COUNT/2 * TEST_IMAGE_SIDE_PIXEL_COUNT/2 * 2]
        pool2_flat = tf.reshape(pool1, [-1, (side_pixel_count // 2 * side_pixel_count // 2 * 2)])
        return pool2_flat


class MnistCNNModelWithGeneratedDataset(MnistCNNModel):
    @property
    def dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return FromGeneratorDatasetProvider

    @property
    def summary(self) -> str:
        return self.name + '_generated_dataset'


class MnistCNNModelWithTfRecordDataset(MnistCNNModel):
    @property
    def summary(self) -> str:
        return self.name + '_TfRecordr_dataset'


MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                                         type=DatasetType.TRAIN,
                                                         with_excludes=True)
MNIST_TRAIN_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                       type=DatasetType.TRAIN,
                                       with_excludes=False)
MNIST_TEST_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                      type=DatasetType.TEST,
                                      with_excludes=False)
MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                                        type=DatasetType.TEST, with_excludes=True)
FAKE_TRAIN_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=FakeRawDataProvider, type=DatasetType.TRAIN,
                                      with_excludes=False)
FAKE_TEST_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=FakeRawDataProvider, type=DatasetType.TEST,
                                     with_excludes=False)
FAKE_DATA_DESCRIPTION = DataDescription(variant=TestDatasetVariant.FOO,
                                        image_side_length=testing_consts.FAKE_IMAGE_SIDE_PIXEL_COUNT,
                                        classes_count=testing_consts.FAKE_IMAGES_CLASSES_COUNT)
