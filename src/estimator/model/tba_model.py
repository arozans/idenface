from abc import ABC
from typing import Type, Dict, Any

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import FmnistRawDataProvider, ExtruderRawDataProvider, MnistRawDataProvider
from src.estimator.model import estimator_conv_model
from src.estimator.model.estimator_conv_model import EstimatorConvModel, merge_two_dicts
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordTrainUnpairedDatasetProvider
from src.utils import utils, consts
from src.utils.configuration import config


class TBAModel(EstimatorConvModel, ABC):

    @property
    def produces_2d_embedding(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "tba"

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(
            super().additional_model_params, {
                consts.NUM_CHANNELS: 32,
                consts.HARD_TRIPLET_MARGIN: 0.5,
                consts.PREDICT_SIMILARITY_MARGIN: 3.0,
                consts.DENSE_UNITS: [64],
                consts.BATCH_SIZE: 64,
                consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
                consts.LEARNING_RATE: 0.001,
                consts.TRAIN_STEPS: 15 * 1000,
            })

    def is_dataset_paired(self, mode) -> bool:
        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
            return True
        return self.dataset_provider.is_train_paired()

    def get_model_fn(self):
        return self.triplet_batch_all_model_fn

    def get_predicted_labels(self, result: np.ndarray):
        return result[consts.INFERENCE_CLASSES]

    def get_predicted_scores(self, result: np.ndarray):
        return result[consts.INFERENCE_DISTANCES]

    def triplet_batch_all_model_fn(self, features, labels, mode, params):
        utils.log('Creating graph wih mode: {}'.format(mode))

        features = unpack_features(features, self.is_dataset_paired(mode))

        embeddings = self.conv_net(features)

        embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
        tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

        middle_idx = tf.cast(tf.shape(embeddings)[0] / 2, tf.int64)
        left_embeddings = embeddings[:middle_idx]
        right_embeddings = embeddings[middle_idx:]

        distances = calculate_distance(left_embeddings, right_embeddings)

        output = is_pair_similar(distances, config[consts.PREDICT_SIMILARITY_MARGIN])
        predictions = {
            consts.INFERENCE_CLASSES: output,
            consts.INFERENCE_DISTANCES: distances,
            consts.INFERENCE_LEFT_EMBEDDINGS: left_embeddings,
            consts.INFERENCE_RIGHT_EMBEDDINGS: right_embeddings,
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        labels, pair_labels = unpack_labels(labels, self.is_dataset_paired(mode))
        # loss, fraction_positive_triplets, num_positive_triplets, num_valid_triplets = batch_all_triplet_loss(
        #     labels, embeddings, margin=config[consts.HARD_TRIPLET_MARGIN])
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings,
                                                                       margin=config[consts.HARD_TRIPLET_MARGIN])

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy_metric = tf.metrics.accuracy(labels=pair_labels, predictions=predictions[consts.INFERENCE_CLASSES],
                                                  name='accuracy_metric')
            recall_metric = tf.metrics.recall(labels=pair_labels, predictions=predictions[consts.INFERENCE_CLASSES],
                                              name='recall_metric')
            precision_metric = tf.metrics.precision(labels=pair_labels,
                                                    predictions=predictions[consts.INFERENCE_CLASSES],
                                                    name='precision_metric')
            f1_metric = tf.contrib.metrics.f1_score(labels=pair_labels,
                                                    predictions=predictions[consts.INFERENCE_CLASSES],
                                                    name='f1_metric')
            mean_metric = tf.metrics.mean(values=distances, name=consts.INFERENCE_CLASSES)
            eval_metric_ops = {
                consts.METRIC_ACCURACY: accuracy_metric,
                consts.METRIC_RECALL: recall_metric,
                consts.METRIC_PRECISION: precision_metric,
                consts.METRIC_F1: f1_metric,
                consts.METRIC_MEAN_DISTANCE: mean_metric,
            }

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = estimator_conv_model.determine_optimizer(config[consts.OPTIMIZER],
                                                                 config[consts.LEARNING_RATE])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())
            training_logging_hook_dict = {}
            if self.is_dataset_paired(mode):
                non_streaming_accuracy = estimator_conv_model.non_streaming_accuracy(
                    tf.cast(tf.squeeze(predictions[consts.INFERENCE_CLASSES]), tf.int32),
                    tf.cast(pair_labels, tf.int32))
                tf.summary.scalar('accuracy', non_streaming_accuracy)
                training_logging_hook_dict.update({"accuracy_logging": non_streaming_accuracy})
            non_streaming_distances = tf.reduce_mean(distances)
            tf.summary.scalar('mean_distance', non_streaming_distances)
            # tf.summary.scalar('postitive_triplets', num_positive_triplets)
            training_logging_hook_dict.update({"distances_logging": non_streaming_distances})
            # training_logging_hook_dict.update(
            #     {
            #         "fraction_positive_triplets": fraction_positive_triplets,
            #         "num_positive_triplets": num_positive_triplets,
            #         "num_valid_triplets": num_valid_triplets,
            #     })
            logging_hook = tf.train.LoggingTensorHook(
                training_logging_hook_dict,
                every_n_iter=config[consts.TRAIN_LOG_STEPS_INTERVAL]
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook]
            )


def is_pair_similar(distances, margin):
    cond = tf.greater(distances, tf.fill(tf.shape(distances), margin))
    out = tf.where(cond, tf.zeros(tf.shape(distances)), tf.ones(tf.shape(distances)))
    return out


def unpack_features(features, is_dataset_paired):
    if is_dataset_paired:
        left_features = features[consts.LEFT_FEATURE_IMAGE]
        right_features = features[consts.RIGHT_FEATURE_IMAGE]
        concat_features = tf.concat((left_features, right_features), axis=0)
        return concat_features
    else:
        return features[consts.FEATURES]


def unpack_labels(labels, is_dataset_paired):
    if is_dataset_paired:
        pair_labels = labels[consts.PAIR_LABEL]
        left_feature_labels = labels[consts.LEFT_FEATURE_LABEL]
        right_feature_labels = labels[consts.RIGHT_FEATURE_LABEL]
        labels = tf.concat((left_feature_labels, right_feature_labels), axis=0)
        return labels, pair_labels
    else:
        return labels[consts.LABELS], None


def calculate_distance(left_stack, right_stack):
    return tf.sqrt(tf.reduce_sum(tf.pow(left_stack - right_stack, 2), 1, keepdims=True))


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)

    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))

    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets, num_positive_triplets, num_valid_triplets


def _pairwise_distances(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


class MnistTBAModel(TBAModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return MnistRawDataProvider()


class FmnistTBAModel(TBAModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return FmnistRawDataProvider()


class FmnistTBAUnpairedTrainModel(TBAModel):

    @property
    def summary(self) -> str:
        return super().summary + "_unpaired"

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return FmnistRawDataProvider()

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordTrainUnpairedDatasetProvider


class ExtruderTBAModel(TBAModel):
    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(
            super().additional_model_params, {
                consts.NUM_CHANNELS: 32,
                consts.FILTERS: [8, 16, 32, 64, 128, 256, 512],
                consts.KERNEL_SIDE_LENGTHS: [5, 5, 5, 5, 5, 5, 5],
                consts.HARD_TRIPLET_MARGIN: 0.5,
                consts.PREDICT_SIMILARITY_MARGIN: 6.3,
                consts.DENSE_UNITS: [80],
                consts.BATCH_SIZE: 256,  # maybe larger batch?
                consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
                consts.LEARNING_RATE: 0.0005,
                consts.TRAIN_STEPS: 2000,
                consts.SHUFFLE_BUFFER_SIZE: 1000,
                consts.EVAL_STEPS_INTERVAL: 100,
                consts.TRAIN_LOG_STEPS_INTERVAL: 100,
                consts.GLOBAL_SUFFIX: "semihard_v2",
            })

    @property
    def summary(self) -> str:
        return "tri_ba_extruder"

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordTrainUnpairedDatasetProvider

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return ExtruderRawDataProvider(consts.EXTRUDER_REDUCED_SIZE_IMAGE_SIDE_PIXEL_COUNT)
