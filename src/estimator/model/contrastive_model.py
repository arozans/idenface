from abc import ABC
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import MnistRawDataProvider, FmnistRawDataProvider, ExtruderRawDataProvider
from src.estimator.model import estimator_conv_model
from src.estimator.model.estimator_conv_model import EstimatorConvModel, merge_two_dicts
from src.utils import utils, consts, image_summaries
from src.utils.configuration import config


class ContrastiveModel(EstimatorConvModel, ABC):

    @property
    def produces_2d_embedding(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "contrastive"

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {
            consts.BATCH_SIZE: 300,
            consts.TRAIN_STEPS: 5 * 1000,
            consts.PREDICT_SIMILARITY_MARGIN: 0.4,
            consts.TRAIN_SIMILARITY_MARGIN: 0.5,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
            consts.FILTERS: [32, 64, 128, 256],
            consts.KERNEL_SIDE_LENGTHS: [7, 5, 3, 1],
            consts.DENSE_UNITS: [30, 2]
        }

    def get_model_fn(self):
        return self.contrastive_model_fn

    def get_predicted_labels(self, result: np.ndarray):
        return result[consts.INFERENCE_CLASSES]

    def get_predicted_scores(self, result: np.ndarray):
        return result[consts.INFERENCE_DISTANCES]

    def contrastive_model_fn(self, features, labels, mode, params):
        utils.log('Creating graph wih mode: {}'.format(mode))

        with tf.name_scope('left_cnn_stack'):
            left_stack = self.conv_net(features[consts.LEFT_FEATURE_IMAGE], reuse=False)
        with tf.name_scope('right_cnn_stack'):
            right_stack = self.conv_net(features[consts.RIGHT_FEATURE_IMAGE], reuse=True)

        train_similarity_margin = config[consts.TRAIN_SIMILARITY_MARGIN]
        predict_similarity_margin = config[consts.PREDICT_SIMILARITY_MARGIN]

        utils.log("train_similarity_margin: {}".format(train_similarity_margin))
        utils.log("predict_similarity_margin: {}".format(predict_similarity_margin))

        distances = calculate_distance(left_stack, right_stack)

        output = is_pair_similar(distances, predict_similarity_margin)

        predictions = {
            consts.INFERENCE_CLASSES: output,
            consts.INFERENCE_DISTANCES: distances,
            consts.INFERENCE_LEFT_EMBEDDINGS: left_stack,
            consts.INFERENCE_RIGHT_EMBEDDINGS: right_stack,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        pair_labels = labels[consts.PAIR_LABEL]
        loss = contrastive_loss(left_stack, right_stack, pair_labels, train_similarity_margin)

        if mode == tf.estimator.ModeKeys.EVAL:
            left_feature_labels = labels[consts.LEFT_FEATURE_LABEL]
            right_feature_labels = labels[consts.RIGHT_FEATURE_LABEL]

            image_tensor = image_summaries.draw_tf_clusters_plot(tf.concat((left_stack, right_stack), axis=0),
                                                                 tf.concat((left_feature_labels, right_feature_labels),
                                                                           axis=0))
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=config[consts.EVAL_STEPS_INTERVAL],
                output_dir=params[consts.MODEL_DIR] + "/clusters",
                summary_op=tf.summary.image('clusters', image_tensor)
            )
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
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[eval_summary_hook])

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = estimator_conv_model.determine_optimizer(config[consts.OPTIMIZER],
                                                                 config[consts.LEARNING_RATE])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())

            non_streaming_accuracy = estimator_conv_model.non_streaming_accuracy(
                tf.cast(tf.squeeze(predictions[consts.INFERENCE_CLASSES]), tf.int32),
                tf.cast(pair_labels, tf.int32))
            non_streaming_distances = tf.reduce_mean(distances)
            tf.summary.scalar('accuracy', non_streaming_accuracy)
            tf.summary.scalar('mean_distance', non_streaming_distances)

            logging_hook = tf.train.LoggingTensorHook(
                {
                    "accuracy_logging": non_streaming_accuracy,
                    "distances_logging": non_streaming_distances,
                    # "step": tf.train.get_or_create_global_step(),
                    # "loss": loss,
                    # "left_feature_labels": labels[consts.LEFT_FEATURE_LABEL],
                    # "right_feature_labels": labels[consts.RIGHT_FEATURE_LABEL]
                    # "left": left,
                },
                every_n_iter=config[consts.TRAIN_LOG_STEPS_INTERVAL])

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def contrastive_loss(left_embeddings, right_embeddings, labels, margin):
    labels = tf.cast(labels, tf.float32)
    labels = tf.expand_dims(labels, axis=1)
    with tf.name_scope("contrastive-loss"):
        distances = tf.sqrt(tf.reduce_sum(tf.pow(left_embeddings - right_embeddings + 0.001, 2), 1, keep_dims=True))
        same_elems_part = labels * tf.square(distances)
        different_elems_part = (1 - labels) * tf.square(tf.maximum((margin - distances), 0))
        return tf.reduce_mean(same_elems_part + different_elems_part) / 2


def is_pair_similar(distances, margin):
    cond = tf.greater(distances, tf.fill(tf.shape(distances), margin))
    out = tf.where(cond, tf.zeros(tf.shape(distances)), tf.ones(tf.shape(distances)))
    return out


def calculate_distance(left_stack, right_stack):
    return tf.sqrt(tf.reduce_sum(tf.pow(left_stack - right_stack, 2), 1, keepdims=True))


class MnistContrastiveModel(ContrastiveModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return MnistRawDataProvider()

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(
            super().additional_model_params, {
                consts.TRAIN_STEPS: 7 * 1000
            })


class FmnistContrastiveModel(ContrastiveModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return FmnistRawDataProvider()

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(
            super().additional_model_params, {
                consts.TRAIN_STEPS: 7 * 1000
            })


class ExtruderContrastiveModel(ContrastiveModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return ExtruderRawDataProvider(100)

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(
            super().additional_model_params, {
                consts.BATCH_SIZE: 64,
                consts.FILTERS: [8, 16, 32, 64, 128, 320, 2],
                consts.KERNEL_SIDE_LENGTHS: [5, 5, 5, 5, 5, 5, 5],
                consts.LEARNING_RATE: 0.001,
            })
