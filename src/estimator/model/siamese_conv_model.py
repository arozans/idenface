from typing import Type, Dict, Any

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import MnistRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel
from src.utils import utils, consts, image_summaries
from src.utils.configuration import config


class MnistSiameseModel(EstimatorModel):

    @property
    def produces_2d_embedding(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "siamese"

    @property
    def summary(self) -> str:
        return self.name

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {}

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return MnistRawDataProvider

    def get_model_fn(self):
        return siamese_model_fn

    def get_predicted_labels(self, result: np.ndarray):
        return result[consts.INFERENCE_CLASSES]

    def get_predicted_scores(self, result: np.ndarray):
        return result[consts.INFERENCE_DISTANCES]


def conv_net(conv_input, reuse=False):
    conv_input = tf.reshape(conv_input, [-1, 28, 28, 1])
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(conv_input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        net = tf.contrib.layers.flatten(net)

    return net


def contrastive_loss(model1, model2, labels, margin):
    labels = tf.cast(labels, tf.float32)
    labels = tf.expand_dims(labels, axis=1)
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keep_dims=True))
        tmp = labels * tf.square(d)
        tmp2 = (1 - labels) * tf.square(tf.maximum((margin - d), 0))
        return tf.reduce_mean(tmp + tmp2) / 2


def is_pair_similar(distances, margin):
    cond = tf.greater(distances, tf.fill(tf.shape(distances), margin))
    out = tf.where(cond, tf.zeros(tf.shape(distances)), tf.ones(tf.shape(distances)))
    return out


def siamese_model_fn(features, labels, mode, params):
    utils.log('Creating graph wih mode: {}'.format(mode))

    left_stack = conv_net(features[consts.LEFT_FEATURE_IMAGE], reuse=False)
    right_stack = conv_net(features[consts.RIGHT_FEATURE_IMAGE], reuse=True)

    train_similarity_margin = config.train_similarity_margin
    predict_similarity_margin = config.predict_similarity_margin

    distances = tf.sqrt(tf.reduce_sum(tf.pow(left_stack - right_stack, 2), 1, keepdims=True))

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

    accuracy_metric = tf.metrics.accuracy(labels=pair_labels, predictions=predictions["classes"], name='acc_op')
    mean_metric = tf.metrics.mean(values=distances, name='mean_op')
    eval_metric_ops = {
        "accuracy": accuracy_metric,
        "mean_distance": mean_metric
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        left_feature_labels = labels[consts.LEFT_FEATURE_LABEL]
        right_feature_labels = labels[consts.RIGHT_FEATURE_LABEL]

        image_tensor = image_summaries.draw_tf_clusters_plot(tf.concat((left_stack, right_stack), axis=0),
                                                             tf.concat((left_feature_labels, right_feature_labels),
                                                                       axis=0))
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=config.eval_steps_interval,
            output_dir=params[consts.MODEL_DIR] + "/clusters",
            summary_op=tf.summary.image('clusters', image_tensor)
        )

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[eval_summary_hook])

    train_acc = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"], tf.cast(pair_labels, tf.float32)), tf.float32))
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_or_create_global_step())

        # tf.summary.scalar('accuracy3', accuracy[1])
        tf.summary.scalar('mean_distance', tf.reduce_mean(distances))
        tf.summary.scalar('accuracy', accuracy_metric[1])

        logging_hook = tf.train.LoggingTensorHook(
            {
                "accuracy": accuracy_metric[1],
                "distances": mean_metric[1]
            }, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def determine_optimizer(optimizer_param):
    if optimizer_param == 'GradientDescent':
        return tf.train.GradientDescentOptimizer
    elif optimizer_param == 'AdamOptimizer':
        return tf.train.AdamOptimizer
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_param))


def calc_norm(l, r):
    diff = l - r
    return tf.norm(diff, ord='euclidean', axis=1)
