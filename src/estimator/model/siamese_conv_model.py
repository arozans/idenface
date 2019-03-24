from typing import Type, Dict, Any

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import MnistRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel
from src.utils import utils, consts, image_summaries


class MnistSiameseModel(EstimatorModel):

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
        return result['classes']

    def get_predicted_scores(self, result: np.ndarray):
        return result['distances']
        # return None #todo


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


def contrastive_loss(model1, model2, y, margin):
    y = tf.cast(y, tf.float32)
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keep_dims=True))
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d), 0))
        return tf.reduce_mean(tmp + tmp2) / 2


def is_over_distance_margin(distances, margin):
    cond = tf.greater(distances, tf.fill(tf.shape(distances), margin))
    out = tf.where(cond, tf.zeros(tf.shape(distances)), tf.ones(tf.shape(distances)))
    return out


def siamese_model_fn(features, labels, mode, params):
    utils.log('Creating graph wih mode: {}'.format(mode))

    # with tf.name_scope('left_cnn_stack'):
    left_stack = conv_net(features[consts.LEFT_FEATURE_IMAGE], reuse=False)
    # with tf.name_scope('right_cnn_stack'):
    right_stack = conv_net(features[consts.RIGHT_FEATURE_IMAGE], reuse=True)

    margin = 0.5
    predict_margin = 0.15

    # distances = calc_norm(left_stack, right_stack)
    distances = tf.sqrt(tf.reduce_sum(tf.pow(left_stack - right_stack, 2), 1, keepdims=True))

    output = is_over_distance_margin(distances, predict_margin)
    # logits = tf.map_fn(lambda x: 1 if x > 0.2 else 0, distances)
    predictions = {
        "classes": output,
        "distances": distances
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = contrastive_loss(left_stack, right_stack, labels, margin)

    accuracy_metric = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='acc_op')
    mean_metric = tf.metrics.mean(values=distances, name='mean_op')
    eval_metric_ops = {
        "accuracy": accuracy_metric,
        "mean_distance": mean_metric
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        image_tensor = image_summaries.draw_scatters(left_stack, labels)
        image_summary = tf.summary.image('scatter', image_tensor)
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=params[consts.MODEL_DIR] + "/scatter",
            summary_op=image_summary
        )

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[eval_summary_hook])

    train_acc = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"], tf.cast(labels, tf.float32)), tf.float32))
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = determine_optimizer(config.optimizer)(config.learning_rate)
        optimizer = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_or_create_global_step())

        # tf.summary.scalar('accuracy3', accuracy[1])
        tf.summary.scalar('mean_distance', tf.reduce_mean(distances))
        tf.summary.scalar('accuracy', train_acc)

        # tf.summary.histogram("distances", distances)

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


#
# def squared_dist(A, B):
#     assert A.shape.as_list() == B.shape.as_list()
#
#     row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
#     row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
#
#     row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
#     row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
#
#     return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


def calc_norm(l, r):
    diff = l - r
    return tf.norm(diff, ord='euclidean', axis=1)
