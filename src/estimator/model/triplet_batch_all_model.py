from typing import Type, Dict, Any

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import FmnistRawDataProvider, ExtruderRawDataProvider
from src.estimator.model import estimator_model
from src.estimator.model.estimator_model import EstimatorModel
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordTrainUnpairedDatasetProvider
from src.utils import utils, consts
from src.utils.configuration import config


class FmnistTripletBatchAllModel(EstimatorModel):

    @property
    def produces_2d_embedding(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "tri_ba"

    @property
    def summary(self) -> str:
        return self.name

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {
            consts.NUM_CHANNELS: 32,
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.PREDICT_SIMILARITY_MARGIN: 3.0,
            consts.EMBEDDING_SIZE: 64,
            consts.BATCH_SIZE: 64,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.001,
            consts.TRAIN_STEPS: 15 * 1000,
        }

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return FmnistRawDataProvider

    def is_dataset_paired(self, mode, params) -> bool:
        if mode == tf.estimator.ModeKeys.EVAL:
            return True
        dataset_provider_cls = params[consts.DATASET_PROVIDER_CLS]
        dataset_provider = dataset_provider_cls(params[consts.RAW_DATA_PROVIDER_CLS])
        return dataset_provider.is_train_paired()

    def get_model_fn(self):
        return self.triplet_batch_all_model_fn

    def get_predicted_labels(self, result: np.ndarray):
        return result[consts.INFERENCE_CLASSES]

    def get_predicted_scores(self, result: np.ndarray):
        return result[consts.INFERENCE_DISTANCES]

    def triplet_batch_all_model_fn(self, features, labels, mode, params):
        utils.log('Creating graph wih mode: {}'.format(mode))

        features = unpack_features(features, self.is_dataset_paired(mode, params))

        with tf.variable_scope('model'):
            # Compute the embeddings with the model
            embeddings = self.triplet_net(features)
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

        labels, pair_labels = unpack_labels(labels, self.is_dataset_paired(mode, params))
        # loss = contrastive_loss(left_stack, right_stack, pair_labels, train_similarity_margin)
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=config[consts.HARD_TRIPLET_MARGIN])

        if mode == tf.estimator.ModeKeys.EVAL:
            # image_tensor = image_summaries.draw_tf_clusters_plot(tf.concat((left_stack, right_stack), axis=0),
            #                                                      tf.concat((left_feature_labels, right_feature_labels),
            #                                                                axis=0))
            # eval_summary_hook = tf.train.SummarySaverHook(
            #     save_steps=config[consts.EVAL_STEPS_INTERVAL],
            #     output_dir=params[consts.MODEL_DIR] + "/clusters",
            #     summary_op=tf.summary.image('clusters', image_tensor)
            # )
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
            optimizer = estimator_model.determine_optimizer(config[consts.OPTIMIZER],
                                                            config[consts.LEARNING_RATE])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())
            training_logging_hook_dict = {}
            if self.is_dataset_paired(mode, params):
                non_streaming_accuracy = estimator_model.non_streaming_accuracy(
                    tf.cast(tf.squeeze(predictions[consts.INFERENCE_CLASSES]), tf.int32),
                    tf.cast(pair_labels, tf.int32))
                tf.summary.scalar('accuracy', non_streaming_accuracy)
                training_logging_hook_dict.update({"accuracy_logging": non_streaming_accuracy})
            non_streaming_distances = tf.reduce_mean(distances)
            tf.summary.scalar('mean_distance', non_streaming_distances)
            training_logging_hook_dict.update({"distances_logging": non_streaming_distances})
            logging_hook = tf.train.LoggingTensorHook(training_logging_hook_dict, every_n_iter=100)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

    def triplet_net(self, concat_features):
        data_description = self.raw_data_provider_cls.description()
        conv_input = tf.reshape(concat_features,
                                [-1, data_description.image_dimensions.width, data_description.image_dimensions.height,
                                 data_description.image_dimensions.channels])
        num_channels = config[consts.NUM_CHANNELS]
        channels = [num_channels, num_channels * 2]
        for i, c in enumerate(channels):
            with tf.variable_scope('block_{}'.format(i + 1)):
                conv_input = tf.contrib.layers.conv2d(conv_input, c, 3, activation_fn=tf.nn.relu,
                                                      padding='SAME',
                                                      weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                conv_input = tf.contrib.layers.max_pool2d(conv_input, 2, 2)

        image_side_after_pooling = data_description.image_dimensions.width // 4
        assert conv_input.shape[1:] == [image_side_after_pooling, image_side_after_pooling, num_channels * 2]

        conv_input = tf.reshape(conv_input,
                                [-1, image_side_after_pooling * image_side_after_pooling * num_channels * 2])
        with tf.variable_scope('fc_1'):
            conv_input = tf.layers.dense(conv_input, config[consts.EMBEDDING_SIZE])

        return conv_input


def conv_net(conv_input, reuse=False):
    conv_input = tf.reshape(conv_input, [-1, 28, 28, 1])
    filters = config[consts.FILTERS]
    kernel_side_lengths = config[consts.KERNEL_SIDE_LENGTHS]

    assert len(filters) >= 5
    assert len(kernel_side_lengths) >= 5

    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(conv_input, filters[0], [kernel_side_lengths[0], kernel_side_lengths[0]],
                                           activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, filters[1], [kernel_side_lengths[1], kernel_side_lengths[1]],
                                           activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, filters[2], [kernel_side_lengths[2], kernel_side_lengths[2]],
                                           activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, filters[3], [kernel_side_lengths[3], kernel_side_lengths[3]],
                                           activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        if len(filters) == 5:
            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, filters[4], [kernel_side_lengths[4], kernel_side_lengths[4]],
                                               activation_fn=None,
                                               padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        else:
            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, filters[4], [kernel_side_lengths[4], kernel_side_lengths[4]],
                                               activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            with tf.variable_scope("conv6") as scope:
                net = tf.contrib.layers.conv2d(net, filters[5], [kernel_side_lengths[5], kernel_side_lengths[5]],
                                               activation_fn=None,
                                               padding='SAME',
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
    return out  # todo: fix margin comparison!


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


def determine_optimizer(optimizer_param):
    if optimizer_param == consts.GRADIENT_DESCEND_OPTIMIZER:
        return tf.train.GradientDescentOptimizer
    elif optimizer_param == consts.ADAM_OPTIMIZER:
        return tf.train.AdamOptimizer
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_param))


def calc_norm(l, r):
    diff = l - r
    return tf.norm(diff, ord='euclidean', axis=1)


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


class FmnistTripletBatchAllUnpairedTrainModel(FmnistTripletBatchAllModel):

    @property
    def summary(self) -> str:
        return super().summary + "_unpaired"

    @property
    def dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordTrainUnpairedDatasetProvider


class ExtruderTripletBatchAllModel(FmnistTripletBatchAllModel):
    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {
            consts.NUM_CHANNELS: 32,
            consts.HARD_TRIPLET_MARGIN: 0.5,
            consts.PREDICT_SIMILARITY_MARGIN: 4.0,
            consts.EMBEDDING_SIZE: 64,
            consts.BATCH_SIZE: 8,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.01,
            consts.TRAIN_STEPS: 15 * 1000,
            consts.SHUFFLE_BUFFER_SIZE: 1000,
        }

    @property
    def summary(self) -> str:
        return "extruder_tri_ba"

    @property
    def dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordTrainUnpairedDatasetProvider

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return ExtruderRawDataProvider
