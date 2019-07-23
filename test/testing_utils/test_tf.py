import numpy as np
import pytest
import tensorflow as tf

from testing_utils.tf_helpers import run_eagerly


def get_indexes_of_given_label(labels, label_to_select):
    with tf.compat.v1.Session() as sess:
        labels = tf.constant(labels, dtype=tf.int64)
        label_to_select = tf.constant(label_to_select, dtype=tf.int64)
        searched_label_positions = tf.math.equal(labels, label_to_select)
        searched_label_coor = tf.where(searched_label_positions)
        return sess.run(searched_label_coor)


def get_point_coor_at_indexes(points, indexes, column):
    with tf.compat.v1.Session() as sess:
        points = tf.constant(points, dtype=tf.int64)
        c = tf.constant(column)[None, None]
        c = tf.tile(c, [tf.shape(indexes)[0], 1])
        b = tf.concat([indexes, c], axis=1)
        res = tf.gather_nd(points, b)
        return sess.run(res)


def test_should_simulate_advanced_indexing():
    a = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    labels = np.array([0, 1, 1, 0])
    label_to_select = 0
    res = get_indexes_of_given_label(labels, label_to_select)
    assert (res == np.array([[0], [3]])).all()

    label_to_select = 1
    res2 = get_indexes_of_given_label(labels, label_to_select)
    assert (res2 == np.array([[1], [2]])).all()

    res3 = get_point_coor_at_indexes(a, res, 0)
    assert (res3 == np.array([1, 7])).all()

    res4 = get_point_coor_at_indexes(a, res2, 1)
    assert (res4 == np.array([4, 6])).all()


def test_concat():
    with tf.compat.v1.Session() as sess:
        a = tf.constant(np.array([[0], [3]]))
        b = tf.constant(np.array([[1], [2]]))
        c = tf.concat([a, b], axis=1)
        d = tf.zeros(shape=[a.shape[0], 1], dtype=tf.int64)
        e = tf.concat([c, d], axis=1)
        res, res2 = sess.run([c, e])
    assert (res == np.array([[0, 1], [3, 2]])).all()
    assert (res2 == np.array([[0, 1, 0], [3, 2, 0]])).all()


def contrastive_loss(distance, y, margin):
    import numpy as np
    print("distance shape: ", distance.shape)
    print("labels shape: ", y.shape)
    print("margin shape: ", np.array(margin).shape)

    with tf.name_scope("contrastive-loss"):
        print("tf.square(distance)  shape: ", (tf.square(distance)).shape)
        similarity = y * tf.square(distance)
        print("similarity shape: ", similarity.shape)

        print("(margin - distance) shape: ", (margin - distance).shape)
        print("tf.maximum((margin - distance),0) shape: ", tf.maximum((margin - distance), 0).shape)
        print("(1 - y) shape: ", (1 - y).shape)
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
        print("dissimilarity shape: ", dissimilarity.shape)
        print("tf.reduce_mean(dissimilarity + similarity) / 2 shape: ",
              (tf.reduce_mean(dissimilarity + similarity) / 2).shape)

        return tf.reduce_mean(dissimilarity + similarity) / 2


@pytest.mark.parametrize('distance, labels, margin',
                         [
                             (np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32),
                              np.array([[1.0], [0.0], [0.0], [0.0], [1.0]], dtype=np.float32), 0.5),
                             (np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32),
                              np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), 0.5)
                         ],
                         ids=['correct', 'not_correct'])
@run_eagerly
def test_contrastive_loss_dims(distance, labels, margin):
    contrastive_loss(distance, labels, margin)
