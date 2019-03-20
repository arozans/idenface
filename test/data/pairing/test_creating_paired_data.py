import collections
from collections import OrderedDict

import numpy as np
from hamcrest import assert_that, is_, is_not, is_in
from matplotlib import pyplot as plt

from data.conftest import TRANSLATIONS_TRAIN_DATASET_SPEC
from helpers.test_helpers import NumberTranslation
from src.data.pairing import creating_paired_data
from src.utils import consts


def test_should_create_labels_and_dict_with_pairs():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    assert_that(features, is_(OrderedDict))
    assert len(features.keys()) == 2

    assert_that(labels, is_(np.ndarray))
    assert len(labels.shape) == 1


def test_features_should_have_correct_key_names():
    features, _ = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    keys = list(features.keys())
    assert keys[0] == consts.LEFT_FEATURE_IMAGE
    assert keys[1] == consts.RIGHT_FEATURE_IMAGE


def test_left_and_right_should_have_same_length():
    features, _ = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    values = list(features.values())
    assert len(values[0]) == len(values[1])


def test_should_create_correct_number_of_pairs(number_translation_features_and_labels):
    features, labels = number_translation_features_and_labels
    paired_features_dict, _ = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)  # (9, 9)

    left_pair_elem = paired_features_dict[consts.LEFT_FEATURE_IMAGE]
    assert len(left_pair_elem) >= len(features)


def test_number_of_same_and_different_pairs_should_be_equal():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    counter = collections.Counter(labels)

    assert len(set(counter.values())) == 1


def test_same_pairs_should_consists_of_same_class():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    same_features_left = features[consts.LEFT_FEATURE_IMAGE][labels == 1]
    same_features_right = features[consts.RIGHT_FEATURE_IMAGE][labels == 1]

    for e, f in zip(same_features_left, same_features_right):
        assert (e.number == f.number)


def test_different_pairs_should_consists_of_different_classes():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    same_features_left = features[consts.LEFT_FEATURE_IMAGE][labels == 0]
    same_features_right = features[consts.RIGHT_FEATURE_IMAGE][labels == 0]

    for e, f in zip(same_features_left, same_features_right):
        assert (e.number != f.number)


def test_each_class_should_be_included(number_translation_features_dict):
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    same_left_images = get_same_left_features(features, labels)
    unique_classes = set(same_left_images)

    assert len(unique_classes) == len(number_translation_features_dict.keys())


def test_should_honor_excludes(mocker, number_translation_features_dict):
    mocker.patch('src.utils.configuration._excluded_keys', ['three'])
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    same_left_images = get_same_left_features(features, labels)
    unique_classes = set(same_left_images)

    assert len(unique_classes) == len(number_translation_features_dict.keys()) - 1
    assert_that(NumberTranslation(3, "trzy"), is_not(is_in(unique_classes)))


def test_each_class_should_have_same_quantity():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    same_left_images = get_same_left_features(features, labels)
    c = collections.Counter(same_left_images)

    min_occurrence = min(c.values())
    max_occurrence = max(c.values())

    assert min_occurrence == max_occurrence


def test_each_class_should_have_proper_quantity(number_translation_features_dict):
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    same_left_images = get_same_left_features(features, labels)
    c = collections.Counter(same_left_images)

    expected_class_count = 2  # 3 classes, 3 examples each

    for e in c.values():
        assert e == expected_class_count


def get_same_left_features(features, labels):
    left_images = list(features.values())[0]
    same_labels = labels == 1
    same_left_images = left_images[same_labels]
    return [x.number for x in same_left_images]


def show_mnist_image(image):
    two_d = (np.reshape(image, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()
