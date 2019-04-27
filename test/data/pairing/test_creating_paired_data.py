import collections
from collections import OrderedDict

import numpy as np
import pytest
from hamcrest import assert_that, is_, is_not, is_in
from matplotlib import pyplot as plt

from data.conftest import TRANSLATIONS_TRAIN_DATASET_SPEC
from src.data.processing import creating_paired_data
from src.utils import consts
from testing_utils.testing_helpers import NumberTranslation


def test_should_create_labels_and_dict_with_pairs():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    assert_that(features, is_(OrderedDict))
    assert len(features.keys()) == 2

    assert_that(labels, is_(OrderedDict))
    assert len(labels.keys()) == 3


def test_features_should_have_correct_key_names():
    features, _ = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    keys = list(features.keys())
    assert keys[0] == consts.LEFT_FEATURE_IMAGE
    assert keys[1] == consts.RIGHT_FEATURE_IMAGE


def test_labels_should_have_correct_key_names():
    _, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    keys = list(labels.keys())
    assert keys[0] == consts.LEFT_FEATURE_LABEL
    assert keys[1] == consts.RIGHT_FEATURE_LABEL
    assert keys[2] == consts.PAIR_LABEL


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
    counter = collections.Counter(labels[consts.PAIR_LABEL])

    assert len(set(counter.values())) == 1


def test_digit_labels_shout_correspond_to_features(number_translation_features_dict):
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    features_left = features[consts.LEFT_FEATURE_IMAGE]
    left_labels = labels[consts.LEFT_FEATURE_LABEL]
    features_right = features[consts.RIGHT_FEATURE_IMAGE]
    right_labels = labels[consts.RIGHT_FEATURE_LABEL]

    for feat, label in zip(list(features_left) + list(features_right), list(left_labels) + list(right_labels)):
        assert feat in number_translation_features_dict[label]


def test_same_pairs_should_consists_of_same_class():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    pair_labels = labels[consts.PAIR_LABEL]
    same_features_left = features[consts.LEFT_FEATURE_IMAGE][pair_labels == 1]
    same_features_right = features[consts.RIGHT_FEATURE_IMAGE][pair_labels == 1]

    for e, f in zip(same_features_left, same_features_right):
        assert (e.number == f.number)


def test_same_pairs_should_have_same_digit_label():
    _, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    pair_labels = labels[consts.PAIR_LABEL]
    left_labels = labels[consts.LEFT_FEATURE_LABEL]
    right_labels = labels[consts.RIGHT_FEATURE_LABEL]
    same_labels_left = left_labels[pair_labels == 1]
    same_labels_right = right_labels[pair_labels == 1]

    for e, f in zip(same_labels_left, same_labels_right):
        assert (e == f)


def test_different_pairs_should_consists_of_different_classes():
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    pair_labels = labels[consts.PAIR_LABEL]

    same_features_left = features[consts.LEFT_FEATURE_IMAGE][pair_labels == 0]
    same_features_right = features[consts.RIGHT_FEATURE_IMAGE][pair_labels == 0]

    for e, f in zip(same_features_left, same_features_right):
        assert (e.number != f.number)


def test_different_pairs_should_have_different_digit_label():
    _, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    pair_labels = labels[consts.PAIR_LABEL]
    left_labels = labels[consts.LEFT_FEATURE_LABEL]
    right_labels = labels[consts.RIGHT_FEATURE_LABEL]
    same_labels_left = left_labels[pair_labels == 0]
    same_labels_right = right_labels[pair_labels == 0]

    for e, f in zip(same_labels_left, same_labels_right):
        assert (e != f)


def test_each_class_should_be_included(number_translation_features_dict):
    features, labels = creating_paired_data.create_paired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    same_left_images = get_same_left_features(features, labels)
    unique_classes = set(same_left_images)

    assert len(unique_classes) == len(number_translation_features_dict.keys())


@pytest.mark.parametrize('patched_excluded', [['three']], indirect=True)
def test_should_honor_excludes(number_translation_features_dict, patched_excluded):
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

    expected_class_count = 3  # 3 classes, 5 examples each

    for e in c.values():
        assert e == expected_class_count


def get_same_left_features(features, labels):
    left_images = list(features.values())[0]
    same_labels = labels[consts.PAIR_LABEL] == 1
    same_left_images = left_images[same_labels]
    return [x.number for x in same_left_images]


def show_mnist_image(image):
    two_d = (np.reshape(image, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()
