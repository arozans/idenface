from collections import OrderedDict

import pytest
from hamcrest import assert_that, is_, is_not, is_in

from data.conftest import TRANSLATIONS_TRAIN_DATASET_SPEC
from src.data.processing import creating_unpaired_data
from src.utils import consts
from testing_utils.testing_helpers import NumberTranslation


def test_should_create_labels_and_dict_without_pairs():
    features, labels = creating_unpaired_data.create_unpaired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    assert_that(features, is_(OrderedDict))
    assert len(features.keys()) == 1

    assert_that(labels, is_(OrderedDict))
    assert len(labels.keys()) == 1


def test_features_should_have_correct_key_names():
    features, _ = creating_unpaired_data.create_unpaired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    keys = list(features.keys())
    assert keys[0] == consts.FEATURES


def test_labels_should_have_correct_key_names():
    _, labels = creating_unpaired_data.create_unpaired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    keys = list(labels.keys())
    assert keys[0] == consts.LABELS


def test_features_should_have_same_length_as_raw_data(number_translation_features_dict):
    features, labels = creating_unpaired_data.create_unpaired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    values = list(features.values())
    labels = list(labels.values())
    assert len(values[0]) == len(list(x for y in list(number_translation_features_dict.values()) for x in y)) == len(
        labels[0])


def test_digit_labels_should_correspond_to_features(number_translation_features_dict):
    features, labels = creating_unpaired_data.create_unpaired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)
    features = features[consts.FEATURES]
    labels = labels[consts.LABELS]

    for feat, label in zip(features, labels):
        assert feat in number_translation_features_dict[label]


def test_each_class_should_be_included(number_translation_features_dict):
    features, labels = creating_unpaired_data.create_unpaired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    unique_classes = set([x.number for x in features[consts.FEATURES].flat])

    assert len(unique_classes) == len(number_translation_features_dict.keys())


@pytest.mark.parametrize('patched_excluded', [['three']], indirect=True)
def test_should_honor_excludes(number_translation_features_dict, patched_excluded):
    features, labels = creating_unpaired_data.create_unpaired_data(TRANSLATIONS_TRAIN_DATASET_SPEC)

    unique_classes = set([x.number for x in features[consts.FEATURES].flat])

    assert len(unique_classes) == len(number_translation_features_dict.keys()) - 1
    assert_that(NumberTranslation(3, "trzy"), is_not(is_in(unique_classes)))
