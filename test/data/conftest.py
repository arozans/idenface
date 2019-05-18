from typing import Tuple

import numpy as np
import pytest

from src.data.common_types import AbstractRawDataProvider, DataDescription, DatasetSpec, DatasetType
from testing_utils.testing_classes import TestDatasetVariant
from testing_utils.testing_helpers import NumberTranslation


@pytest.fixture()
def number_translation_features_dict():
    one1 = NumberTranslation(1, "jeden")
    one2 = NumberTranslation(1, "uno")
    one3 = NumberTranslation(1, "ein")
    one4 = NumberTranslation(1, "un")
    one5 = NumberTranslation(1, "hana")

    two1 = NumberTranslation(2, "dwa")
    two2 = NumberTranslation(2, "dos")
    two3 = NumberTranslation(2, "zwei")
    two4 = NumberTranslation(2, "deux")
    two5 = NumberTranslation(2, "tul")

    three1 = NumberTranslation(3, "trzy")
    three2 = NumberTranslation(3, "tres")
    three3 = NumberTranslation(3, "drei")
    three4 = NumberTranslation(3, "trois")
    three5 = NumberTranslation(3, "set")

    features = {
        'one': [one1, one2, one3, one4, one5],
        'two': [two1, two2, two3, two4, two5],
        'three': [three1, three2, three3, three4, three5]
    }
    return features


@pytest.fixture
def number_translation_features_and_labels(number_translation_features_dict):
    feature_and_label_pairs = []
    for key, value in number_translation_features_dict.items():
        for elem in value:
            feature_and_label_pairs.append((key, elem))
    labels, features = zip(*feature_and_label_pairs)
    return list(features), list(labels)


class NumberTranslationRawDataProvider(AbstractRawDataProvider):
    # noinspection PyTypeChecker
    @property
    def description(self) -> DataDescription:
        return DataDescription(TestDatasetVariant.NUMBERTRANSLATION, None, 3)

    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        return number_translation_features_and_labels(number_translation_features_dict())

    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return number_translation_features_and_labels(number_translation_features_dict())


TRANSLATIONS_TRAIN_DATASET_SPEC = DatasetSpec(raw_data_provider=NumberTranslationRawDataProvider(),
                                              type=DatasetType.TRAIN,
                                              with_excludes=False)
