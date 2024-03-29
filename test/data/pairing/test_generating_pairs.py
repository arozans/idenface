import collections

import pytest
from hamcrest import assert_that, is_in

from src.data.processing.generating_pairs import create_same_pairs, create_different_pairs, determine_class_size, \
    get_random_element
from testing_utils import gen

DATASET_SPEC = gen.dataset_spec()

MIN_PAIRS_NUM = 30


@pytest.mark.parametrize('min_pairs_num, actual_pair_num', [(15, 15), (20, 21), (1000, 1002)])
def test_should_create_correct_pair_number(number_translation_features_dict, min_pairs_num, actual_pair_num):
    pairs, same_labels = create_same_pairs(number_translation_features_dict, min_pairs_num, DATASET_SPEC)

    assert len(pairs) == len(same_labels) == actual_pair_num


def test_should_create_same_pairs(number_translation_features_dict):
    pairs, _ = create_same_pairs(number_translation_features_dict, MIN_PAIRS_NUM, DATASET_SPEC)
    assert len(pairs) == MIN_PAIRS_NUM

    for left, right in pairs:
        assert left.number == right.number


def test_should_create_same_pairs_without_identical_pair_elements(number_translation_features_dict):
    pairs, _ = create_same_pairs(number_translation_features_dict, MIN_PAIRS_NUM, DATASET_SPEC)
    assert len(pairs) == MIN_PAIRS_NUM

    for left, right in pairs:
        assert left.number == right.number
        assert left.trans != right.trans


def test_should_create_same_pairs_without_repeating_pairs(number_translation_features_dict):
    pairs, _ = create_same_pairs(number_translation_features_dict, MIN_PAIRS_NUM,
                                 gen.dataset_spec(repeating_pairs=False))

    seen = set()
    for left, right in pairs:
        assert (left.trans, right.trans) not in seen
        seen.add((left.trans, right.trans))


def test_should_create_different_pairs(number_translation_features_dict):
    pairs, left_labels, right_labels = create_different_pairs(number_translation_features_dict, MIN_PAIRS_NUM)
    assert len(pairs) == MIN_PAIRS_NUM

    for left, right in pairs:
        assert left != right
    for a, b in zip(left_labels, right_labels):
        assert a != b


@pytest.mark.parametrize('min_pairs_num', [30, 55, 100])
def test_should_create_correct_number_of_same_pair_classes(number_translation_features_dict, min_pairs_num):
    pairs, _ = create_same_pairs(number_translation_features_dict, min_pairs_num, DATASET_SPEC)
    expected_frequency = min_pairs_num // len(number_translation_features_dict.keys())
    counter = collections.Counter([(x[0].number, x[1].number) for x in pairs])

    for freq in counter.values():
        assert freq >= expected_frequency


@pytest.mark.parametrize('min_pairs_num', [30, 55, 100])
def test_should_create_correct_number_of_different_pairs_classes(number_translation_features_dict, min_pairs_num):
    pairs, _, _ = create_different_pairs(number_translation_features_dict, min_pairs_num)
    # (1,3) and (3,1) are different pairs according to __eq__, so dividing by  2
    expected_frequency = (min_pairs_num / len(number_translation_features_dict.keys())) // 2
    counter = collections.Counter([(x[0].number, x[1].number) for x in pairs])

    for freq in counter.values():
        assert freq >= expected_frequency


def test_determine_class_size():
    class_size = determine_class_size(classes_num=3, pair_dataset_min_size=20)
    assert class_size == 7

    class_size = determine_class_size(classes_num=1, pair_dataset_min_size=20)
    assert class_size == 20

    class_size = determine_class_size(classes_num=8, pair_dataset_min_size=81)
    assert class_size == 11


def test_get_random_element():
    dataset = [0, 1, 2]
    nums = [get_random_element(dataset, exclude_elem=1) for _ in range(100)]
    assert len(nums) == 100
    assert_that(1, not (is_in(nums)))
