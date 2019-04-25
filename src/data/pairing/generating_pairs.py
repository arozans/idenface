from typing import Dict, List, Optional, TypeVar, Tuple

import numpy as np

import src.data.common_types

T = TypeVar('T', np.ndarray, int)


def create_same_pairs(labeled_features: Dict[int, List[np.ndarray]],
                      min_pairs_num: int,
                      dataset_spec: src.data.common_types.DatasetSpec
                      ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    class_size = determine_class_size(len(labeled_features.keys()), min_pairs_num)

    same_pairs = []
    seen = set()

    labels = []
    for class_label, class_examples in labeled_features.items():
        for _ in range(class_size):
            elem_left, elem_right = get_random_pair(class_examples, dataset_spec.identical_pairs)

            if not dataset_spec.repeating_pairs:
                while make_hashable(elem_left, elem_right) in seen:
                    elem_left, elem_right = get_random_pair(class_examples, dataset_spec.identical_pairs)
                seen.add((make_hashable(elem_left, elem_right)))

            same_pairs.append((elem_left, elem_right))
            labels.append(class_label)

    return same_pairs, labels


def make_hashable(elem_left, elem_right):
    return str(elem_left), str(elem_right)


def get_random_pair(class_examples, with_identical_pairs: bool):
    idxs = np.random.choice(len(class_examples), 2, replace=with_identical_pairs)
    (elem_left, elem_right) = np.array(class_examples)[idxs]
    return elem_left, elem_right


def create_different_pairs(labeled_features: Dict[int, List[np.ndarray]],
                           min_pairs_num: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int], List[int]]:
    classes_num = len(labeled_features.keys())
    class_size = determine_class_size(classes_num, min_pairs_num)

    different_pairs = []
    left_labels = []
    right_labels = []

    inner_class_size = class_size // (classes_num - 1)
    inner_class_remainder = class_size % (classes_num - 1)

    for (left_label, left_examples) in labeled_features.items():
        for (right_label, right_examples) in labeled_features.items():
            if left_label == right_label:
                continue

            for _ in range(inner_class_size):
                elem_left = get_random_element(left_examples)
                elem_right = get_random_element(right_examples)
                different_pairs.append((elem_left, elem_right))
                left_labels.append(left_label)
                right_labels.append(right_label)

        for _ in range(inner_class_remainder):
            elem_left = get_random_element(left_examples)
            random_key = get_random_element(list(labeled_features), exclude_elem=left_label)
            elem_right = get_random_element(labeled_features[random_key])
            different_pairs.append((elem_left, elem_right))
            left_labels.append(left_label)
            right_labels.append(random_key)

    return different_pairs, left_labels, right_labels


def determine_class_size(classes_num: int, pair_dataset_min_size: int) -> int:
    class_size = pair_dataset_min_size // classes_num
    if pair_dataset_min_size % classes_num != 0:
        class_size += 1
    return class_size


def get_random_element(class_examples: List[T], exclude_elem: Optional[T] = None) -> T:
    class_examples = list(class_examples)
    elem = _get_random(class_examples)
    if exclude_elem is not None:
        while np.array(elem == exclude_elem).all():
            elem = _get_random(class_examples)

    return elem


def _get_random(data):
    index = np.random.choice(len(data))
    elem = data[index]
    return elem
