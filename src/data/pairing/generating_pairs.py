from typing import Dict, List, Optional, TypeVar, Tuple

import numpy as np

T = TypeVar('T', np.ndarray, int)


def create_same_pairs(
        labeled_features: Dict[int, List[np.ndarray]],
        min_pairs_num: int,
        with_identical: bool

) -> List[Tuple[np.ndarray, np.ndarray]]:
    class_size = determine_class_size(len(labeled_features.keys()), min_pairs_num)

    same_pairs = []
    for class_examples in labeled_features.values():
        for _ in range(class_size):
            elem_left = get_random_element(class_examples)
            elem_right = get_random_element(class_examples,
                                            exclude_elem=None if with_identical else elem_left)
            same_pairs.append((elem_left, elem_right))

    return same_pairs


def create_different_pairs(labeled_features: Dict[int, List[np.ndarray]],
                           min_pairs_num: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    classes_num = len(labeled_features.keys())
    class_size = determine_class_size(classes_num, min_pairs_num)

    different_pairs = []

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

        for _ in range(inner_class_remainder):
            elem_left = get_random_element(left_examples)
            random_key = get_random_element(list(labeled_features), exclude_elem=left_label)
            elem_right = get_random_element(labeled_features[random_key])
            different_pairs.append((elem_left, elem_right))

    return different_pairs


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
