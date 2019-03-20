import pytest

from helpers.test_consts import MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES, MNIST_TRAIN_DATASET_SPEC, \
    MNIST_TEST_DATASET_SPEC, MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES, FAKE_TRAIN_DATASET_SPEC
from src.data import preparing_data

CORRECT_NAME_DATASET_CONFIG_EXCLUDED_PARAMETERS = [
    ('mnist_train', MNIST_TRAIN_DATASET_SPEC, []),
    ('foo_train', FAKE_TRAIN_DATASET_SPEC, []),
    ('mnist_train_ex_3-2-4', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_test', MNIST_TEST_DATASET_SPEC, []),
    ('mnist_test_ex_1-2-3', MNIST_TEST_DATASET_SPEC, [1, 2, 3])
]


@pytest.mark.parametrize('correct_name, dataset_config, excluded', CORRECT_NAME_DATASET_CONFIG_EXCLUDED_PARAMETERS)
def test_should_match_correct_name(correct_name, dataset_config, excluded, mocker):
    assert match(dataset_config, excluded, mocker, correct_name)


DIFFERENT_ORDER_NAME_DATASET_CONFIG_EXCLUDED_PARAMETERS = [
    ('mnist_train_ex_3-2-4', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_2-4-3', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_4-2-3', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_2-3-4', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_3-4-2', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
]


@pytest.mark.parametrize('different_order_name, dataset_config, excluded',
                         DIFFERENT_ORDER_NAME_DATASET_CONFIG_EXCLUDED_PARAMETERS)
def test_should_match_correct_name_regardless_of_the_excluded_order(different_order_name, dataset_config, excluded,
                                                                    mocker):
    assert match(dataset_config, excluded, mocker, different_order_name)


incorrect_name_dataset_config_excluded_parameters = [
    ('fmnist_train', MNIST_TRAIN_DATASET_SPEC, []),
    ('mnist_train_d180711_t022345', MNIST_TRAIN_DATASET_SPEC, []),
    ('mnist_train', FAKE_TRAIN_DATASET_SPEC, []),
    ('mnist_unknownmode', MNIST_TRAIN_DATASET_SPEC, []),
    ('mnist_train_ex_3-2-4', MNIST_TRAIN_DATASET_SPEC, []),
    ('mnist_train', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_aa_4-2-3', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_3-2-5', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_3-4-2', MNIST_TEST_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_3-2-41', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train_ex_3-2_4', MNIST_TRAIN_DATASET_SPEC, [3, 2, 4]),
]


@pytest.mark.parametrize('incorrect_name, dataset_config, excluded',
                         incorrect_name_dataset_config_excluded_parameters)
def test_should_not_match_incorrect_name(incorrect_name, dataset_config, excluded,
                                         mocker):
    assert not match(dataset_config, excluded, mocker, incorrect_name)


CORRECT_NAME_DATASET_CONFIG_IGNORING_EXCLUDES_EXCLUDED_PARAMETERS = [
    ('mnist_train', MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES, []),
    ('mnist_test', MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES, [3, 2, 4]),
    ('mnist_train', MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES, [3, 2, 4, 7, 8, 9]),
]


@pytest.mark.parametrize('correct_name, dataset_config, excluded',
                         CORRECT_NAME_DATASET_CONFIG_IGNORING_EXCLUDES_EXCLUDED_PARAMETERS)
def test_should_include_excludes(correct_name, dataset_config, excluded,
                                 mocker):
    assert match(dataset_config, excluded, mocker, correct_name)


INCORRECT_NAME_DATASET_CONFIG_IGNORING_EXCLUDES_EXCLUDED_PARAMETERS = [
    ('mnist_test_ex_', MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES, []),
    ('mnist_test_ex_3-2-4', MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES, [3, 2, 4]),
    ('mnist_train_ex_3', MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES, [3, 2, 4, 7, 8, 9]),
]


@pytest.mark.parametrize('incorrect_name, dataset_config, excluded',
                         INCORRECT_NAME_DATASET_CONFIG_IGNORING_EXCLUDES_EXCLUDED_PARAMETERS)
def test_should_not_match_excludes_when_ignoring_excludes(incorrect_name, dataset_config, excluded,
                                                          mocker):
    assert not match(dataset_config, excluded, mocker, incorrect_name)


def match(dataset_config, excluded, mocker, name):
    mocker.patch('src.utils.configuration._excluded_keys', excluded)
    matcher_fn = preparing_data.get_dataset_dir_matcher_fn(dataset_config)
    return matcher_fn(name)
