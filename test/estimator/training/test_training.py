from hamcrest import assert_that, contains

from estimator.training.integration.test_integration_training import FakeExperimentLauncher
from helpers import gen
from helpers.fake_estimator_model import FakeModel
from src.estimator.launcher.launchers import DefaultLauncher
from src.estimator.training import training
from src.utils import consts


def prepare_mocks(launcher, mocker):
    mocker.patch('src.estimator.launcher.providing_launcher.provide_launcher', return_value=launcher)
    before_run_mock = mocker.patch('src.utils.before_run.prepare_env', autospec=True)
    mocker.patch('src.estimator.training.training.create_estimator', autospec=True)
    in_memory_training_mock = mocker.patch('src.estimator.training.training.in_memory_train_eval', autospec=True)
    return before_run_mock, in_memory_training_mock


def test_training_in_memory_only(mocker):
    launcher = DefaultLauncher([
        FakeModel()
    ])
    before_run_mock, in_memory_training_mock = prepare_mocks(launcher, mocker)

    distributed_training_mock = mocker.patch('src.estimator.training.training.distributed_train_eval', autospec=True)
    import sys

    training.main(sys.argv)

    before_run_mock.assert_called_once()
    in_memory_training_mock.assert_called_once()
    distributed_training_mock.assert_not_called()


def test_should_train_with_each_model(mocker):
    cnn_model_uno = FakeModel()
    cnn_model_dos = FakeModel()
    launcher = FakeExperimentLauncher([
        cnn_model_uno,
        cnn_model_dos
    ])

    before_run_mock, in_memory_training_mock = prepare_mocks(launcher, mocker)
    import sys

    training.main(sys.argv)

    assert before_run_mock.call_count == 2
    assert in_memory_training_mock.call_count == 2
    model_arguments = get_model_called_with(before_run_mock)
    assert_that(model_arguments,
                contains(*[x for x in launcher.runs_data]))


def get_model_called_with(before_run_mock):
    return [x[0][-1] for x in before_run_mock.call_args_list]


def test_should_pass_models_dataset_provider_to_model():
    model = FakeModel()
    run_data = gen.run_data(model)
    estimator = training.create_estimator(run_data)
    data_provider_cls = estimator.params[consts.DATASET_PROVIDER_CLS]
    assert data_provider_cls == model.dataset_provider_cls


# fixme - should pass full provider object
def test_should_pass_models_raw_data_provider_to_model():
    model = FakeModel()
    run_data = gen.run_data(model)
    estimator = training.create_estimator(run_data)
    data_provider_cls = estimator.params[consts.RAW_DATA_PROVIDER_CLS]
    assert data_provider_cls == model.raw_data_provider_cls
