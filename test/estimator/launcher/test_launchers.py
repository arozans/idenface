import pytest

from estimator.training.integration.test_integration_training import FakeExperimentLauncher
from helpers import test_consts
from src.estimator.launcher.launchers import DefaultLauncher
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.utils import consts


@pytest.fixture
def default_launcher():
    return DefaultLauncher([MnistCNNModel()])


experiment_models = [MnistCNNModel(), MnistCNNModel()]


@pytest.fixture
def experiment_launcher():
    return FakeExperimentLauncher(experiment_models)


def test_default_launcher_is_not_experiment_one(default_launcher):
    assert default_launcher.is_experiment is False


def test_default_launcher_should_use_model_name_as_run_directory(default_launcher):
    assert default_launcher.runs_directory_name == 'models'


def test_default_launcher_should_use_model_name_as_its_name(default_launcher):
    assert default_launcher.launcher_name == default_launcher.models[0].name


def test_default_launcher_should_create_correct_run_data(default_launcher):
    run_datas = default_launcher.runs_data
    assert len(run_datas) == 1
    run_data = run_datas[0]
    assert run_data.is_experiment is False
    assert run_data.launcher_name == default_launcher.launcher_name
    assert run_data.model == default_launcher.models[0]
    assert run_data.runs_directory_name == default_launcher.runs_directory_name
    assert run_data.run_no == 1


def test_should_throw_when_creating_default_launcher_with_more_than_one_model():
    with pytest.raises(ValueError):
        DefaultLauncher([MnistCNNModel(), MnistCNNModel()])


def test_experiment_launcher_has_is_experiment_property_set(experiment_launcher):
    assert experiment_launcher.is_experiment is True


def test_experiment_launcher_should_use_experiment_runs_directory(experiment_launcher):
    assert experiment_launcher.runs_directory_name == consts.EXPERIMENT_LAUNCHER_RUNS_DIR_NAME


def test_experiment_launcher_should_use_its_name_as_run_directory(experiment_launcher):
    assert experiment_launcher.launcher_name == test_consts.FAKE_EXPERIMENT_LAUNCHER_NAME


def test_experiment_launcher_should_create_correct_run_data(experiment_launcher):
    run_datas = experiment_launcher.runs_data
    assert len(run_datas) == 2
    for idx, run_data in enumerate(run_datas):
        assert run_data.is_experiment is True
        assert run_data.launcher_name == experiment_launcher.launcher_name
        assert run_data.model == experiment_models[idx]
        assert run_data.runs_directory_name == experiment_launcher.runs_directory_name
        assert run_data.run_no == idx + 1
