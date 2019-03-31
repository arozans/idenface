from pathlib import Path
from typing import List

import pytest

from helpers import test_helpers, gen, test_consts
from helpers.fake_estimator_model import FakeModel, MnistCNNModelWithGeneratedDataset
from src.estimator.launcher import providing_launcher
from src.estimator.launcher.experiments.standard_cnn_images_encoding import EncodingMnistCNNModel, \
    NoEncodingMnistCNNModel
from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.training import training, supplying_datasets
from src.utils import filenames, before_run
from src.utils.configuration import config


@pytest.fixture
def input_fn_spies(mocker):
    train_input_fn_spy = mocker.spy(supplying_datasets.AbstractDatasetProvider, 'train_input_fn')
    eval_input_fn_spy = mocker.spy(supplying_datasets.AbstractDatasetProvider, 'eval_input_fn')
    eval_with_excludes_fn_spy = mocker.spy(supplying_datasets.AbstractDatasetProvider, 'eval_with_excludes_input_fn')

    return train_input_fn_spy, eval_input_fn_spy, eval_with_excludes_fn_spy


class FakeParametrizedModel(FakeModel):

    def __init__(self, foo=0) -> None:
        super().__init__()
        self.foo = foo

    @property
    def summary(self) -> str:
        return '_'.join(list(str(x) for item in self.params.items() for x in item))

    @property
    def additional_model_params(self):
        return {
            'foo': self.foo
        }


@pytest.mark.integration
@pytest.mark.parametrize('patched_excluded', [([]), ([1, 2])], indirect=True)
@pytest.mark.parametrize('patched_read_dataset', [
    FakeModel,
], indirect=True)
def test_should_call_in_memory_evaluator_hooks(input_fn_spies,
                                               patched_read_dataset,
                                               patched_excluded):
    (train_input_fn_spy, eval_input_fn_spy, eval_with_excludes_fn_spy) = input_fn_spies
    run_data = gen.run_data(model=patched_read_dataset.param())
    before_run.prepare_env([], run_data)
    training.train(run_data)

    train_input_fn_spy.assert_called_once()
    eval_input_fn_spy.assert_called_once()
    assert eval_with_excludes_fn_spy.call_count == (1 if config.excluded_keys else 0)
    verify_log_directory(run_data, config.excluded_keys)
    assert run_data.model.model_fn_calls == (3 if config.excluded_keys else 2)


@pytest.mark.integration
@pytest.mark.parametrize('injected_raw_data_provider', [
    NoEncodingMnistCNNModel,
    EncodingMnistCNNModel,
    # MnistCNNModelWithTfRecordDataset,
    MnistCNNModelWithGeneratedDataset,
], indirect=True)
def test_should_train_with_each_model(injected_raw_data_provider):
    run_data = gen.run_data(model=injected_raw_data_provider())
    before_run.prepare_env([], run_data)
    training.train(run_data)

    verify_log_directory(run_data, config.excluded_keys)


class FakeExperimentLauncher(ExperimentLauncher):
    @property
    def launcher_name(self):
        return test_consts.FAKE_EXPERIMENT_LAUNCHER_NAME


@pytest.mark.integration
def test_should_train_with_all_experiment_models(mocker, patched_read_dataset):
    mocker.patch('src.utils.image_summaries.create_pair_summaries')
    models = [
        FakeParametrizedModel(1),
        FakeParametrizedModel(2)
    ]
    mocker.patch('src.estimator.launcher.providing_launcher.provide_launcher',
                 return_value=FakeExperimentLauncher(models))

    test_helpers.run_app()

    text_logs = []
    for run_data in providing_launcher.provide_launcher().runs_data:
        verify_log_directory(run_data, excluding=config.excluded_keys)
        text_logs.append(get_runs_text_logs(run_data))

    verify_text_logs_directory(text_logs, size=len(models))
    verify_text_logs_directory(get_all_text_dir_logs(), size=len(models))


def verify_text_logs_directory(text_logs: List[Path], size: int):
    assert len(text_logs) == size
    lines_num = len(text_logs[0].read_text().split('\n'))
    for e in text_logs[1:]:
        assert len(e.read_text().split('\n')) == lines_num


def get_all_text_dir_logs():
    return list(filenames.get_all_text_logs_dir().iterdir())


def get_runs_text_logs(run_data):
    return list(filenames.get_run_text_logs_dir(run_data).iterdir())[0]


def verify_log_directory(run_data, excluding=False):
    eval_dirs = 2 if excluding else 1
    _check_for_existence(run_data, 'graph.pbtxt')
    _check_for_existence(run_data, 'model.ckpt*')
    _check_for_existence(run_data, 'events*')
    _check_for_existence(run_data, 'eval*', quantity=eval_dirs)


def _check_for_existence(run_data, glob, quantity=None):
    files = list(filenames.get_run_logs_data_dir(run_data).glob(glob))
    assert len(files) != 0
    if quantity:
        assert len(files) == quantity
