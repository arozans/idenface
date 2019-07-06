from pathlib import Path

import pytest
from hamcrest import assert_that, only_contains

from estimator.training.integration.test_integration_training import FakeExperimentLauncher
from src.estimator.launcher.launchers import DefaultLauncher
from src.estimator.model.contrastive_model import MnistContrastiveModel
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.estimator.model.tba_model import ExtruderTBAModel
from src.utils import utils, filenames, consts
from src.utils.inference import inference
from testing_utils import gen
from testing_utils.testing_classes import FakeModel


@pytest.mark.integration
@pytest.mark.parametrize('patched_dataset_reading',
                         [
                             MnistSoftmaxModel,
                             MnistContrastiveModel,
                             ExtruderTBAModel
                         ],
                         ids=lambda x: str(x.description.variant),
                         indirect=True)
@pytest.mark.parametrize('patched_params', [{consts.IS_INFER_CHECKPOINT_OBLIGATORY: False}], indirect=True)
def test_should_create_summaries_for_different_models(patched_dataset_reading, patched_params):
    model = patched_dataset_reading.param
    run_data = gen.run_data(model=model())

    inference.single_run_inference(run_data=run_data, show=False)

    infer_results_dir_path = filenames.get_infer_dir(run_data)
    assert utils.check_filepath(infer_results_dir_path, is_directory=True, is_empty=False)


@pytest.mark.integration
@pytest.mark.parametrize('patched_dataset_reading',
                         [
                             MnistSoftmaxModel,
                             MnistContrastiveModel
                         ],
                         indirect=True)
@pytest.mark.parametrize('patched_params', [{consts.IS_INFER_CHECKPOINT_OBLIGATORY: False}], indirect=True)
def test_should_create_correct_number_of_inference_files(patched_dataset_reading, patched_params):
    model = patched_dataset_reading.param()
    run_data = gen.run_data(model=model)

    inference.single_run_inference(run_data=run_data, show=False)

    infer_results_dir_path = filenames.get_infer_dir(run_data)
    expected_file_count = 4 if model.produces_2d_embedding else 2
    assert utils.check_filepath(infer_results_dir_path, is_directory=True, is_empty=False,
                                expected_len=expected_file_count)


def test_should_throw_if_model_dir_not_exists():
    run_data = gen.run_data(with_model_dir=False)

    with pytest.raises(AssertionError):
        inference.single_run_inference(run_data=run_data, show=False)


def test_should_throw_if_model_has_no_checkpoints():
    run_data = gen.run_data(with_model_dir=True)

    with pytest.raises(AssertionError):
        inference.single_run_inference(run_data=run_data, show=False)


def test_should_throw_if_model_has_only_0_step_checkpoints():
    run_data = gen.run_data(with_model_dir=True)

    model_dir = filenames.get_run_logs_data_dir(run_data)
    empty_checkpoints = [model_dir / ("model.ckpt-0.foobar{}".format(x)) for x in range(5)]
    [f.write_text("this is sparta") for f in empty_checkpoints]
    with pytest.raises(AssertionError):
        inference.single_run_inference(run_data=run_data, show=False)


def test_should_not_throw_if_model_has_more_checkpoints():
    run_data = gen.run_data(with_model_dir=True)

    model_dir = filenames.get_run_logs_data_dir(run_data)
    empty_checkpoints = [(model_dir / ("model.ckpt-{}.foobar".format(x))) for x in range(2)]
    [f.write_text("this is sparta") for f in empty_checkpoints]
    inference.single_run_inference(run_data=run_data, show=False)


@pytest.mark.integration
@pytest.mark.parametrize('launcher', [
    DefaultLauncher([FakeModel()]),
    FakeExperimentLauncher([FakeModel(), FakeModel()]),
])
@pytest.mark.parametrize('patched_params', [{consts.IS_INFER_CHECKPOINT_OBLIGATORY: False}], indirect=True)
def test_should_run_inference_for_different_launchers(mocker, launcher, patched_params):
    mocker.patch('src.estimator.launcher.providing_launcher.provide_launcher', return_value=launcher)
    result = mocker.patch('builtins.input', return_value='0')
    inference.infer()
    if launcher.is_experiment:
        result.assert_called_once()
    else:
        result.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize('patched_dataset_reading', [MnistContrastiveModel], indirect=True)
@pytest.mark.parametrize('patched_params', [{consts.IS_INFER_CHECKPOINT_OBLIGATORY: False}], indirect=True)
def test_should_override_plots_with_newer_inference(patched_dataset_reading, patched_params):
    run_data = gen.run_data(model=MnistContrastiveModel())

    inference.single_run_inference(run_data=run_data, show=False)
    infer_results_dir_path = filenames.get_infer_dir(run_data)

    old_cr_times = [x.stat().st_ctime for x in list(Path(infer_results_dir_path).iterdir()) if x.suffix != consts.LOG]

    inference.single_run_inference(run_data=run_data, show=False)

    new_cr_times = [x.stat().st_ctime for x in list(Path(infer_results_dir_path).iterdir()) if x.suffix != consts.LOG]

    var = [(x not in new_cr_times) for x in old_cr_times]
    assert_that(var, only_contains(True))


@pytest.mark.integration
@pytest.mark.parametrize('patched_dataset_reading', [MnistContrastiveModel], indirect=True)
@pytest.mark.parametrize('patched_params', [{consts.IS_INFER_CHECKPOINT_OBLIGATORY: False}], indirect=True)
def test_should_not_override_old_inference_log(patched_dataset_reading, patched_params):
    run_data = gen.run_data(model=MnistContrastiveModel())

    inference.single_run_inference(run_data=run_data, show=False)
    infer_results_dir_path = filenames.get_infer_dir(run_data)

    old_cr_times = [x.stat().st_ctime for x in list(Path(infer_results_dir_path).iterdir()) if x.suffix == consts.LOG]

    inference.single_run_inference(run_data=run_data, show=False)

    new_cr_times = [x.stat().st_ctime for x in list(Path(infer_results_dir_path).iterdir()) if x.suffix == consts.LOG]

    assert len(old_cr_times) == 1
    assert len(new_cr_times) == 2

    assert old_cr_times[0] in new_cr_times
