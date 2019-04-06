import pytest

from helpers import gen
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import utils, filenames
from src.utils.inference import inference


@pytest.mark.integration
@pytest.mark.parametrize('patched_read_dataset',
                         [
                             MnistCNNModel,
                             MnistSiameseModel
                         ],
                         ids=lambda x: str(x.description().variant),
                         indirect=True)
def test_should_create_summaries_for_different_models(mocker, patched_read_dataset):
    mocker.patch('src.utils.configuration._is_infer_checkpoint_obligatory', False)

    model = patched_read_dataset.param
    run_data = gen.run_data(model=model())

    infer_results_dir_path = inference.run_inference(run_data=run_data, show=False)
    assert utils.check_filepath(infer_results_dir_path, is_directory=True, is_empty=False)


@pytest.mark.parametrize('patched_read_dataset',
                         [
                             MnistSiameseModel
                         ],
                         indirect=True)
def test_should_create_directory_for_inference(mocker, patched_read_dataset):
    mocker.patch('src.utils.configuration._is_infer_checkpoint_obligatory', False)

    run_data = gen.run_data(model=MnistSiameseModel())
    infer_results_dir_path = inference.run_inference(run_data=run_data, show=False)
    assert utils.check_filepath(infer_results_dir_path, is_directory=True, is_empty=False, expected_len=3)


def test_should_throw_if_model_dir_not_exists():
    run_data = gen.run_data(with_model_dir=False)

    with pytest.raises(AssertionError):
        inference.run_inference(run_data=run_data, show=False)


def test_should_throw_if_model_has_no_checkpoints():
    run_data = gen.run_data(with_model_dir=True)

    with pytest.raises(AssertionError):
        inference.run_inference(run_data=run_data, show=False)


def test_should_throw_if_model_has_only_0_step_checkpoints():
    run_data = gen.run_data(with_model_dir=True)

    model_dir = filenames.get_run_logs_data_dir(run_data)
    empty_checkpoints = [model_dir / ("model.ckpt-0.foobar{}".format(x)) for x in range(5)]
    [f.write_text("this is sparta") for f in empty_checkpoints]
    with pytest.raises(AssertionError):
        inference.run_inference(run_data=run_data, show=False)


def test_should_not_throw_if_model_has_more_checkpoints():
    run_data = gen.run_data(with_model_dir=True)

    model_dir = filenames.get_run_logs_data_dir(run_data)
    empty_checkpoints = [(model_dir / ("model.ckpt-{}.foobar".format(x))) for x in range(2)]
    [f.write_text("this is sparta") for f in empty_checkpoints]
    inference.run_inference(run_data=run_data, show=False)
