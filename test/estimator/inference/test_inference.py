import pytest

from helpers import gen
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import utils
from src.utils.inference import inference


@pytest.mark.integration
@pytest.mark.parametrize('patched_read_dataset',
                         [
                             MnistCNNModel,
                             MnistSiameseModel
                         ],
                         ids=lambda x: str(x.description().variant),
                         indirect=True)
def test_should_create_summaries_for_different_models(patched_read_dataset):
    model = patched_read_dataset.param
    run_data = gen.run_data(model=model())

    infer_results_dir_path = inference.run_inference(run_data=run_data, show=False)
    assert utils.check_filepath(infer_results_dir_path, is_directory=True, is_empty=False)


@pytest.mark.parametrize('patched_read_dataset',
                         [
                             MnistSiameseModel
                         ],
                         indirect=True)
def test_should_create_directory_for_inference(patched_read_dataset):
    run_data = gen.run_data(model=MnistSiameseModel())
    infer_results_dir_path = inference.run_inference(run_data=run_data, show=False)
    assert utils.check_filepath(infer_results_dir_path, is_directory=True, is_empty=False, expected_len=3)
