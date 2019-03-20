import pytest

from helpers import gen
from src.estimator import inference
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import utils, filenames


@pytest.mark.integration
@pytest.mark.parametrize('patched_read_dataset',
                         [
                             MnistCNNModel,
                             MnistSiameseModel
                         ],
                         ids=lambda x: str(x.description().variant),
                         indirect=True)
def test_should_create_board_summaries_for_different_models(patched_read_dataset):
    model = patched_read_dataset.param
    infer_results_image_path = filenames.get_infer_dir() / "abc.png"
    run_data = gen.run_data(model=model())

    assert utils.check_filepath(infer_results_image_path, exists=False)
    inference.run_inference(run_data=run_data, predicted_images_path=infer_results_image_path, show=False)
    assert utils.check_filepath(infer_results_image_path, is_directory=False, is_empty=False)
