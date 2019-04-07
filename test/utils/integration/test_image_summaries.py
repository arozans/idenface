import numpy as np
import pytest

from helpers import gen
from helpers.fake_estimator_model import FakeModel
from helpers.test_helpers import FakeRawDataProvider
from src.data.raw_data.raw_data_providers import MnistRawDataProvider
from src.utils import utils, filenames, image_summaries, consts


@pytest.mark.integration
@pytest.mark.parametrize('patched_read_dataset',
                         [
                             MnistRawDataProvider,
                             FakeRawDataProvider
                         ],
                         ids=lambda x: str(x.description().variant),
                         indirect=True)
def test_create_pair_summaries(patched_read_dataset):
    provider = patched_read_dataset.param
    run_data = gen.run_data(model=FakeModel(data_provider=provider))
    dir_with_pair_summaries = filenames.get_run_logs_data_dir(run_data) / 'images'
    assert utils.check_filepath(dir_with_pair_summaries, exists=False)

    image_summaries.create_pair_summaries(run_data)

    assert utils.check_filepath(dir_with_pair_summaries, is_directory=True, is_empty=False)
    assert len(list(dir_with_pair_summaries.iterdir())) == 1


@pytest.mark.integration
@pytest.mark.parametrize('fake_dict_and_labels',
                         [
                             MnistRawDataProvider,
                             FakeRawDataProvider,
                         ],
                         ids=lambda x: str(x.description().variant),
                         indirect=True)
def test_should_create_pair_board_for_different_datasets(fake_dict_and_labels):
    dict_images, labels = fake_dict_and_labels

    infer_results_image_path = filenames._get_home_infer_dir() / "board.png"
    assert utils.check_filepath(infer_results_image_path, exists=False)

    image_summaries.create_pairs_board(features_dict=dict_images, labels_dict=labels,
                                       predicted_labels=labels[consts.PAIR_LABEL],
                                       predicted_scores=None, path=infer_results_image_path,
                                       show=False)  # switch to True to see generated board

    assert utils.check_filepath(infer_results_image_path, is_directory=False, is_empty=False)


def test_should_correct_map_pair_of_points_to_plot_data():
    left_points = np.array([(1, 2), (3, 4), (0, -1), (-4, -8)])
    right_points = np.array([(9, 0), (11, 4), (9, 3), (7, 6)])
    x, y = image_summaries.map_pair_of_points_to_plot_data(left_points, right_points)
    assert (x == np.array([(1, 3, 0, -4), (9, 11, 9, 7)])).all()
    assert (y == np.array([(2, 4, -1, -8), (0, 4, 3, 6)])).all()
