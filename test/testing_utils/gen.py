import numpy as np

from src.data.common_types import DatasetSpec, DatasetType
from src.estimator.launcher.launchers import RunData
from src.utils import consts, filenames
from testing_utils.testing_classes import FakeModel, CuratedFakeRawDataProvider


def run_data(model=FakeModel(),
             launcher_name="launcher_name",
             runs_directory_name="runs_directory_name",
             is_experiment=False,
             run_no=1,
             models_count=1,
             with_model_dir=False):
    _run_data = RunData(model=model,
                        launcher_name=launcher_name,
                        runs_directory_name=runs_directory_name,
                        is_experiment=is_experiment,
                        run_no=run_no,
                        models_count=models_count,
                        launcher_params={})
    if with_model_dir:
        filenames.get_run_logs_data_dir(_run_data).mkdir(parents=True, exist_ok=True)
    return _run_data


def dataset_spec(
        raw_data_provider_cls=CuratedFakeRawDataProvider,
        type=DatasetType.TRAIN,
        with_excludes=False,
        encoding=True,
        paired=True,
        repeating_pairs=True,
        identical_pairs=False
):
    return DatasetSpec(
        raw_data_provider_cls=raw_data_provider_cls,
        type=type,
        with_excludes=with_excludes,
        encoding=encoding,
        paired=paired,
        repeating_pairs=repeating_pairs,
        identical_pairs=identical_pairs
    )


def paired_labels_dict(pair_label: int = 1, left_label: int = 2, right_label: int = 3, batch_size: int = 1):
    return {
        consts.PAIR_LABEL: np.array([pair_label] * batch_size),
        consts.LEFT_FEATURE_LABEL: np.array([left_label] * batch_size),
        consts.RIGHT_FEATURE_LABEL: np.array([right_label] * batch_size),
    }


def labels_dict(label: int = 999, batch_size: int = 1):
    return {
        consts.LABELS: np.array([label] * batch_size),
    }
