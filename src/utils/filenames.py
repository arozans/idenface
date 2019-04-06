import time
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING

from src.data.common_types import DatasetType, DatasetSpec
from src.utils import utils, consts
from src.utils.configuration import config

if TYPE_CHECKING:
    from src.estimator.launcher.launchers import RunData
    from src.estimator.model.estimator_model import EstimatorModel


def create_pairs_dataset_directory_name(dataset_spec: DatasetSpec) -> Optional[str]:
    dataset_type = dataset_spec.type
    if dataset_type == DatasetType.TRAIN or dataset_type == DatasetType.TEST:
        return _create_pairs_dataset_dir_name(dataset_spec)
    elif dataset_type == DatasetType.EXCLUDED:
        return None


def _create_pairs_dataset_dir_name(dataset_spec: DatasetSpec) -> str:
    excluded_fragment = create_excluded_name_fragment(with_prefix=True) if not dataset_spec.with_excludes else ''
    return dataset_spec.raw_data_provider_cls().description().variant.name.lower() + '_' + dataset_spec.type.value + excluded_fragment


def create_excluded_name_fragment(with_prefix: bool = False, with_suffix: bool = False) -> str:
    pattern = ''
    to_exclude = config.excluded_keys
    if to_exclude:
        pattern += 'ex_' + '-'.join(map(str, to_exclude))
        if with_prefix:
            pattern = '_' + pattern
        if with_suffix:
            pattern = pattern + '_'
    return pattern


def _create_date_name_fragment() -> str:
    return time.strftime('d%y%m%dt%H%M%S')


def _get_datetime_pattern():
    return r'^d\d{6}t\d{6}$'


def _get_home_directory() -> Path:
    return Path.home()


def _get_home_tf_directory() -> Path:
    return _get_home_directory() / consts.TF_DIR_SUFFIX


def get_all_text_logs_dir() -> Path:
    """~/tf/text_logs/"""
    return _get_home_tf_directory() / consts.TEXT_LOGS_DIR_SUFFIX


def get_infer_dir() -> Path:
    """~/tf/infer/"""
    return _get_home_tf_directory() / consts.INFER_DIR_SUFFIX


def get_input_data_dir() -> Path:
    """~/tf/datasets/"""
    return _get_home_tf_directory() / consts.INPUT_DATA_DIR_SUFFIX


def get_raw_input_data_dir() -> Path:
    """~/tf/datasets/raw/"""
    return get_input_data_dir() / consts.INPUT_DATA_RAW_DIR_SUFFIX


def get_processed_input_data_dir(encoding: bool) -> Path:
    """~/tf/datasets/paired/"""
    encoding_fragment = (consts.NOT_ENCODED_DIR_FRAGMENT if not encoding else '')
    return get_input_data_dir() / encoding_fragment / consts.INPUT_DATA_PAIRED_DIR_SUFFIX


def get_runs_dir(run_data: 'RunData') -> Path:
    """~/tf/runs/models or ~/tf/runs/experiments"""
    return _get_home_tf_directory() / consts.RUNS_DIR / run_data.runs_directory_name


def get_launcher_dir(run_data: 'RunData') -> Path:
    """~/tf/runs/models/standardCNN or ~/tf/runs/experiments/different_convolutions"""
    return get_runs_dir(run_data) / run_data.launcher_name


def get_run_dir(run_data: 'RunData') -> Path:
    """~/tf/runs/models/CNN/CNN-est_0.99_lr_0_ex_1-2-3/ or
    ~/tf/runs/experiments/different_convolutions/CNN-est_0.99_lr_0_ex_1-2-3/"""
    return get_launcher_dir(run_data) / utils.get_run_summary(run_data.model)


def get_run_logs_data_dir(run_data: 'RunData') -> Path:
    """~/tf/models/CNN/CNN-est_0.99_0.5_123/logs/"""
    return get_run_dir(run_data) / consts.LOGS_DIR_SUFFIX


def get_run_text_logs_dir(run_data: 'RunData') -> Path:
    """~/tf/models/CNN/CNN-est_0.99_0.5_123/text_logs/"""
    return get_run_dir(run_data) / consts.TEXT_LOGS_DIR_SUFFIX


def _with_suffix(name, suffix) -> str:
    return name + '.' + suffix


def summary_to_name(model, suffix: str, with_date_fragment: bool, name: str = '') -> str:
    run_summary = utils.get_run_summary(model)
    date_fragment = ('_' + _create_date_name_fragment()) if with_date_fragment else ''
    name_fragment = ('_' + name) if name else ''
    return _with_suffix(run_summary + name_fragment + date_fragment, suffix)


def create_text_log_name(model: 'EstimatorModel') -> str:
    return summary_to_name(model, suffix='log', with_date_fragment=True)


def create_infer_images_name(model: 'EstimatorModel') -> str:
    return summary_to_name(model, suffix='png')
