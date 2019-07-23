import time
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING

from src.data.common_types import DatasetType, DatasetSpec, ImageDimensions, DatasetVariant
from src.utils import utils, consts
from src.utils.configuration import config

if TYPE_CHECKING:
    from src.estimator.launcher.launchers import RunData
    from src.estimator.model.estimator_conv_model import EstimatorConvModel


def create_dataset_directory_name(dataset_spec: DatasetSpec) -> Optional[str]:
    dataset_type = dataset_spec.type
    if dataset_type == DatasetType.TRAIN or dataset_type == DatasetType.TEST:
        return _create_dataset_dir_name(dataset_spec)
    elif dataset_type == DatasetType.EXCLUDED:
        return None


def _create_dataset_dir_name(dataset_spec: DatasetSpec) -> str:
    excluded_fragment = create_excluded_name_fragment(
        with_prefix=True) if not dataset_spec.with_excludes else consts.EMPTY_STR
    return dataset_spec.raw_data_provider.description.variant.name.lower() + '_' + dataset_spec.type.value + excluded_fragment


def create_excluded_name_fragment(with_prefix: bool = False, with_suffix: bool = False) -> str:
    pattern = ''
    to_exclude = config[consts.EXCLUDED_KEYS]
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


def _get_home_infer_dir() -> Path:
    """~/tf/infer/"""
    return _get_home_tf_directory() / consts.INFER_DIR_SUFFIX


def get_infer_dir(run_data: 'RunData') -> Path:
    return _get_home_infer_dir() / (
        run_data.launcher_name if run_data.is_experiment else consts.EMPTY_STR) / utils.get_run_summary(run_data.model)


def get_input_data_dir() -> Path:
    """~/tf/datasets/"""
    return _get_home_tf_directory() / consts.INPUT_DATA_DIR_SUFFIX


def get_raw_input_data_dir() -> Path:
    """~/tf/datasets/raw/"""
    return get_input_data_dir() / consts.INPUT_DATA_RAW_DIR_FRAGMENT


def _resolve_reduced_image_size_fragment(dataset_spec: DatasetSpec) -> str:
    demanded_image_dimensions = dataset_spec.raw_data_provider.description.image_dimensions
    sample_feature = dataset_spec.raw_data_provider.get_sample_feature()
    actual_image_dimensions = ImageDimensions.of(sample_feature)
    if demanded_image_dimensions == actual_image_dimensions:
        utils.log("Images to save of spec {} already have correct size: {}".format(dataset_spec,
                                                                                   actual_image_dimensions))
        return consts.EMPTY_STR
    else:
        utils.log("Images to save of spec {} have wrong size {}, will be reduced to {}".format(dataset_spec,
                                                                                               actual_image_dimensions,
                                                                                               demanded_image_dimensions))
        return consts.INPUT_DATA_REDUCED_IMAGE_SIZE_DIR_FRAGMENT + '_' + str(demanded_image_dimensions.width)


def get_processed_input_data_dir(dataset_spec: DatasetSpec) -> Path:
    """~/tf/datasets/[not_encoded]/[not_]paired/[size_XX]"""
    not_encoded_fragment = (
        consts.INPUT_DATA_NOT_ENCODED_DIR_FRAGMENT if not dataset_spec.encoding else consts.EMPTY_STR)
    paired_fragment = (
        consts.INPUT_DATA_NOT_PAIRED_DIR_FRAGMENT if not dataset_spec.paired else consts.INPUT_DATA_PAIRED_DIR_FRAGMENT)
    reduced_image_size_fragment = _resolve_reduced_image_size_fragment(dataset_spec)
    return get_input_data_dir() / not_encoded_fragment / paired_fragment / reduced_image_size_fragment


def get_runs_dir(run_data: 'RunData') -> Path:
    """~/tf/runs/models or ~/tf/runs/experiments"""
    return _get_home_tf_directory() / consts.RUNS_DIR / run_data.runs_directory_name


def get_launcher_dir(run_data: 'RunData') -> Path:
    """~/tf/runs/models/softmax or ~/tf/runs/experiments/different_convolutions"""
    runs_dir = get_runs_dir(run_data) / run_data.launcher_name
    if run_data.is_experiment:
        runs_dir = runs_dir.parent / (runs_dir.stem + utils.global_suffix_or_emtpy())
    return runs_dir


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


def get_sprites_filename(dataset_variant: DatasetVariant) -> Path:
    filename = dataset_variant.name.lower() + '_' + _create_date_name_fragment()
    full_filename = get_input_data_dir() / consts.SPRITES_DIR / _with_suffix(filename, consts.PNG)
    full_filename.parent.mkdir(exist_ok=True, parents=True)
    return full_filename


def _with_suffix(name, suffix) -> str:
    if suffix[0] != '.':
        suffix = '.' + suffix
    return name + suffix


def summary_to_name(model, suffix: str, with_date_fragment: bool, name: str = consts.EMPTY_STR) -> str:
    run_summary = utils.get_run_summary(model)
    date_fragment = ('_' + _create_date_name_fragment()) if with_date_fragment else consts.EMPTY_STR
    name_fragment = ('_' + name) if name else consts.EMPTY_STR
    return _with_suffix(run_summary + name_fragment + date_fragment, suffix)


def create_text_log_name(model: 'EstimatorConvModel') -> str:
    return summary_to_name(model, suffix=consts.LOG, with_date_fragment=True)


def create_infer_log_name(model: 'EstimatorConvModel') -> str:
    return summary_to_name(model, suffix=consts.LOG, with_date_fragment=True, name="inference")
