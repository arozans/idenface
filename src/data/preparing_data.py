from pathlib import Path
from typing import Callable, Dict

import numpy as np

from src.data.common_types import DatasetSpec
from src.data.pairing import creating_paired_data
from src.data.saving import saving_tfrecords
from src.utils import utils, filenames, consts
from src.utils.configuration import config


def not_empty(folder: Path) -> bool:
    contents = [x for x in folder.glob('*')]
    return len(contents) > 0


def find_or_create_paired_data_dir(dataset_spec: DatasetSpec) -> Path:
    utils.log('Searching for dataset: {}'.format(dataset_spec))
    dataset_dir: Path = filenames.get_processed_input_data_dir(dataset_spec.encoding)
    if dataset_dir.exists():
        matcher_fn: Callable[[str], bool] = get_dataset_dir_matcher_fn(dataset_spec)
        for folder in dataset_dir.glob('*'):
            if matcher_fn(folder.name) and not_empty(folder):
                utils.log('Dataset found: {} full path: {}'.format(folder.name, folder.resolve()))
                return folder

    return _create_paired_data_dir(dataset_spec)


def _create_paired_data_dir(dataset_spec: DatasetSpec) -> Path:
    utils.log('Creating new dataset: {}'.format(dataset_spec))
    dataset_dir = filenames.create_pairs_dataset_directory_name(dataset_spec)
    images, labels = creating_paired_data.create_paired_data(dataset_spec)
    full_dir_path = save_to_tfrecord(images, labels, dataset_dir, dataset_spec.encoding)
    return full_dir_path.parent


def save_to_tfrecord(images: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], dataset_dir: str, encoding: bool):
    tfrecord_full_path = _create_tfrecord_filename(dataset_dir, encoding)
    saving_tfrecords.save_to_tfrecord(images, labels, tfrecord_full_path, encoding)
    return tfrecord_full_path


def _create_tfrecord_filename(dataset_dir: str, encoding: bool) -> Path:
    full_dir_path = Path(filenames.get_processed_input_data_dir(encoding)) / dataset_dir
    full_dir_path.mkdir(parents=True, exist_ok=True)
    full_filename_path = full_dir_path / (str(full_dir_path.name) + (
        ('_' + consts.NOT_ENCODED_FILENAME_MARKER) if not encoding else ''))
    full_filename_path = full_filename_path.with_suffix('.tfrecord')
    return full_filename_path


def get_dataset_dir_matcher_fn(dataset_spec: DatasetSpec) -> Callable[[str], bool]:
    def matcher_fn(dir_name: str) -> bool:
        parts = dir_name.split('_')
        dataset_variant_part = parts[0]
        dataset_type_part = parts[1]
        if len(parts) > 2:
            excludes_keyword = parts[2]
            excludes_part = parts[3]
            excludes = excludes_part.split('-')
        else:
            excludes_keyword = None
            excludes = []
        # date_time_part = parts[-1]

        if dataset_spec.type.value != dataset_type_part:
            return False
        if dataset_spec.raw_data_provider_cls.description().variant.name.lower() != dataset_variant_part:
            return False
        # if not re.match(pattern=_get_datetime_pattern(), string=date_time_part):
        #     return False
        if excludes_keyword and excludes_keyword != 'ex':
            return False
        if dataset_spec.with_excludes:
            return not excludes
        else:
            return set(excludes) == set(map(str, config[consts.EXCLUDED_KEYS]))

    return matcher_fn
