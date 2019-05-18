from pathlib import Path
from typing import Dict, Type

import numpy as np

from src.data.common_types import DatasetStorageMethod, DatasetSpec
from src.data.tfrecord.saving.tfrecord_savers import AbstractSaver, UnpairedFromMemoryRawBytesSaver, \
    PairedFromMemoryRawBytesSaver, PairedFromMemoryEncodingSaver, UnpairedFromMemoryEncodingSaver, \
    PairedFromDiscRawBytesSaver, UnpairedFromDiscRawBytesSaver
from src.utils import utils


def save_to_tfrecord(data_dict: Dict[str, np.ndarray], data_labels: Dict[str, np.ndarray], path: Path,
                     dataset_spec: DatasetSpec):
    utils.log('Saving .tfrecord file: {} using spec: {}'.format(path, dataset_spec))
    path.parent.mkdir(parents=True, exist_ok=True)
    storage_method = dataset_spec.raw_data_provider.description.storage_method

    if storage_method == DatasetStorageMethod.IN_MEMORY:
        saver = get_from_memory_saver(dataset_spec)
    elif storage_method == DatasetStorageMethod.ON_DISC:
        saver = get_from_disc_saver(dataset_spec)
    else:
        raise NotImplementedError("Storage method {} is not implemented ", storage_method)
    saver(dataset_spec).save(data_dict, data_labels, path)


def get_from_memory_saver(dataset_spec: DatasetSpec) -> Type[AbstractSaver]:
    tf_savers = {
        (False, False): UnpairedFromMemoryRawBytesSaver,
        (True, False): PairedFromMemoryRawBytesSaver,
        (True, True): PairedFromMemoryEncodingSaver,
        (False, True): UnpairedFromMemoryEncodingSaver
    }
    return tf_savers[(dataset_spec.paired, dataset_spec.encoding)]


def get_from_disc_saver(dataset_spec: DatasetSpec) -> Type[AbstractSaver]:
    if dataset_spec.paired:
        return PairedFromDiscRawBytesSaver
    else:
        return UnpairedFromDiscRawBytesSaver
