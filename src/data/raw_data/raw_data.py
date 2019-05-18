from typing import Tuple

import numpy as np

from src.data.common_types import DatasetType, DatasetSpec


def get_raw_data(dataset_spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray]:
    raw_data_provider = dataset_spec.raw_data_provider
    dataset_type = dataset_spec.type
    if dataset_type == DatasetType.TRAIN:
        return raw_data_provider.get_raw_train()
    elif dataset_type == DatasetType.TEST:
        return raw_data_provider.get_raw_test()
