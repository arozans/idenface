from typing import Tuple, Union, Iterable

import numpy as np
from dataclasses import replace

from src.data.common_types import DatasetSpec, DatasetType, DataDescription, DatasetStorageMethod, DatasetVariant, \
    ImageDimensions, RawDatasetFragment, DictsDataset
from src.estimator.launcher.launchers import RunData
from src.utils import consts, filenames
from testing_utils import testing_helpers, testing_consts, testing_classes
from testing_utils.testing_helpers import save_arrays_as_images_on_disc


def dataset_spec(
        description: DataDescription = None,
        raw_dataset_fragment: RawDatasetFragment = None,
        type=DatasetType.TRAIN,
        with_excludes=False,
        encoding=True,
        paired=True,
        repeating_pairs=True,
        identical_pairs=False,
):
    from testing_utils.testing_classes import FakeRawDataProvider
    return DatasetSpec(
        raw_data_provider=FakeRawDataProvider(description, raw_dataset_fragment, curated=True),
        type=type,
        with_excludes=with_excludes,
        encoding=encoding,
        paired=paired,
        repeating_pairs=repeating_pairs,
        identical_pairs=identical_pairs
    )


def dataset_desc(
        variant: DatasetVariant = None,
        image_dimensions: ImageDimensions = None,
        classes_count: int = None,
        storage_method: DatasetStorageMethod = None
):
    args = locals()
    not_none_args = {k: v for (k, v) in args.items() if v is not None}
    from testing_utils.testing_classes import FAKE_DATA_DESCRIPTION
    return replace(FAKE_DATA_DESCRIPTION, **not_none_args)


def paired_labels_dict(batch_size: int = 1):
    return {
        consts.PAIR_LABEL: labels(length=batch_size, classes=2),
        consts.LEFT_FEATURE_LABEL: labels(length=batch_size, classes=10),
        consts.RIGHT_FEATURE_LABEL: labels(length=batch_size, classes=10),
    }


def unpaired_labels_dict(batch_size: int = 1):
    return {
        consts.LABELS: labels(length=batch_size, classes=10)
    }


def random_str(length: int = 5):
    import random, string
    rand = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(length)])
    return rand


def features(size: Iterable,
             storage_method: DatasetStorageMethod = DatasetStorageMethod.IN_MEMORY,
             mimic_values=None,
             normalize=False):
    fake_random_images = np.random.uniform(size=size).astype(np.float32) - (0.5 if normalize else 0)
    if mimic_values is not None:
        for idx, label in enumerate(mimic_values):
            fake_random_images[idx][0] = label / 10
    if storage_method == DatasetStorageMethod.ON_DISC:
        return save_arrays_as_images_on_disc(fake_random_images, mimic_values)
    else:
        return fake_random_images


def labels(length: int, classes=10, curated=False):
    if curated:
        two_elems_of_each_class = list(np.arange(classes)) * 2
        remainder = np.random.randint(low=0, high=classes, size=length - 2 * classes).astype(np.int64)
        return np.concatenate((two_elems_of_each_class, remainder))
    else:
        return np.random.randint(low=0, high=classes, size=length).astype(np.int64)


def dicts_dataset(batch_size: int = 1,
                  image_dims: ImageDimensions = ImageDimensions(testing_consts.TEST_IMAGE_SIZE),
                  paired: bool = False,
                  normalize: bool = False,
                  save_on_disc: bool = False) -> Union[
    Tuple[DictsDataset, DictsDataset], DictsDataset]:
    gen_feats = lambda: (features(size=[batch_size, *image_dims], normalize=normalize))
    if paired:
        fake_images_data = {
            consts.LEFT_FEATURE_IMAGE: gen_feats(),
            consts.RIGHT_FEATURE_IMAGE: gen_feats(),
        }
        fake_images_labels = paired_labels_dict(batch_size=batch_size)
    else:
        fake_images_data = {
            consts.FEATURES: gen_feats(),
        }
        fake_images_labels = unpaired_labels_dict(batch_size=batch_size)
    image_dicts_dataset = DictsDataset(fake_images_data, fake_images_labels)
    if save_on_disc:
        path_dicts_dataset = testing_helpers.save_images_dict_on_disc(image_dicts_dataset.features,
                                                                      image_dicts_dataset.labels)
        return image_dicts_dataset, path_dicts_dataset
    return image_dicts_dataset


def run_data(model=None,
             launcher_name="launcher_name",
             runs_directory_name="runs_directory_name",
             is_experiment=False,
             run_no=1,
             models_count=1,
             with_model_dir=False):
    _run_data = RunData(model=model if model is not None else testing_classes.FakeModel(),
                        launcher_name=launcher_name,
                        runs_directory_name=runs_directory_name,
                        is_experiment=is_experiment,
                        run_no=run_no,
                        models_count=models_count,
                        launcher_params={})
    if with_model_dir:
        filenames.get_run_logs_data_dir(_run_data).mkdir(parents=True, exist_ok=True)
    return _run_data
