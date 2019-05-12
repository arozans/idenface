from typing import Tuple, Union

import numpy as np
from dataclasses import replace

from src.data.common_types import DatasetSpec, DatasetType, DataDescription, DatasetStorageMethod, DatasetVariant, \
    ImageDimensions, DatasetFragment, DictsDataset
from src.estimator.launcher.launchers import RunData
from src.utils import consts, filenames
from testing_utils import testing_helpers, testing_consts
from testing_utils.testing_classes import FakeModel, CuratedFakeRawDataProvider, FAKE_DATA_DESCRIPTION


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
        description: DataDescription = None,
        dataset_fragment: DatasetFragment = None,
        type=DatasetType.TRAIN,
        with_excludes=False,
        encoding=True,
        paired=True,
        repeating_pairs=True,
        identical_pairs=False,
):
    # FIXME: RawDataProvider should take those two as constuctor arguments
    if description and dataset_fragment:
        class DataDescriptionAndDatasetFragmentOverriddenCuratedFakeRawOnProvider(CuratedFakeRawDataProvider):
            @staticmethod
            def description() -> DataDescription:
                return description

            def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
                return dataset_fragment.features, dataset_fragment.labels

        raw_data_provider_cls = DataDescriptionAndDatasetFragmentOverriddenCuratedFakeRawOnProvider
    elif description:
        class DataDescriptionOverriddenCuratedFakeRawOnProvider(CuratedFakeRawDataProvider):
            @staticmethod
            def description() -> DataDescription:
                return description

        raw_data_provider_cls = DataDescriptionOverriddenCuratedFakeRawOnProvider

    elif dataset_fragment:
        class DatasetFragmentOverriddenCuratedFakeRawOnProvider(CuratedFakeRawDataProvider):

            def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
                return dataset_fragment.features, dataset_fragment.labels

        raw_data_provider_cls = DatasetFragmentOverriddenCuratedFakeRawOnProvider
    else:
        raw_data_provider_cls = CuratedFakeRawDataProvider

    return DatasetSpec(
        raw_data_provider_cls=raw_data_provider_cls,
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
    return replace(FAKE_DATA_DESCRIPTION, **not_none_args)


def paired_labels_dict(pair_label: int = 1, left_label: int = 2, right_label: int = 3, batch_size: int = 1):
    return {
        consts.PAIR_LABEL: np.array([pair_label] * batch_size),
        consts.LEFT_FEATURE_LABEL: np.array([left_label] * batch_size),
        consts.RIGHT_FEATURE_LABEL: np.array([right_label] * batch_size),
    }


def unpaired_labels_dict(label: int = 999, batch_size: int = 1):
    return {
        consts.LABELS: np.array([label] * batch_size),
    }


def random_str(length: int = 5):
    import random, string
    rand = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(length)])
    return rand


def _random_images(batch_size: int = 1):  # move to conftest
    single_image_size = testing_consts.TEST_IMAGE_SIZE
    return np.random.uniform(size=[batch_size, *single_image_size]).astype(np.float32)


def images(batch_size: int = 1, paired: bool = False, save_on_disc: bool = False) -> Union[
    Tuple[DictsDataset, DictsDataset], DictsDataset]:
    if paired:
        left_images = _random_images(batch_size)  # fixme move that
        right_images = _random_images(batch_size)
        fake_images_data = {
            consts.LEFT_FEATURE_IMAGE: left_images,
            consts.RIGHT_FEATURE_IMAGE: right_images,
        }
        fake_images_labels = paired_labels_dict(batch_size=batch_size)
    else:
        images = _random_images(batch_size)
        fake_images_data = {
            consts.FEATURES: images,
        }
        fake_images_labels = unpaired_labels_dict(batch_size=batch_size)
    image_dicts_dataset = DictsDataset(fake_images_data, fake_images_labels)
    if save_on_disc:
        path_dicts_dataset = testing_helpers.save_images_dict_on_disc(image_dicts_dataset.features,
                                                                      image_dicts_dataset.labels)
        return image_dicts_dataset, path_dicts_dataset
    return image_dicts_dataset
