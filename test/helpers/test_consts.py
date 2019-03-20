from helpers.test_helpers import FakeRawDataProvider, TestDatasetVariant
from src.data.common_types import DatasetSpec, DatasetType, DataDescription
from src.data.raw_data.raw_data_providers import MnistRawDataProvider

DEBUG_MODE = False

FAKE_IMAGE_SIDE_PIXEL_COUNT = 2
MNIST_IMAGE_SIDE_PIXEL_COUNT = 28
FAKE_IMAGES_IN_DATASET_COUNT = 30
MNIST_IMAGES_CLASSES_COUNT = 10
FAKE_IMAGES_CLASSES_COUNT = 5
FAKE_IMAGE_PIXEL_COUNT = FAKE_IMAGE_SIDE_PIXEL_COUNT * FAKE_IMAGE_SIDE_PIXEL_COUNT

TEST_EXCLUDED_KEYS = []
TEST_EVAL_STEPS_INTERVAL = 300
TEST_TRAIN_STEPS = 3
TEST_BATCH_SIZE = 2

MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                                         type=DatasetType.TRAIN,
                                                         with_excludes=True)
MNIST_TRAIN_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                       type=DatasetType.TRAIN,
                                       with_excludes=False)
MNIST_TEST_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                      type=DatasetType.TEST,
                                      with_excludes=False)
MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES = DatasetSpec(raw_data_provider_cls=MnistRawDataProvider,
                                                        type=DatasetType.TEST, with_excludes=True)
FAKE_TRAIN_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=FakeRawDataProvider, type=DatasetType.TRAIN,
                                      with_excludes=False)

FAKE_TEST_DATASET_SPEC = DatasetSpec(raw_data_provider_cls=FakeRawDataProvider, type=DatasetType.TEST,
                                     with_excludes=False)

# noinspection PyTypeChecker
FAKE_DATA_DESCRIPTION = DataDescription(variant=TestDatasetVariant.FOO,
                                        image_side_length=FAKE_IMAGE_SIDE_PIXEL_COUNT,
                                        classes_count=FAKE_IMAGES_CLASSES_COUNT)

FAKE_EXPERIMENT_LAUNCHER_NAME = "fake-experiment"
STUB_LAUNCHER_RUNS_DIR_NAME = "STUB_LAUNCHER_RUNS_DIR_NAME"

PAIRING_WITH_IDENTICAL = True
