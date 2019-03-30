from src.data.common_types import DataDescription, DatasetVariant

TF_DIR_SUFFIX: str = "tf"
INPUT_DATA_DIR_SUFFIX: str = "datasets"
INPUT_DATA_RAW_DIR_SUFFIX: str = "raw"
INPUT_DATA_PAIRED_DIR_SUFFIX: str = "paired"
LEFT_FEATURE_IMAGE: str = "left_image"
RIGHT_FEATURE_IMAGE: str = "right_image"
LOGS_DIR_SUFFIX: str = "logs"
TEXT_LOGS_DIR_SUFFIX: str = 'text_logs'
INFER_DIR_SUFFIX: str = "infer"
MODELS_DIR_SUFFIX: str = 'models'
RUNS_DIR: str = 'runs'
DATASET_PROVIDER_CLS: str = 'dataset_provider_cls'
RAW_DATA_PROVIDER_CLS: str = 'raw_data_provider_cls'
FULL_PROVIDER: str = 'full_provider'  # fixme!

MNIST_IMAGE_SIDE_PIXEL_COUNT: int = 28
MNIST_DATA_DESCRIPTION = DataDescription(variant=DatasetVariant.MNIST, image_side_length=MNIST_IMAGE_SIDE_PIXEL_COUNT,
                                         classes_count=10)

EXPERIMENT_LAUNCHER_RUNS_DIR_NAME = "experiments"
LEARNING_RATE = "learning_rate"
EXCLUDED_KEYS = "excluded_keys"
BATCH_SIZE = "batch_size"
OPTIMIZER = "optimizer"
TRAIN_STEPS = "train_steps"
EVAL_STEPS_INTERVAL = "eval_steps_interval"

INFER_IMAGE_COUNT = 25

MODEL_DIR = "model_dir"
NOT_ENCODED_FILENAME_MARKER = "raw"
NOT_ENCODED_DIR_FRAGMENT = "not_encoded"
