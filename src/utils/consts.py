from typing import Tuple

from src.data.common_types import DataDescription, DatasetVariant

TF_DIR_SUFFIX: str = "tf"
INPUT_DATA_DIR_SUFFIX: str = "datasets"
INPUT_DATA_RAW_DIR_SUFFIX: str = "raw"
INPUT_DATA_PAIRED_DIR_SUFFIX: str = "paired"

LEFT_FEATURE_IMAGE: str = "left_image"
LEFT_FEATURE_LABEL: str = "left_label"
RIGHT_FEATURE_IMAGE: str = "right_image"
RIGHT_FEATURE_LABEL: str = "right_label"
PAIR_LABEL: str = "pair_label"

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

EXPERIMENT_LAUNCHER_RUNS_DIR_NAME: str = "experiments"
DEFAULT_LAUNCHER_RUNS_DIR_NAME: str = "models"

LEARNING_RATE: str = "learning_rate"
EXCLUDED_KEYS: str = "excluded_keys"
BATCH_SIZE: str = "batch_size"
OPTIMIZER: str = "optimizer"
TRAIN_STEPS: str = "train_steps"
EVAL_STEPS_INTERVAL: str = "eval_steps_interval"
PREDICT_MARGIN: str = "predict_margin"
TRAIN_MARGIN: str = "train_margin"

INFER_IMAGE_COUNT: int = 25

MODEL_DIR: str = "model_dir"
NOT_ENCODED_FILENAME_MARKER: str = "raw"
NOT_ENCODED_DIR_FRAGMENT: str = "not_encoded"

TFRECORD_LEFT_BYTES: str = "left_bytes"
TFRECORD_RIGHT_BYTES: str = "right_bytes"
TFRECORD_PAIR_LABEL: str = "pair_label"
TFRECORD_LEFT_LABEL: str = "left_label"
TFRECORD_RIGHT_LABEL: str = "right_label"
TFRECORD_HEIGHT: str = "height"
TFRECORD_WEIGHT: str = "weight"
TFRECORD_DEPTH: str = "depth"

INFERENCE_CLASSES: str = "classes"
INFERENCE_SOFTMAX_PROBABILITIES: str = "probabilities"
INFERENCE_DISTANCES: str = "distances"
INFERENCE_LEFT_EMBEDDINGS: str = "left_embeddings"
INFERENCE_RIGHT_EMBEDDINGS: str = "right_embeddings"

METRIC_ACCURACY: str = "accuracy"
METRIC_RECALL: str = "recall"
METRIC_PRECISION: str = "precision"
METRIC_F1: str = "f1"
METRIC_MEAN_DISTANCE: str = "mean_distance"

INFER_FIG_SIZE: Tuple[float, float] = (16, 10)

INFER_PLOT_BOARD_NAME: str = "board"
INFER_PLOT_DISTANCES_NAME: str = "distances"
INFER_PLOT_CLUSTERS_NAME: str = "clusters"

INFER_PLOT_COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'white', 'purple', 'pink',
                     'maroon', 'orange', 'teal', 'coral', 'darkgoldenrod', 'lime', 'chocolate',
                     'turquoise', 'darkslategray', 'tan', 'salmon', 'indigo', 'hotpink', 'olive', 'navy', 'dimgray']

PNG: str = 'png'
LOG: str = 'log'
TXT: str = 'txt'
