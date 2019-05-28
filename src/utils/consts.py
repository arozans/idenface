from typing import Tuple

LEARNING_RATE: str = "learning_rate"
EXCLUDED_KEYS: str = "excluded_keys"
BATCH_SIZE: str = "batch_size"
OPTIMIZER: str = "optimizer"
TRAIN_STEPS: str = "train_steps"
EVAL_STEPS_INTERVAL: str = "eval_steps_interval"
SHUFFLE_BUFFER_SIZE: str = "shuffle_buffer_size"
REMOVE_OLD_MODEL_DIR: str = "remove_old_model_dir"
GLOBAL_SUFFIX: str = "global_suffix"
IS_INFER_CHECKPOINT_OBLIGATORY: str = "is_infer_checkpoint_obligatory"
PREDICT_SIMILARITY_MARGIN: str = "predict_similarity_margin"
TRAIN_SIMILARITY_MARGIN: str = "train_similarity_margin"
GRADIENT_DESCEND_OPTIMIZER: str = "GradientDescent"
MOMENTUM_OPTIMIZER: str = "Momentum"
NESTEROV_OPTIMIZER: str = "Nesterov"
ADAM_OPTIMIZER: str = "Adam"
FILTERS: str = "filters"
KERNEL_SIDE_LENGTHS: str = "kernel_side_lengths"

TF_DIR_SUFFIX: str = "tf"
INPUT_DATA_DIR_SUFFIX: str = "datasets"
INPUT_DATA_PAIRED_DIR_FRAGMENT: str = "paired"
INPUT_DATA_RAW_DIR_FRAGMENT: str = "raw"
INPUT_DATA_NOT_ENCODED_DIR_FRAGMENT: str = "not_encoded"
INPUT_DATA_NOT_PAIRED_DIR_FRAGMENT: str = "not_paired"
INPUT_DATA_REDUCED_IMAGE_SIZE_DIR_FRAGMENT: str = "size"

LEFT_FEATURE_IMAGE: str = "left_image"
LEFT_FEATURE_LABEL: str = "left_label"
RIGHT_FEATURE_IMAGE: str = "right_image"
RIGHT_FEATURE_LABEL: str = "right_label"
PAIR_LABEL: str = "pair_label"

FEATURES: str = "features"
LABELS: str = "labels"

LOGS_DIR_SUFFIX: str = "logs"
TEXT_LOGS_DIR_SUFFIX: str = 'text_logs'
INFER_DIR_SUFFIX: str = "infer"
MODELS_DIR_SUFFIX: str = 'models'
SPRITES_DIR: str = 'sprites'
RUNS_DIR: str = 'runs'

MNIST_IMAGE_SIDE_PIXEL_COUNT: int = 28
MNIST_IMAGE_CLASSES_COUNT: int = 10

EXTRUDER_IMAGE_SIDE_PIXEL_COUNT: int = 600
EXTRUDER_REDUCED_SIZE_IMAGE_SIDE_PIXEL_COUNT: int = 200
EXTRUDER_IMAGE_CLASSES_COUNT: int = 1000

EXPERIMENT_LAUNCHER_RUNS_DIR_NAME: str = "experiments"
DEFAULT_LAUNCHER_RUNS_DIR_NAME: str = "models"

INFER_IMAGE_COUNT: int = 25

MODEL_DIR: str = "model_dir"

TFRECORD_LEFT_BYTES: str = "left_bytes"
TFRECORD_RIGHT_BYTES: str = "right_bytes"
TFRECORD_PAIR_LABEL: str = "pair_label"
TFRECORD_LEFT_LABEL: str = "left_label"
TFRECORD_RIGHT_LABEL: str = "right_label"

TFRECORD_IMAGE_BYTES: str = "image_bytes"
TFRECORD_LABEL: str = "label"

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

PNG: str = '.png'
JPG: str = '.jpg'
LOG: str = '.log'

NUM_CHANNELS: str = "num_channels"
HARD_TRIPLET_MARGIN: str = "hard_triplet_margin"
EMBEDDING_SIZE: str = "embedding_size"

EMPTY_STR = ''
