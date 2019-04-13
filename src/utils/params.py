from src.utils import consts

PARAMS = {
    consts.SHUFFLE_BUFFER_SIZE: 10000,
    consts.REMOVE_OLD_MODEL_DIR: True,
    consts.BATCH_SIZE: 512,
    consts.OPTIMIZER: 'GradientDescent',
    consts.LEARNING_RATE: 0.01,
    consts.TRAIN_STEPS: 7 * 1000,
    consts.EVAL_STEPS_INTERVAL: 500,
    consts.PAIRING_WITH_IDENTICAL: False,
    consts.EXCLUDED_KEYS: [],
    consts.GLOBAL_SUFFIX: None,
    consts.ENCODING_TFRECORDS: True,
    consts.IS_INFER_CHECKPOINT_OBLIGATORY: True,
    consts.PREDICT_SIMILARITY_MARGIN: 0.25,
    consts.TRAIN_SIMILARITY_MARGIN: 0.5,
}
