from src.utils import consts

PARAMS = {
    consts.SHUFFLE_BUFFER_SIZE: 10000,
    consts.REMOVE_OLD_MODEL_DIR: True,
    consts.BATCH_SIZE: 512,
    consts.OPTIMIZER: consts.GRADIENT_DESCEND_OPTIMIZER,
    consts.LEARNING_RATE: 0.01,
    consts.TRAIN_STEPS: 7 * 1000,
    consts.EVAL_STEPS_INTERVAL: 500,
    consts.TRAIN_LOG_STEPS_INTERVAL: 100,
    consts.EXCLUDED_KEYS: [],
    consts.GLOBAL_SUFFIX: None,
    consts.IS_INFER_CHECKPOINT_OBLIGATORY: True,
    consts.IS_INFER_PLOT_BOARD_WITH_TITLE: False,
    consts.IS_INFER_PLOT_DARK_BG_ENABLED: False
}
