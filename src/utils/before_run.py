import logging
import shutil
from pathlib import Path
from typing import List, Union

import tensorflow as tf

from src.estimator.launcher.launchers import RunData
from src.utils import filenames, consts
from src.utils import utils, image_summaries
from src.utils.configuration import config


def _enable_training_logging(run_data: RunData):
    filename = filenames.create_text_log_name(run_data.model)

    model_dir_log_file = filenames.get_run_text_logs_dir(run_data) / filename
    all_logs_log_file = filenames.get_all_text_logs_dir() / filename

    _set_logging_handlers([all_logs_log_file, model_dir_log_file])


def _set_logging_handlers(handlers_to_set: List[Path]):
    [log_file.parent.mkdir(exist_ok=True, parents=True) for log_file in handlers_to_set]
    tf.logging.set_verbosity(tf.logging.DEBUG)
    log = logging.getLogger('tensorflow')
    for handler in log.handlers[1:]:
        log.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers_to_set:
        _set_logging_handler(handler, formatter, log)


def _set_logging_handler(text_log_filename, formatter, logger):
    fh = logging.FileHandler(str(text_log_filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def _log_configuration(args: List[str]):
    _log_config()

    commandline_args = [x for x in args if not x.startswith('--')]
    undefined_flags = [x for x in args if x.startswith('--')]
    utils.log('Remainder commandline arguments: {}'.format(commandline_args))
    utils.log('Undefined commandline flags: {}'.format(undefined_flags))


def _log_config():
    utils.log('Code-defined params: {}'.format(config.file_defined_params))
    utils.log('Model params: {}'.format(config.model_params))
    utils.log('Launcher params: {}'.format(config.launcher_params))
    utils.log('Commandline flags: {}'.format(config.tf_flags))
    utils.log(config.pretty_full_dict_summary())


def _prepare_dirs(deleted_old_exp_path: Union[None, Path], run_data: RunData):
    if run_data.is_experiment:
        if deleted_old_exp_path:
            utils.log('Found not empty experiment dir from previous runs: {}'.format(deleted_old_exp_path))
            utils.log('Deleting old experiment dir: {}'.format(deleted_old_exp_path))
        else:
            launcher_dir = filenames.get_launcher_dir(run_data)
            utils.log('Experiment dir from previous runs not found.'.format(launcher_dir))
            utils.log('Creating experiment dir: {}'.format(launcher_dir))

    _prepare_runs_dir(run_data)
    _prepare_log_dir(run_data)
    launcher_dir = filenames.get_run_dir(run_data).parent
    utils.log("Inspect this run results with command: \ntensorboard --logdir={}\n".format(launcher_dir))


def _prepare_runs_dir(run_data: RunData):
    run_dir = filenames.get_run_dir(run_data)
    if not run_dir.exists():
        utils.log('Creating directory for run: {}'.format(run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)


def _prepare_launcher_dir(run_data: RunData) -> Union[Path, None]:
    if not run_data.is_experiment_and_first_run():
        return None

    launcher_dir = filenames.get_launcher_dir(run_data)
    if utils.check_filepath(filename=launcher_dir, exists=True, is_directory=True, is_empty=False):
        shutil.rmtree(str(launcher_dir))
        return launcher_dir
    else:
        launcher_dir.mkdir(exist_ok=False, parents=True)
        return None


def _prepare_log_dir(run_data: RunData):
    log_dir = filenames.get_run_logs_data_dir(run_data)
    if utils.check_filepath(filename=log_dir, exists=True, is_directory=True, is_empty=False):
        utils.log('Found not empty logs directory from previous runs: {}'.format(log_dir))
        if config[consts.REMOVE_OLD_MODEL_DIR]:
            utils.log('Deleting old model_dir: {}'.format(log_dir))
            shutil.rmtree(str(log_dir))
    else:
        utils.log('Logs directory from previous runs not found. Creating new: {}'.format(log_dir))
        log_dir.mkdir(exist_ok=False, parents=True)


def _log_training_model(run_data: RunData):
    utils.log("Initiate launcher: {}, model: {} ({} of {})".format(run_data.launcher_name, run_data.model.summary,
                                                                   run_data.run_no, run_data.models_count))


def create_text_summary(run_data: RunData):
    tf.reset_default_graph()
    with tf.Session() as sess:
        txt_summary = tf.summary.text('configuration', tf.constant(config.pretty_full_dict_summary()))

        dir = filenames.get_run_logs_data_dir(run_data)
        dir.mkdir(exist_ok=True, parents=True)
        writer = tf.summary.FileWriter(str(dir), sess.graph)

        sess.run(tf.global_variables_initializer())

        summary = sess.run(txt_summary)
        writer.add_summary(summary)
        writer.flush()


def prepare_env(args: List[str], run_data: RunData):
    config.update_tf_flags()
    config.update_model_params(run_data.model.params)
    config.update_launcher_params(run_data.launcher_params)
    deleted_old_exp_path = _prepare_launcher_dir(run_data)
    _enable_training_logging(run_data)
    _log_training_model(run_data)
    _log_configuration(args)
    _prepare_dirs(deleted_old_exp_path, run_data)
    image_summaries.create_pair_summaries(run_data)
    create_text_summary(run_data)


def prepare_infer_env(run_data: RunData):
    config.update_model_params(run_data.model.params)

    inference_dir = filenames.get_infer_dir(run_data)
    filename = filenames.create_infer_log_name(run_data.model)
    _set_logging_handlers([(inference_dir / filename)])

    utils.log("Inference data will be saved into: {}".format(inference_dir))
    _check_model_checkpoint_existence(run_data)
    _log_inference_model(run_data)


def _check_model_checkpoint_existence(run_data: RunData):
    strict: bool = config[consts.IS_INFER_CHECKPOINT_OBLIGATORY]
    if not strict:
        utils.log("Not checking checkpoint existence")
        return
    model_dir = filenames.get_run_logs_data_dir(run_data)
    assert model_dir.exists(), "{} does not exists - no model to load!".format(model_dir)

    checkpoints = model_dir.glob('*.ckpt-*')
    checkpoints_with_number = {x for y in checkpoints for x in str(y).split('.') if x.startswith("ckpt")}
    step_numbers = {int(x.split('-')[-1]) for x in checkpoints_with_number}

    assert bool(step_numbers), "No checkpoints exists!"
    assert len(step_numbers) > 1 or 0 not in step_numbers, "Only one checkpoint  - for 0th step exists!"
    utils.log("Checkpoint directory: ok, max checkoint number: {}".format(max(step_numbers)))


def _log_inference_model(run_data: RunData):
    utils.log(
        "Initiate model for inference, name: {}, summary: {}".format(run_data.launcher_name, run_data.model.summary))
    _log_config()
