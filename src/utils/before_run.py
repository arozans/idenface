import logging
import shutil
from pathlib import Path
from typing import List, Union

import tensorflow as tf

from src.estimator.launcher.launchers import RunData
from src.estimator.model.estimator_model import EstimatorModel
from src.utils import configuration, filenames
from src.utils import utils, image_summaries
from src.utils.configuration import config


def _enable_logging(run_data: RunData):
    filename = filenames.create_text_log_name(run_data.model)

    model_dir_location = filenames.get_run_text_logs_dir(run_data) / filename
    model_dir_location.parent.mkdir(exist_ok=True, parents=True)
    all_logs_location = filenames.get_all_text_logs_dir() / filename
    all_logs_location.parent.mkdir(exist_ok=True, parents=True)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    log = logging.getLogger('tensorflow')
    for handler in log.handlers[1:]:
        log.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    _set_logging_handler(all_logs_location, formatter, log)
    _set_logging_handler(model_dir_location, formatter, log)


def _set_logging_handler(text_log_filename, formatter, logger):
    fh = logging.FileHandler(str(text_log_filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def _log_flags(args: List[str]):
    utils.log('Code-defined params: {}'.format(configuration.get_file_params()))
    utils.log('Model params: {}'.format(configuration.get_model_params()))
    utils.log('Commandline flags: {}'.format(configuration.get_commandline_flags()))

    commandline_args = [x for x in args if not x.startswith('--')]
    undefined_flags = [x for x in args if x.startswith('--')]
    utils.log('Remainder commandline arguments: {}'.format(commandline_args))
    utils.log('Undefined commandline flags: {}'.format(undefined_flags))


def _prepare_dirs(deleted_old_exp_path: Union[None, Path], run_data: RunData):
    if deleted_old_exp_path:
        utils.log('Found not empty experiment dir from previous runs: {}'.format(deleted_old_exp_path))
        utils.log('Deleting old experiment dir: {}'.format(deleted_old_exp_path))
    _prepare_runs_dir(run_data)
    _prepare_log_dir(run_data)
    launcher_dir = filenames.get_run_dir(run_data).parent
    utils.log("Inspect this run results with command: \ntensorboard --logdir={}\n".format(launcher_dir))


def _prepare_runs_dir(run_data: RunData):
    run_dir = filenames.get_run_dir(run_data)
    if not run_dir.exists():
        utils.log('Creating directory for run: {}'.format(run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)


def _prepare_launcher_dir(run_data) -> Union[Path, None]:
    if not run_data.is_experiment:
        return None
    if run_data.run_no != 1:
        return None
    launcher_dir = filenames.get_launcher_dir(run_data)
    if utils.check_filepath(filename=launcher_dir, exists=True, is_directory=True, is_empty=False):
        shutil.rmtree(str(launcher_dir))
        return launcher_dir
    else:
        utils.log('Experiment dir from previous runs not found.'.format(launcher_dir))
        utils.log('Creating experiment dir: {}'.format(launcher_dir))
        launcher_dir.mkdir(exist_ok=False, parents=True)
        return None


def _prepare_log_dir(run_data: RunData):
    log_dir = filenames.get_run_logs_data_dir(run_data)
    if utils.check_filepath(filename=log_dir, exists=True, is_directory=True, is_empty=False):
        utils.log('Found not empty logs directory from previous runs: {}'.format(log_dir))
        if config.remove_old_model_dir:
            utils.log('Deleting old model_dir: {}'.format(log_dir))
            shutil.rmtree(str(log_dir))
    else:
        utils.log('Logs directory from previous runs not found. Creating new: {}'.format(log_dir))
        log_dir.mkdir(exist_ok=False, parents=True)


def _register_model_variables(model: EstimatorModel):
    config.set_model_params(model.params)


def _log_training_model(run_data: RunData):
    utils.log("Initiate launcher: {}, model: {} ({} of {})".format(run_data.launcher_name, run_data.model.summary,
                                                                   run_data.run_no, run_data.models_count))


def prepare_env(args: List[str], run_data: RunData):
    _register_model_variables(run_data.model)
    deleted_old_exp_path = _prepare_launcher_dir(run_data)
    _enable_logging(run_data)
    _log_training_model(run_data)
    _log_flags(args)
    _prepare_dirs(deleted_old_exp_path, run_data)
    image_summaries.create_pair_summaries(run_data)
