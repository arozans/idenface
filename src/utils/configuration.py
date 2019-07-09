import collections
import copy
import time
from typing import Any, Dict, Iterable

from absl.flags import UnrecognizedFlagError
from tensorflow import flags as tf_flags

from src.estimator.launcher.launchers import RunData
from src.utils import params, consts

DESC = 'no-help :('


def define_cli_args():
    tf_flags.DEFINE_integer(consts.BATCH_SIZE, None, DESC)
    tf_flags.DEFINE_string(consts.OPTIMIZER, None, DESC)
    tf_flags.DEFINE_float(consts.LEARNING_RATE, None, DESC)
    tf_flags.DEFINE_integer(consts.TRAIN_STEPS, None, DESC)
    tf_flags.DEFINE_integer(consts.EVAL_STEPS_INTERVAL, None, DESC)
    tf_flags.DEFINE_list(consts.EXCLUDED_KEYS, None, DESC)


def remove_unnecessary_flags(commandline_flags):
    redundant_flags = ['h', 'help', 'helpshort', 'helpfull']
    return {k: v for k, v in commandline_flags.items() if k not in redundant_flags}


class ConfigDict(collections.MutableMapping):

    def __init__(self):
        self.tf_flags: Dict[str, Any] = {}
        self.model_params: Dict[str, Any] = {}
        self.launcher_params: Dict[str, Any] = {}
        self.file_defined_params: Dict[str, Any] = copy.deepcopy(params.PARAMS)
        self.full_config = copy.deepcopy(params.PARAMS)

    def __getitem__(self, key):
        try:
            return self.full_config[key]
        except KeyError:
            return None

    def _rebuild_full_config(self):
        self.full_config.update({k: v for k, v in self.file_defined_params.items()})
        self.full_config.update({k: v for k, v in self.model_params.items()})
        self.full_config.update({k: v for k, v in self.launcher_params.items()})
        self.full_config.update({k: v for k, v in self.tf_flags.items()})

    def update_tf_flags(self):
        import sys
        try:
            tf_flags.FLAGS(sys.argv)
        except UnrecognizedFlagError:
            pass
        commandline_flags = tf_flags.FLAGS.flag_values_dict()
        commandline_flags = remove_unnecessary_flags(commandline_flags)
        self.tf_flags = {k: v for k, v in commandline_flags.items() if v is not None}  # TODO: allow for 'None' flags
        self._rebuild_full_config()

    def update_model_params(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self._rebuild_full_config()

    def update_launcher_params(self, launcher_params: Dict[str, Any]):
        self.launcher_params = launcher_params
        self._rebuild_full_config()

    def __setitem__(self, key, value):
        raise TypeError("Configuration can only be set on app startup")

    def __delitem__(self, key):
        raise TypeError("Configuration cannot be modified after app startup")

    def __iter__(self):
        return iter(self.full_config)

    def __len__(self):
        return len(self.full_config)

    def pretty_full_dict_summary(self, run_data: RunData):
        def filter_keys(full_dict: dict, keys: Iterable, complement: bool = False) -> Dict[str, any]:
            if not complement:
                return dict((x, full_dict[x]) for x in keys if x in full_dict.keys())
            else:
                return {k: v for (k, v) in full_dict.items() if complement and k not in keys}

        def create_line(lenght: int = 50, text: str = ""):
            return text + "-" * lenght + '\n'

        def log_dict(summary_dict: dict, text: str = ""):
            for k, v in summary_dict.items():
                text = text + str(k) + ": " + str(v) + '  \n'
            return text

        run_keys = [
            consts.MODEL_SUMMARY,
            consts.DATASET_VARIANT,
            consts.DATASET_IMAGE_DIMS,
            consts.EXCLUDED_KEYS,
            consts.GLOBAL_SUFFIX
        ]
        model_keys = [
            consts.BATCH_SIZE,
            consts.LEARNING_RATE,
            consts.FILTERS,
            consts.KERNEL_SIDE_LENGTHS,
            consts.DENSE_UNITS,
            consts.CONCAT_DENSE_UNITS,
            consts.CONCAT_DROPOUT_RATES,
        ]
        params_count_keys = [
            consts.ALL_PARAMS_COUNT,
            consts.ALL_PARAMS_SIZE_MB,
            consts.CONV_PARAMS_COUNT,
            consts.DENSE_PARAMS_COUNT,
            consts.CONCAT_DENSE_PARAMS_COUNT,
        ]

        model_parameters_count = run_data.model.get_parameters_count_dict()
        single_image_dims = str(run_data.model.raw_data_provider.description.image_dimensions.as_tuple())
        summary = "Full configuration:  \n"
        currdate = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        summary = log_dict({consts.CURRENT_DATE: currdate}, text=summary)
        summary = create_line(text=summary)
        summary = log_dict(
            filter_keys({**self.full_config, **{consts.DATASET_IMAGE_DIMS: single_image_dims}}, run_keys), text=summary)
        summary = create_line(text=summary)
        summary = log_dict(filter_keys(self.full_config, model_keys), text=summary)
        summary = create_line(text=summary)
        summary = log_dict(filter_keys(model_parameters_count, params_count_keys), text=summary)
        summary = create_line(text=summary)
        summary = log_dict(filter_keys(self.full_config, run_keys + model_keys, True), text=summary)

        return summary


config = ConfigDict()
