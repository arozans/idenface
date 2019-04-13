import collections
import copy
from typing import Any, Dict

from absl.flags import UnrecognizedFlagError
from tensorflow import flags as tf_flags

from src.utils import params, consts

DESC = 'no-help :('


def define_cli_args():
    tf_flags.DEFINE_integer(consts.BATCH_SIZE, None, DESC)
    tf_flags.DEFINE_string(consts.OPTIMIZER, None, DESC)
    tf_flags.DEFINE_float(consts.LEARNING_RATE, None, DESC)
    tf_flags.DEFINE_integer(consts.TRAIN_STEPS, None, DESC)
    tf_flags.DEFINE_integer(consts.EVAL_STEPS_INTERVAL, None, DESC)
    tf_flags.DEFINE_list(consts.EXCLUDED_KEYS, None, DESC)


class ConfigDict(collections.MutableMapping):

    def __init__(self):
        self.tf_flags: Dict[str, Any] = {}
        self.model_params: Dict[str, Any] = {}
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
        self.full_config.update({k: v for k, v in self.tf_flags.items()})

    def update_tf_flags(self):
        import sys
        try:
            tf_flags.FLAGS(sys.argv)
        except UnrecognizedFlagError:
            pass
        commandline_flags = tf_flags.FLAGS.flag_values_dict()
        self.tf_flags = {k: v for k, v in commandline_flags.items() if v is not None}  # TODO: allow for 'None' flags
        self._rebuild_full_config()

    def update_model_params(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self._rebuild_full_config()

    def __setitem__(self, key, value):
        raise TypeError("Configuration can only be set on app startup")

    def __delitem__(self, key):
        raise TypeError("Configuration cannot be modified after app startup")

    def __iter__(self):
        return iter(self.full_config)

    def __len__(self):
        return len(self.full_config)


config = ConfigDict()
