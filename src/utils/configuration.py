from typing import Mapping, Any, Dict

from absl.flags import UnrecognizedFlagError
from tensorflow import flags as tf_flags


def define_cli_args():
    tf_flags.DEFINE_integer('batch_size', None, 'no-help :(')
    tf_flags.DEFINE_string('optimizer', None, 'no-help :(')
    tf_flags.DEFINE_float('learning_rate', None, 'no-help :(')
    tf_flags.DEFINE_integer('train_steps', None, 'no-help :(')
    tf_flags.DEFINE_integer('eval_steps_interval', None, 'no-help :(')
    tf_flags.DEFINE_list('excluded_keys', None, 'no-help :(')


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    model_params: Dict[str, Any] = {}

    def __getattr__(self, flag_name):
        try:
            return self.try_search_commandline_flags(flag_name)
        except (AttributeError, AssertionError, UnrecognizedFlagError):
            try:
                return self.try_search_model_params(flag_name)
            except KeyError:
                return self.try_search_file_defined_param(flag_name)

    @staticmethod
    def try_search_commandline_flags(flag_name):
        flag = getattr(tf_flags.FLAGS, flag_name)
        assert flag is not None
        return flag

    def try_search_model_params(self, flag_name):
        return self.model_params[flag_name]

    @staticmethod
    def try_search_file_defined_param(flag_name):
        try:
            return globals()['_' + flag_name]
        except (AttributeError, KeyError):
            return None

    def set_model_params(self, params: Dict[str, Any]):
        self.model_params = params


# fixme - move code defined flags to new, possibly 'private' module - stop gimmicks with _
def get_file_params() -> Mapping[str, Any]:
    def _is_file_param(flag_name):
        return not flag_name.startswith('__') and flag_name.startswith('_') and not flag_name.endswith('_')

    def _trim_underscore(flag_name):
        return flag_name[1:]

    return {_trim_underscore(k): v for k, v in globals().items() if _is_file_param(k)}


def get_commandline_flags() -> Mapping[str, Any]:
    return tf_flags.FLAGS.flag_values_dict()


def get_model_params():
    return config.model_params


config = Config()
_shuffle_buffer_size = 10000
_remove_old_model_dir = True
_batch_size = 512
_optimizer = 'GradientDescent'
_learning_rate = 0.01
_train_steps = 5 * 10000
_eval_steps_interval = 2000
_pairing_with_identical = False
_excluded_keys = []
_global_suffix = None
_encoding_tfrecords = True
