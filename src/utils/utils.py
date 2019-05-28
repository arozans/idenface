from pathlib import Path
from typing import TYPE_CHECKING, List
from typing import Union

import PIL
import numpy as np
import tensorflow as tf

from src.estimator.launcher.launchers import Launcher
from src.utils import consts

if TYPE_CHECKING:
    from src.estimator.model.estimator_model import EstimatorModel
from src.utils.configuration import config


def log(text: str):
    tf.logging.info(' --- ' + str(text) + ' --- ')


def lognl(text: str):
    tf.logging.info('--- ' + str(text) + ' --- ' + '\n' * 6)


def split_columns(params, column_type=None):
    zipped = zip(*params)
    columns = [np.array(e) for e in zipped]
    if column_type:
        try:
            columns = [e.astype(column_type[i]) for i, e in enumerate(columns)]
        except TypeError:  # fail gracefully if same type for every column
            columns = [e.astype(column_type) for i, e in enumerate(columns)]
    return columns


def global_suffix_or_emtpy() -> str:
    global_suffix = config[consts.GLOBAL_SUFFIX]
    return ('_' + global_suffix) if global_suffix is not None else ""


def get_run_summary(model: 'EstimatorModel'):
    from src.utils import filenames
    excluded_fragment = filenames.create_excluded_name_fragment(with_prefix=True)
    return model.summary + global_suffix_or_emtpy() + excluded_fragment


def check_filepath(filename: Union[str, Path], exists=True, is_directory=True, is_empty=False,
                   expected_len=None) -> bool:
    p = Path(filename)
    if not exists:
        return not p.exists()

    if is_directory:
        if not p.is_dir():
            return False
        empty = not bool(list(p.iterdir()))
    else:
        if not p.is_file():
            return False
        empty = not bool(p.stat().st_size)

    correct_len = len(list(p.iterdir())) == expected_len if expected_len else True
    return (empty == is_empty) and correct_len


def user_run_selection(launcher: Launcher):
    print("Launcher {} contains below models: ".format(launcher.name))
    runs = {}
    for idx, run_data in enumerate(launcher.runs_data):
        print("{}: {}".format(idx, run_data.model.summary))
        runs.update({idx: run_data})
    while True:
        user_input = input("Select model to perform inference: ")
        try:
            return runs[eval(user_input)]
        except SyntaxError:
            continue


def pretty_print_list(l: List):
    return '_'.join([str(x) for x in l])


def load_image(image_path: Union[Path, str]) -> PIL.Image:
    return PIL.Image.open(str(image_path))


def get_first_batch(dataset):
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()
    with tf.Session() as sess:
        res = sess.run(first_batch)
    return res
