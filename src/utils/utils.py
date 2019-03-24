from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf

from src.utils.configuration import config


def log(text: str):
    tf.logging.info(' --- ' + text + ' --- ')


def lognl(text: str):
    tf.logging.info('--- ' + text + ' --- \n\n')


def split_columns(params, column_type=None):
    zipped = zip(*params)
    columns = [np.array(e) for e in zipped]
    if column_type:
        try:  # different type for each columns
            columns = [e.astype(column_type[i]) for i, e in enumerate(columns)]
        except TypeError:  # fail gracefully if same type for every column
            columns = [e.astype(column_type) for i, e in enumerate(columns)]
    return columns


def get_run_summary(model_summary: str):
    from src.utils import filenames
    global_suffix = config.global_suffix
    excluded_fragment = filenames.create_excluded_name_fragment(with_prefix=True)
    return model_summary + ('_' + global_suffix if global_suffix is not None else "") + excluded_fragment


def check_filepath(filename: Union[str, Path], exists=True, is_directory=True, is_empty=False) -> bool:
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

    return empty == is_empty
