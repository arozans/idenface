from typing import List

import numpy as np

from src.data.common_types import ImageDimensions


def calculate_conv_params(input_image_dims: ImageDimensions,
                          filters: List[int],
                          kernel_side_lengths: List[int]):
    param_num = 0
    for i, (f, k) in enumerate(zip(filters, kernel_side_lengths)):
        if i == 0:
            param_num = param_num + ((k * k * input_image_dims.channels + 1) * f)
        else:
            param_num = param_num + ((k * k * filters[i - 1] + 1) * f)
    return param_num


def calculate_dense_params(conv_output_size: int,
                           filters: List[int],
                           dense_units: List[int]):
    return _calculate_dense_params(conv_output_size, filters, dense_units, 1)


def calculate_concat_dense_params(conv_output_size: int,
                                  filters: List[int],
                                  dense_units: List[int],
                                  subnets_num: int):
    return _calculate_dense_params(conv_output_size, filters, dense_units, subnets_num)


def _calculate_dense_params(conv_output_size: int, filters: List[int], dense_units: List[int], subnets_num: int):
    params_num = 0
    for i, units in enumerate(dense_units):
        if i == 0:
            params_num = params_num + ((conv_output_size * conv_output_size * filters[-1] * subnets_num + 1) * units)
        else:
            params_num = params_num + ((dense_units[i - 1] + 1) * units)
    return params_num


def calculate_convmax_output(input_size: int, conv_filter_num: int, maxpool_stride=2):
    maxpool_output = input_size
    for _ in range(conv_filter_num):
        maxpool_output = np.ceil(maxpool_output / maxpool_stride)

    return int(maxpool_output)
