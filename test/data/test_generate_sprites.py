from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
import skimage

from src.data import generate_sprites
from src.data.common_types import ImageDimensions
from testing_utils import gen
from testing_utils.testing_classes import TestDatasetVariant, FakeRawDataProvider


@pytest.mark.parametrize('is_rgb', [False, True])
@pytest.mark.parametrize('with_border', [False, True])
@pytest.mark.parametrize('sprite_expected_side_length', [150, 200])
def test_should_create_correctly_sized_sprite(sprite_expected_side_length, is_rgb, with_border):
    image_dims = ImageDimensions(20, 20, 3 if is_rgb else 1)
    features = gen.dicts_dataset(batch_size=150, image_dims=image_dims, paired=True, normalize=True).features
    expected_dims = ImageDimensions(sprite_expected_side_length)
    sprite = generate_sprites.create_sprite_image(features=features, expected_dims=expected_dims,
                                                  with_border=with_border)
    assert sprite.height == sprite_expected_side_length
    assert np.array(sprite).max() > 0  # make sure image is not black due to PIL poor float to uint conversion


def test_should_save_sprite_on_disc(mocker):
    mocker.patch('builtins.input', return_value='0')
    mocker.patch.dict(generate_sprites.raw_data_providers, values={TestDatasetVariant.FOO: FakeRawDataProvider},
                      clear=True)
    mocker.patch.object(generate_sprites, 'SPRITE_ELEMENTS_SIDE_LENGTH', 7)

    path = generate_sprites.generate_sprites_and_save()
    assert isinstance(path, Path)
    assert path.exists()
    show = False
    if show:
        sprite_image = mpimg.imread(str(path))
        sprite_image = skimage.img_as_float(sprite_image)
        image_arr = sprite_image[None, :]
        plt.imshow(image_arr.squeeze())
        plt.show()
