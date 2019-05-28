import PIL
import numpy as np
import skimage

from src.data.common_types import DatasetVariant, DictsDataset, FeaturesDict, ImageDimensions
from src.data.raw_data.raw_data_providers import MnistRawDataProvider, FmnistRawDataProvider, ExtruderRawDataProvider
from src.estimator.training.supplying_datasets import TFRecordDatasetProvider
from src.utils import utils, filenames

raw_data_providers = {
    DatasetVariant.MNIST: MnistRawDataProvider,
    DatasetVariant.FMNIST: FmnistRawDataProvider,
    DatasetVariant.EXTRUDER: ExtruderRawDataProvider
}
SPRITE_ELEMENTS_SIDE_LENGTH = 10
SPRITE_SIDE_DIMENSION = 1000


def generate_sprites_and_save():
    variant = _user_raw_data_provider_selection()
    raw_data_provider = raw_data_providers[variant]()
    provider = TFRecordDatasetProvider(raw_data_provider)

    dataset = provider.infer(SPRITE_ELEMENTS_SIDE_LENGTH ** 2)
    first_batch = utils.get_first_batch(dataset)
    dicts_dataset = DictsDataset(*first_batch)
    sprite: PIL.Image = create_sprite_image(
        dicts_dataset.features,
        ImageDimensions(SPRITE_SIDE_DIMENSION),
        SPRITE_ELEMENTS_SIDE_LENGTH,
        with_border=True
    )
    path = filenames.get_sprites_filename(raw_data_provider.description.variant)
    sprite.save(path)
    return path


def _user_raw_data_provider_selection():
    print("Choose from below raw providers: ")
    for idx, variant in enumerate(raw_data_providers):
        print("{}: {}".format(idx, variant))
    while True:
        user_input = input("Insert number: ")
        try:
            return list(raw_data_providers.keys())[eval(user_input)]
        except SyntaxError:
            continue


def create_sprite_image(features: FeaturesDict, expected_dims: ImageDimensions, rows=10, with_border: bool = False):
    assert features.is_paired, "Only paired dicts sprites are supported"
    feature_dims = ImageDimensions(features.left[0].shape)
    features_iterator = iter(np.concatenate((features.left, features.right)))
    cols = rows - 1
    sprite = None
    for row in range(rows):
        row_result = _get_next_normalized_image(features_iterator, with_border)

        for _ in range(cols):
            row_result = np.concatenate((row_result, _get_next_normalized_image(features_iterator, with_border)),
                                        axis=1)
        if row == 0:
            sprite = row_result
        else:
            sprite = np.concatenate((sprite, row_result))
        if with_border:
            border_line = _generate_border((_calc_border_pixel_width(feature_dims), sprite.shape[1], sprite.shape[-1]))
            sprite = np.concatenate((sprite, border_line), axis=0)
    if with_border:
        upper_line = _generate_border((_calc_border_pixel_width(feature_dims), sprite.shape[1], sprite.shape[-1]))
        sprite = np.concatenate((upper_line, sprite), axis=0)
        left_line = _generate_border(shape=[sprite.shape[0], _calc_border_pixel_width(feature_dims), sprite.shape[-1]])
        sprite = np.concatenate((left_line, sprite), axis=1)
    sprite = skimage.img_as_uint(sprite)
    image = PIL.Image.fromarray(sprite.squeeze().astype('uint8'))
    image = image.resize((expected_dims.height, expected_dims.width), PIL.Image.ANTIALIAS)
    return image


def _generate_border(shape):
    border = np.zeros(shape)
    border[:, :, 0] = 0.7  # make dark red
    return border


def _calc_border_pixel_width(param: ImageDimensions):
    res = param.width // 20
    return res if res > 0 else 2


def _get_next_normalized_image(features, with_border: bool):
    image = features.__next__() + 0.5
    if not with_border:
        return image
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    border_line = _generate_border(
        shape=[image.shape[0], _calc_border_pixel_width(ImageDimensions(image.shape)), image.shape[-1]])
    return np.concatenate((image, border_line), axis=1)


if __name__ == '__main__':
    generate_sprites_and_save()
