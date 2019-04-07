import numpy as np

from helpers.tf_helpers import run_eagerly
from src.estimator.model import siamese_conv_model


@run_eagerly
def test_should_correctly_compare_distances_to_margin():
    distances = np.array([0, 0.1, 0.3, 0.4,
                          0.400001, 0.5, 1.5, 89])
    margin = 0.4
    result = siamese_conv_model.is_pair_similar(distances, margin)
    assert (result.numpy() == np.array([1.0, 1.0, 1.0, 1.0,
                                        0.0, 0.0, 0.0, 0.0])).all()
