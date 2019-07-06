import numpy as np
import pytest
import tensorflow as tf

import src.estimator.model.estimator_conv_model
from src.data.common_types import DictsDataset
from src.data.raw_data.raw_data_providers import MnistRawDataProvider, FmnistRawDataProvider
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.estimator.training import training
from src.utils import utils, filenames, consts
from testing_utils import testing_consts, gen
from testing_utils.tf_helpers import run_eagerly


@pytest.mark.integration
@pytest.mark.parametrize('fake_dataset', [
    MnistRawDataProvider,
    FmnistRawDataProvider
], indirect=True)
def test_softmax_model(fake_dataset: DictsDataset):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    features = fake_dataset.features
    steps = np.random.randint(2, 5)
    run_data = gen.run_data(model=MnistSoftmaxModel())
    estimator = training.create_estimator(run_data)

    estimator.train(input_fn=lambda: fake_dataset.as_tuple(), steps=steps)
    names = [gen.random_str() for _ in range(2)]
    eval_results = estimator.evaluate(input_fn=lambda: fake_dataset.as_tuple(), steps=1, name=names[0])
    estimator.evaluate(input_fn=lambda: fake_dataset.as_tuple(), steps=1, name=names[1])

    _assert_log_dirs(filenames.get_run_logs_data_dir(run_data), names)

    loss = eval_results['loss']
    global_step = eval_results['global_step']
    accuracy = eval_results['accuracy']
    assert loss.shape == ()
    assert global_step == steps
    assert accuracy.shape == ()

    predictions_generator = estimator.predict(input_fn=lambda: features)

    for _ in range((list(features.values())[0]).shape[0]):
        predictions = next(predictions_generator)
        assert predictions['probabilities'].shape == (2,)
        assert predictions[consts.INFERENCE_CLASSES].shape == ()


def _create_estimator_spec(mode, images_dataset):
    features, labels = images_dataset
    return MnistSoftmaxModel().softmax_model_fn(features, labels, mode,
                                                params={consts.MODEL_DIR: str(filenames._get_home_infer_dir())})


@pytest.mark.parametrize('fake_dataset', [
    MnistRawDataProvider,
], indirect=True)
def test_model_fn_train_mode(fake_dataset: DictsDataset):
    spec = _create_estimator_spec(tf.estimator.ModeKeys.TRAIN, fake_dataset.as_tuple())

    assert spec.mode == tf.estimator.ModeKeys.TRAIN
    loss = spec.loss
    assert loss.shape == ()
    assert loss.dtype == tf.float32


@pytest.mark.parametrize('fake_dataset', [
    MnistRawDataProvider,
], indirect=True)
def test_model_fn_eval_mode(fake_dataset: DictsDataset):
    spec = _create_estimator_spec(tf.estimator.ModeKeys.EVAL, fake_dataset.as_tuple())

    assert spec.mode == tf.estimator.ModeKeys.EVAL

    eval_metric_ops = spec.eval_metric_ops
    assert eval_metric_ops['accuracy'][0].shape == ()
    assert eval_metric_ops['accuracy'][1].shape == ()
    assert eval_metric_ops['accuracy'][0].dtype == tf.float32
    assert eval_metric_ops['accuracy'][1].dtype == tf.float32


@pytest.mark.parametrize('fake_dataset', [
    MnistRawDataProvider,
], indirect=True)
def test_model_fn_predict_mode(fake_dataset: DictsDataset):
    spec = _create_estimator_spec(tf.estimator.ModeKeys.PREDICT, fake_dataset.as_tuple())

    assert spec.mode == tf.estimator.ModeKeys.PREDICT

    predictions = spec.predictions
    assert predictions['probabilities'].shape == (testing_consts.FAKE_IMAGES_IN_DATASET_COUNT, 2)
    assert predictions['probabilities'].dtype == tf.float32
    assert predictions[consts.INFERENCE_CLASSES].shape == testing_consts.FAKE_IMAGES_IN_DATASET_COUNT
    assert predictions[consts.INFERENCE_CLASSES].dtype == tf.int64


SECOND_EVAL_NAME = 'quux'


def _assert_log_dirs(logs_dir, names):
    eval_dirs = [logs_dir / ('eval_' + names[0]), logs_dir / ('eval_' + names[1])]
    for e in eval_dirs:
        assert utils.check_filepath(e)
    checkpoint_files = logs_dir.glob('model.ckpt*')
    assert list(checkpoint_files)


@run_eagerly
def test_non_streaming_accuracy():
    a = tf.constant([0, 0, 1, 1, 0, 1])
    b = tf.constant([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    a = tf.cast(a, tf.int32)
    b = tf.cast(b, tf.int32)
    accuracy = src.estimator.model.estimator_conv_model.non_streaming_accuracy(a, b)
    assert accuracy.numpy() == 0.5

    accuracy = src.estimator.model.estimator_conv_model.non_streaming_accuracy(b, a)
    assert accuracy.numpy() == 0.5

    c = tf.constant([2, 2, 2, 2, 2, 2])
    accuracy = src.estimator.model.estimator_conv_model.non_streaming_accuracy(b, c)
    assert accuracy.numpy() == 0.0
