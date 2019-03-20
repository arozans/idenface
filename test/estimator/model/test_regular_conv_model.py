import pytest
import tensorflow as tf

from helpers import test_consts, gen
from helpers.tf_helpers import run_eagerly
from src.data.raw_data.raw_data_providers import MnistRawDataProvider
from src.estimator.model import regular_conv_model
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.training import training
from src.utils import utils, filenames, consts


def _create_estimator_spec(mode, images_dataset):
    features, labels = images_dataset
    return regular_conv_model.cnn_model_fn(features, labels, mode,
                                           params={consts.MODEL_DIR: str(filenames.get_infer_dir())})


@pytest.mark.integration
@pytest.mark.parametrize('fake_dict_and_labels', [
    MnistRawDataProvider,
    # FakeModel #fixme: regular conv model has harcoded mnist sizes values! - all test in this file
], indirect=True)
def test_regular_conv_cnn_estimator(fake_dict_and_labels):
    fake_dict = fake_dict_and_labels[0]
    steps = 4
    run_data = gen.run_data(model=MnistCNNModel())
    estimator = training.create_estimator(run_data)

    estimator.train(input_fn=lambda: fake_dict_and_labels, steps=steps)
    eval_results = estimator.evaluate(input_fn=lambda: fake_dict_and_labels, steps=1)
    estimator.evaluate(input_fn=lambda: fake_dict_and_labels, steps=1, name=SECOND_EVAL_NAME)

    _assert_log_dirs(filenames.get_run_logs_data_dir(run_data))

    loss = eval_results['loss']
    global_step = eval_results['global_step']
    accuracy = eval_results['accuracy']
    assert loss.shape == ()
    assert global_step == steps
    assert accuracy.shape == ()

    predictions_generator = estimator.predict(input_fn=lambda: fake_dict)

    for _ in range((list(fake_dict.values())[0]).shape[0]):
        predictions = next(predictions_generator)
        assert predictions['probabilities'].shape == (2,)
        assert predictions['classes'].shape == ()


@pytest.mark.parametrize('fake_dict_and_labels', [
    MnistRawDataProvider,
], indirect=True)
def test_model_fn_train_mode(fake_dict_and_labels):
    spec = _create_estimator_spec(tf.estimator.ModeKeys.TRAIN, fake_dict_and_labels)

    assert spec.mode == tf.estimator.ModeKeys.TRAIN
    loss = spec.loss
    assert loss.shape == ()
    assert loss.dtype == tf.float32


@pytest.mark.parametrize('fake_dict_and_labels', [
    MnistRawDataProvider,
], indirect=True)
def test_model_fn_eval_mode(fake_dict_and_labels):
    spec = _create_estimator_spec(tf.estimator.ModeKeys.EVAL, fake_dict_and_labels)

    assert spec.mode == tf.estimator.ModeKeys.EVAL

    eval_metric_ops = spec.eval_metric_ops
    assert eval_metric_ops['accuracy'][0].shape == ()
    assert eval_metric_ops['accuracy'][1].shape == ()
    assert eval_metric_ops['accuracy'][0].dtype == tf.float32
    assert eval_metric_ops['accuracy'][1].dtype == tf.float32


@pytest.mark.parametrize('fake_dict_and_labels', [
    MnistRawDataProvider,
], indirect=True)
def test_model_fn_predict_mode(fake_dict_and_labels):
    spec = _create_estimator_spec(tf.estimator.ModeKeys.PREDICT, fake_dict_and_labels)

    assert spec.mode == tf.estimator.ModeKeys.PREDICT

    predictions = spec.predictions
    assert predictions['probabilities'].shape == (test_consts.FAKE_IMAGES_IN_DATASET_COUNT, 2)
    assert predictions['probabilities'].dtype == tf.float32
    assert predictions['classes'].shape == test_consts.FAKE_IMAGES_IN_DATASET_COUNT
    assert predictions['classes'].dtype == tf.int64


SECOND_EVAL_NAME = 'quux'


def _assert_log_dirs(logs_dir):
    eval_dirs = [logs_dir / 'eval', logs_dir / ('eval_' + SECOND_EVAL_NAME)]
    for e in eval_dirs:
        assert utils.check_filepath(e)
    checkpoint_files = logs_dir.glob('model.ckpt*')
    assert list(checkpoint_files)


@run_eagerly
def test_non_streaming_accuracy():
    a = tf.constant([0, 0, 1, 1, 0, 1])
    b = tf.constant([1, 0, 0, 1, 1, 1])

    accuracy = regular_conv_model.non_streaming_accuracy(a, b)
    assert accuracy.numpy() == 0.5

    accuracy = regular_conv_model.non_streaming_accuracy(b, a)
    assert accuracy.numpy() == 0.5

    c = tf.constant([2, 2, 2, 2, 2, 2])
    accuracy = regular_conv_model.non_streaming_accuracy(b, c)
    assert accuracy.numpy() == 0.0
