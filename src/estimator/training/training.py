import tensorflow as tf

from src.estimator.launcher import providing_launcher
from src.estimator.launcher.launchers import RunData
from src.estimator.model.estimator_conv_model import EstimatorConvModel
from src.estimator.training import supplying_datasets
from src.utils import utils, before_run, filenames, consts, configuration
from src.utils.configuration import config

dataset_size = 60 * 1000
epochs_between_eval = 10


def main(args=None):
    launcher = providing_launcher.provide_launcher()
    for run_data in launcher.runs_data:
        try:
            before_run.prepare_env(args, run_data)
            train(run_data)
            after_run(run_data)
        except Exception as e:
            utils.error("During execution of {}, crititcal error wa raised: {}".format(run_data, e))


def train(run_data: RunData):
    estimator = create_estimator(run_data)
    utils.log('Starting train - eval loop excluding data classes: {}'.format(config[consts.EXCLUDED_KEYS]))
    in_memory_train_eval(estimator, run_data.model)
    utils.lognl('Finished training with model: {}'.format(run_data.model.summary))


def after_run(run_data: RunData):
    launcher_dir = filenames.get_run_dir(run_data).parent
    utils.log(consts.TENSORBOARD_COMMAND.format(launcher_dir))


def in_memory_train_eval(estimator: tf.estimator.Estimator, model: EstimatorConvModel):
    dataset_provider = model.dataset_provider
    train_steps = config[consts.TRAIN_STEPS]
    eval_steps_interval = config[consts.EVAL_STEPS_INTERVAL]
    if config[consts.EXCLUDED_KEYS]:
        eval_name = filenames.create_excluded_name_fragment()
    else:
        eval_name = None

    evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
        estimator=estimator,
        input_fn=lambda: dataset_provider.eval_input_fn(),
        every_n_iter=eval_steps_interval,
        name=eval_name
    )
    hooks = [evaluator]

    if config[consts.EXCLUDED_KEYS]:
        e = tf.contrib.estimator.InMemoryEvaluatorHook(
            estimator=estimator,
            input_fn=lambda: dataset_provider.eval_with_excludes_input_fn(),
            every_n_iter=eval_steps_interval,
            name='full'
        )
        hooks.append(e)

    estimator.train(
        input_fn=lambda: dataset_provider.train_input_fn(),
        steps=train_steps,
        hooks=hooks
    )


def distributed_train_eval(mnist_estimator):
    for _ in range(4):
        mnist_estimator.train(
            input_fn=lambda: supplying_datasets.train_input_fn(),
            steps=dataset_size // config[consts.BATCH_SIZE] * epochs_between_eval
        )
        if config[consts.EXCLUDED_KEYS]:
            eval_name = filenames.create_excluded_name_fragment()
        else:
            eval_name = None

        eval_results = mnist_estimator.evaluate(input_fn=lambda: supplying_datasets.eval_input_fn(), name=eval_name)
        utils.log('Evaluation results: {}'.format(eval_results))

        if config[consts.EXCLUDED_KEYS]:
            eval_results = mnist_estimator.evaluate(input_fn=lambda: supplying_datasets.eval_with_excludes_fn(),
                                                    name='full')
            utils.log('Evaluation results for whole dataset: {}'.format(eval_results))


def create_estimator(run_data: RunData):
    model = run_data.model
    utils.log('Creating estimator from model: {}'.format(model.summary))
    model_dir = str(filenames.get_run_logs_data_dir(run_data))
    params = model.params
    params[consts.MODEL_DIR] = model_dir
    return tf.estimator.Estimator(
        model_fn=model.get_model_fn(),
        model_dir=model_dir,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1, save_checkpoints_secs=60 * 30),
        params=params
    )


if __name__ == "__main__":
    configuration.define_cli_args()
    tf.app.run()
