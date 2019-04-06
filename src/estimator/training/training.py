import tensorflow as tf

from src.estimator.launcher import providing_launcher
from src.estimator.launcher.launchers import RunData
from src.estimator.training import supplying_datasets
from src.utils import utils, before_run, filenames, consts, configuration
from src.utils.configuration import config

dataset_size = 60 * 1000
epochs_between_eval = 10


def main(args=None):
    launcher = providing_launcher.provide_launcher()
    for run_data in launcher.runs_data:
        before_run.prepare_env(args, run_data)
        train(run_data)
        after_run(run_data)


def train(run_data: RunData):
    estimator = create_estimator(run_data)
    utils.log('Starting train - eval loop excluding data classes: {}'.format(config.excluded_keys))
    in_memory_train_eval(estimator)
    utils.lognl('Finished training with model: {}'.format(run_data.model.summary))


def after_run(run_data: RunData):
    launcher_dir = filenames.get_run_dir(run_data).parent
    utils.log("Inspect results with command: \ntensorboard --logdir={}\n".format(launcher_dir))


def in_memory_train_eval(estimator: tf.estimator.Estimator):
    dataset_provider_cls = estimator.params[consts.DATASET_PROVIDER_CLS]
    dataset_provider = dataset_provider_cls(estimator.params[consts.RAW_DATA_PROVIDER_CLS])
    # dataset_provider = estimator.params[consts.FULL_PROVIDER] #fixme! (can take as an argument...)

    train_steps = config.train_steps
    eval_steps_interval = config.eval_steps_interval
    if config.excluded_keys:
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

    if config.excluded_keys:
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
    for _ in range(4):  # range(flags.train_epochs // epochs_between_eval):  # 500 // 10
        mnist_estimator.train(
            input_fn=lambda: supplying_datasets.train_input_fn(),
            # steps=1000)
            steps=dataset_size // config.batch_size * epochs_between_eval  # 3000
        )  # 300
        # max_steps=150 * 1000)
        # max: 150 000 , batch size: 100, whole dataset size: 60 000
        # so 600 steps per one train epoch
        if config.excluded_keys:
            eval_name = filenames.create_excluded_name_fragment()
        else:
            eval_name = None

        eval_results = mnist_estimator.evaluate(input_fn=lambda: supplying_datasets.eval_input_fn(), name=eval_name)
        utils.log('Evaluation results: {}'.format(eval_results))

        if config.excluded_keys:
            eval_results = mnist_estimator.evaluate(input_fn=lambda: supplying_datasets.eval_with_excludes_fn(),
                                                    name='full')
            utils.log('Evaluation results for whole dataset: {}'.format(eval_results))

        # if eval_results['global_step'] >= flags.max_steps:
        #     break


def create_estimator(run_data: RunData):
    model = run_data.model
    utils.log('Creating estimator from model: {}'.format(model.summary))
    model_dir = str(filenames.get_run_logs_data_dir(run_data))
    print("Estimator created using model_dir: ", model)
    if (filenames.get_run_logs_data_dir(run_data).exists()):
        print("Contests: ", list(filenames.get_run_logs_data_dir(run_data).iterdir()))
    params = model.params
    params[consts.MODEL_DIR] = model_dir  # fixme
    return tf.estimator.Estimator(
        model_fn=model.get_model_fn(),
        model_dir=model_dir,
        params=params
    )


if __name__ == "__main__":
    configuration.define_cli_args()
    tf.app.run()
