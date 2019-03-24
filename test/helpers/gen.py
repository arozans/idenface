from helpers.fake_estimator_model import FakeModel
from src.estimator.launcher.launchers import RunData


def run_data(model=FakeModel(),
             launcher_name="launcher_name",
             runs_directory_name="runs_directory_name",
             is_experiment=False,
             run_no=1,
             models_count=1):
    return RunData(model=model,
                   launcher_name=launcher_name,
                   runs_directory_name=runs_directory_name,
                   is_experiment=is_experiment,
                   run_no=run_no,
                   models_count=models_count)
